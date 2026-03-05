from tinygrad import Tensor as T
from tinygrad import nn
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.state import get_state_dict

from src.configs import LlamaConfig
from src.rope import apply_rope, precompute_freqs_cis


def repeat_kv(x: T, n: int) -> T:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n == 1:
        return x
    return x.repeat((1, 1, 1, n)).reshape(  # NOTE: different from how tg did it
        (batch_size, seq_len, n_kv_heads * n, head_dim)
    )


class MLP:
    def __init__(self, config: LlamaConfig):
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )  # w1
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )  # w2
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )  # w3

    def __call__(self, x: T) -> T:
        x = self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))
        return x


class GQAttn:
    """Group Query Attention Layer
    This is corresponding to the Attention class in llama3 repo
    """

    def __init__(
        self, config: LlamaConfig, max_batch_size: int = 32, max_seq_len: int = 2048
    ):
        """
        config: LlamaConfig
        """
        self.n_kv_heads = config.num_key_value_heads  # total kv heads
        self.n_local_heads = (
            config.num_attention_heads
        )  # total heads, per model (if model-parallel)
        self.n_local_kv_heads = (
            config.num_key_value_heads  # total kv heads per model (if model-parallel)
        )
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim  # single head dim
        dim = config.hidden_size

        self.q_proj = nn.Linear(dim, dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            dim, self.head_dim * self.n_kv_heads, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, self.head_dim * self.n_kv_heads, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.n_local_heads * self.head_dim,
            dim,
            bias=config.attention_bias,
        )

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.cache_k = (
            T.zeros((max_batch_size, max_seq_len, self.n_local_kv_heads, self.head_dim))
            .requires_grad_(False)
            .realize()
        )
        self.cache_v = (
            T.zeros((max_batch_size, max_seq_len, self.n_local_kv_heads, self.head_dim))
            .requires_grad_(False)
            .realize()
        )

    def __call__(self, x: T, start_pos: int, freqs_cis: T, mask: T | None = None) -> T:
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view((batch_size, seq_len, self.n_local_heads, self.head_dim))
        xk = xk.view((batch_size, seq_len, self.n_local_kv_heads, self.head_dim))
        xv = xv.view((batch_size, seq_len, self.n_local_kv_heads, self.head_dim))

        xq, xk = apply_rope(xq, xk, freqs_cis)

        if not hasattr(self, "kv_cache"):
            self.cache_kv = (
                T.zeros(
                    2,
                    batch_size,
                    self.max_seq_len,
                    self.n_kv_heads,
                    self.head_dim,
                    dtype=x.dtype,
                )
                .contiguous()
                .realize()
            )
        self.cache_kv[:, :, start_pos : start_pos + seq_len, :, :].assign(
            T.stack(xk, xv)
        ).realize()

        keys = self.cache_kv[0, :, 0 : start_pos + seq_len, :, :]
        values = self.cache_kv[1, :, 0 : start_pos + seq_len, :, :]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, n=self.n_rep)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        attn = (
            xq.scaled_dot_product_attention(
                keys,
                values,
                mask,
            )
            .transpose(1, 2)
            .reshape((batch_size, seq_len, -1))
        )

        return self.o_proj(attn)


class LlamaTransformer:
    def __init__(self, config: LlamaConfig, max_seq_len: int):
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = GQAttn(config, max_seq_len=max_seq_len)

    def __call__(self, x: T, start_pos: int, freqs_cis: T, mask: T | None) -> T:
        h = x + self.self_attn(self.input_layernorm(x), start_pos, freqs_cis, mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Llama3:
    def __init__(self, config: LlamaConfig, max_seq_len: int = 2048):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            LlamaTransformer(config, max_seq_len)
            for _ in range(config.num_hidden_layers)
        ]
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.tie_word_embeddings:
            self.output.weight = self.embed_tokens.weight
        rope = config.rope_scaling
        self.freqs_cis = (
            precompute_freqs_cis(
                config.head_dim,
                max_seq_len,
                config.rope_theta,
            )
            .contiguous()
            .requires_grad_(False)
        )
        self.max_seq_len = max_seq_len
        self.forward_jit = self.forward

    def __repr__(self):
        return get_state_dict(self)

    def __call__(self, tokens: T, start_pos: int) -> T:
        return self.forward_jit(tokens, start_pos)

    def forward(self, tokens: T, start_pos: int) -> T:
        batch_size, seq_len = tokens.shape
        x = self.embed_tokens(tokens).contiguous()
        freqs_cis = self.freqs_cis.cast(x.dtype)[
            :, start_pos : start_pos + seq_len, :, :, :
        ]

        mask = None
        if seq_len > 1 and self.max_seq_len != 0:
            mask = (
                T.full(
                    (1, 1, seq_len, start_pos + seq_len),
                    float("-inf"),
                    dtype=x.dtype,
                    device=x.device,
                )
                .triu(start_pos + 1)
                .contiguous()
            )

        for layer in self.layers:
            x = layer(x, start_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)

        return logits
