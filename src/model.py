from tinygrad import Tensor as T
from tinygrad import nn
from tinygrad.nn.state import get_state_dict

from src.configs import LlamaConfig
from src.rope import apply_rope, precompute_freqs_cis


def repeat_kv(x: T, n: int) -> T:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n == 1:
        return x
    return x.repeat((1, 1, 1, n, 1)).reshape(  # NOTE: different from how tg did it
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
        x = self.down_proj(self.up_proj(x).silu() * self.gate_proj(x))
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

        self.cache_k = T.zeros(
            (max_batch_size, max_seq_len, self.n_local_kv_heads, self.head_dim)
        )
        self.cache_v = T.zeros(
            (max_batch_size, max_seq_len, self.n_local_kv_heads, self.head_dim)
        )

    def __call__(self, x: T, start_pos: int, freqs_cis: T, mask: T | None = None) -> T:
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view((batch_size, seq_len, self.n_local_heads, self.head_dim))
        xk = xk.view((batch_size, seq_len, self.n_local_kv_heads, self.head_dim))
        xv = xv.view((batch_size, seq_len, self.n_local_kv_heads, self.head_dim))

        xq, xk = apply_rope(xq, xk, freqs_cis)

        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        keys = repeat_kv(self.cache_k[:batch_size, : start_pos + seq_len], self.n_rep)
        values = repeat_kv(
            self.cache_v[:batch_size, : start_pos + seq_len], n=self.n_rep
        )
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        attn = (
            xq.scaled_dot_product_attention(
                keys, values, mask, enable_gqa=True, is_causal=True
            )
            .transpose(1, 2)
            .reshape((batch_size, seq_len, -1))
        )

        return self.o_proj(attn)


class LlamaTransformer:
    def __init__(self, config: LlamaConfig, max_seq_len: int):
        self.input_layernorm = nn.RMSNorm(config.hidden_size)
        self.mlp = MLP(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size)
        self.self_attn = GQAttn(config, max_seq_len=max_seq_len)

    def __call__(self, x: T, start_pos: int, freqs_cis: T, mask: T | None) -> T:
        h = x + self.self_attn(self.input_layernorm(x), start_pos, freqs_cis, mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Llama3:
    def __init__(self, config: LlamaConfig, max_seq_len: int = 2048):
        pass
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            LlamaTransformer(config, max_seq_len)
            for _ in range(config.num_hidden_layers)
        ]
        self.freqs_cis = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            max_seq_len,
            config.rope_theta,
        )

    def __repr__(self):
        return get_state_dict(self)

    def __call__(self, tokens: T, start_pos: int | None = None) -> T:
        x = self.embed_tokens(tokens)
        return x
