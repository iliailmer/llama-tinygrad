from tinygrad import Tensor as T
from tinygrad import nn
from tinygrad.nn.state import get_state_dict

from rope import apply_rope

CONST_LLAMA_LAYERS = 16


def repeat_kv(x: T, n: int) -> T:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n == 1:
        return x
    return x.repeat((1, 1, 1, n, 1)).reshape(  # NOTE: different from how tg did it
        (batch_size, seq_len, n_kv_heads * n, head_dim)
    )


class MLP:
    def __init__(self):
        self.up_proj = nn.Linear(2048, 8192)  # w1
        self.down_proj = nn.Linear(8192, 2048)  # w2
        self.gate_proj = nn.Linear(8192, 2048)  # w3

    def __call__(self, x: T) -> T:
        x = self.down_proj(self.up_proj(x).silu() * self.gate_proj(x))
        return x


class GQAttn:
    """GQA"""

    def __init__(
        self,
        n_kv_heads: int,
        n_heads: int,
        dim: int,
        max_batch_size: int = 1,
        max_seq_len: int = 512,
        n_local_kv_heads: int = 4,
    ):
        """
        dim: int, size of token embedding
        """
        self.n_kv_heads = n_kv_heads  # total kv heads
        self.n_local_heads = n_heads  # total heads, per model (if model-parallel)
        self.n_local_kv_heads = (
            n_local_kv_heads  # total kv heads per model (if model-parallel)
        )
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads  # single head dim

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.head_dim * self.n_kv_heads, bias=False)
        self.v_proj = nn.Linear(dim, self.head_dim * self.n_kv_heads, bias=False)
        self.o_proj = nn.Linear(dim, self.head_dim * self.n_kv_heads, bias=False)

        self.cache_k = T.zeros(
            (max_batch_size, max_seq_len, n_local_kv_heads, self.head_dim)
        )
        self.cache_v = T.zeros(
            (max_batch_size, max_seq_len, n_local_kv_heads, self.head_dim)
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
    def __init__(self, input_shape: int, output_shape: int):
        self.input_layernorm = nn.LayerNorm(input_shape)
        self.mlp = MLP()
        self.post_attention_layernorm = nn.LayerNorm(output_shape)
        self.self_attn = GQAttn(1, 1)


class Llama3:
    def __init__(self, vocab_size: int, embed_size: int):
        pass
        self.embed_tokens = nn.Embedding(vocab_size, embed_size)
        self.layers = [
            LlamaLayer(embed_size, embed_size) for _ in range(CONST_LLAMA_LAYERS)
        ]

    def __repr__(self):
        return get_state_dict(self)

    def __call__(self, x: T) -> T:
        return x
