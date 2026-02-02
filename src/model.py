from tinygrad import Tensor as T
from tinygrad import nn
from tinygrad.nn.state import get_state_dict

CONST_LLAMA_LAYERS = 16


class MLP:
    def __init__(self):
        self.up_proj = nn.Linear(2048, 8192)  # w1
        self.down_proj = nn.Linear(8192, 2048)  # w2
        self.gate_proj = nn.Linear(8192, 2048)  # w3

    def forward(self, x: T) -> T:
        x = self.down_proj(self.up_proj(x).silu() * self.gate_proj(x))
        return x


class SelfAttn:
    """GQA"""

    def __init__(self):
        self.k_proj = None
        self.o_proj = None
        self.q_proj = None
        self.v_proj = None

    def forward(self, x: T) -> T:
        return x


class LlamaLayer:
    def __init__(self, input_shape: int, output_shape: int):
        self.input_layernorm = nn.LayerNorm(input_shape)
        self.mlp = MLP()
        self.post_attention_layernorm = nn.LayerNorm(output_shape)
        self.self_attn = SelfAttn()


class Llama3:
    def __init__(self, vocab_size: int, embed_size: int):
        pass
        self.embed_tokens = nn.Embedding(vocab_size, embed_size)
        self.layers = [
            LlamaLayer(embed_size, embed_size) for _ in range(CONST_LLAMA_LAYERS)
        ]

    def __repr__(self):
        return get_state_dict(self)

    def forward(self, x: T) -> T:
        return x
