from tinygrad import Tensor as T
from tinygrad import nn
from tinygrad.nn.state import get_state_dict

CONST_LLAMA_LAYERS = 16


class MLP:
    def __init__(self):
        self.up_proj = None
        self.down_proj = None
        self.gate_proj = None


class SelfAttn:
    """GQA"""

    def __init__(self):
        self.k_proj = None
        self.o_proj = None
        self.q_proj = None
        self.v_proj = None


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
