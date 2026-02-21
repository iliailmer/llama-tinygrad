from pathlib import Path
from pprint import pprint
from typing import Literal

from pydantic import BaseModel

from src.utils import read_json


class RopeConfig(BaseModel):
    factor: float
    high_freq_factor: float
    low_freq_factor: float
    original_max_position_embeddings: int
    rope_type: Literal["llama3"]


class LlamaConfig(BaseModel):
    architectures: list[str]
    attention_bias: bool
    attention_dropout: float
    bos_token_id: int
    eos_token_id: int
    head_dim: int
    hidden_act: Literal["silu"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    mlp_bias: bool
    model_type: Literal["llama"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    pretraining_tp: int
    rms_norm_eps: float
    rope_scaling: RopeConfig
    rope_theta: float
    tie_word_embeddings: bool
    torch_dtype: Literal["bfloat16"]
    transformers_version: Literal["4.45.0.dev0"]
    use_cache: bool
    vocab_size: int


def load_config(path: Path):
    json_data = read_json(path)
    json_data["rope_scaling"] = RopeConfig(**json_data["rope_scaling"])
    config = LlamaConfig(**json_data)
    return config
