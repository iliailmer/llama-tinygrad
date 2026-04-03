import random
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from tinygrad import dtypes
from tinygrad.helpers import tqdm
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save
from tinygrad.tensor import Tensor

from src.configs import load_config
from src.model import Llama3
from src.tokenizer import Tokenizer


def fix_bf16(weights: dict[Any, Tensor]) -> dict[Any, Tensor]:
    return {
        k: v.cast(dtypes.float32) if v.dtype == dtypes.bfloat16 else v
        for k, v in weights.items()
    }


def set_seed(seed_value):
    # Set seed for standard python random module
    random.seed(seed_value)

    # Set seed for numpy
    np.random.seed(seed_value)

    # Set seed for tinygrad
    Tensor.manual_seed(seed_value)


set_seed(42)


class Llama3Wrapper:
    def __init__(self, model: Llama3):
        self.model = model

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.model(*args, **kwargs)


def sample_top_p(
    logits: Tensor,
    temp: float,
    top_p: float,
    history: list[int],
    repetition_penalty: float,
) -> int:
    adjusted = logits.float().numpy().copy()
    if repetition_penalty > 1.0 and history:
        for token_id in set(history):
            if adjusted[token_id] > 0:
                adjusted[token_id] /= repetition_penalty
            else:
                adjusted[token_id] *= repetition_penalty

    adjusted = adjusted / temp
    adjusted = adjusted - np.max(adjusted)
    probs = np.exp(adjusted)
    probs = probs / probs.sum()

    sorted_idx = np.argsort(-probs)
    sorted_probs = probs[sorted_idx]
    cdf = np.cumsum(sorted_probs)

    cutoff = np.searchsorted(cdf, top_p, side="left") + 1
    kept_idx = sorted_idx[:cutoff]
    kept_probs = probs[kept_idx]
    kept_probs = kept_probs / kept_probs.sum()
    return int(np.random.choice(kept_idx, p=kept_probs))


def main():
    config_path = Path("./llama3.2/config.json")
    config = load_config(config_path)
    weights = safe_load("./llama3.2/model.safetensors")

    # HuggingFace stores Q and K weights in a "half-half" RoPE interleaving,
    # but this model applies RoPE with consecutive pairs. Permute to match.
    def permute(v: Tensor, n_heads: int) -> Tensor:
        return (
            v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1])
            .transpose(1, 2)
            .reshape(*v.shape[:2])
        )

    for lid in range(config.num_hidden_layers):
        q_key = f"model.layers.{lid}.self_attn.q_proj.weight"
        k_key = f"model.layers.{lid}.self_attn.k_proj.weight"
        weights[q_key] = permute(weights[q_key], config.num_attention_heads)
        weights[k_key] = permute(weights[k_key], config.num_key_value_heads)

    # model init
    model = Llama3Wrapper(Llama3(config))
    load_state_dict(model, weights, strict=False)

    # tiktoken
    tokenizer = Tokenizer(Path("./llama3.2/"))

    # sample sentence
    token_ids = tokenizer.encode("The capital of the USA is ")
    tokens = Tensor([token_ids])

    seq_len = tokens.shape[1]
    logger.info("Sequence Length {}", seq_len)
    logits = model(tokens, 0)
    temp = 0.6
    top_p = 0.9
    repetition_penalty = 1.1

    next_token_id = sample_top_p(
        logits[:, -1, :].flatten(),
        temp=temp,
        top_p=top_p,
        history=token_ids,
        repetition_penalty=repetition_penalty,
    )
    max_tokens_gen = 256
    generated = [next_token_id]
    print(generated)
    for i in tqdm(range(1, max_tokens_gen)):
        start_pos = seq_len - 1 + i
        logits = model(Tensor([[generated[-1]]]), start_pos)
        next_token_id = sample_top_p(
            logits[:, -1, :].flatten(),
            temp=temp,
            top_p=top_p,
            history=token_ids + generated,
            repetition_penalty=repetition_penalty,
        )
        generated.append(next_token_id)
        if generated[-1] in (tokenizer.eos_id, tokenizer.eot_id):
            break

    print(tokenizer.decode(token_ids + generated))


if __name__ == "__main__":
    main()
