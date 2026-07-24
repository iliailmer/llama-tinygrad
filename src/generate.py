from pathlib import Path
from typing import Any

import numpy as np
from tinygrad import Device, Variable, dtypes
from tinygrad.helpers import tqdm
from tinygrad.nn.state import load_state_dict, safe_load
from tinygrad.tensor import Tensor

from src.configs import LlamaConfig, load_config
from src.model import Llama3
from src.tokenizer import Tokenizer


class Llama3Wrapper:
    def __init__(self, model: Llama3):
        self.model = model

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.model(*args, **kwargs)


def fix_bf16(weights: dict[Any, Tensor]) -> dict[Any, Tensor]:
    return {
        k: v.to(Device.DEFAULT).cast(dtypes.float32)
        if v.dtype == dtypes.bfloat16
        else v
        for k, v in weights.items()
    }


def permute_qk_weights(v: Tensor, n_heads: int) -> Tensor:
    """HuggingFace stores Q and K weights in a "half-half" RoPE interleaving,
    but this model applies RoPE with consecutive pairs. Permute to match."""
    return (
        v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1])
        .transpose(1, 2)
        .reshape(*v.shape[:2])
    )


def load_model(model_dir: Path, verbose: bool = True) -> tuple[Llama3Wrapper, LlamaConfig]:
    config = load_config(model_dir / "config.json")
    weights = safe_load(str(model_dir / "model.safetensors"))
    weights = fix_bf16(weights)

    for lid in range(config.num_hidden_layers):
        q_key = f"model.layers.{lid}.self_attn.q_proj.weight"
        k_key = f"model.layers.{lid}.self_attn.k_proj.weight"
        weights[q_key] = permute_qk_weights(weights[q_key], config.num_attention_heads)
        weights[k_key] = permute_qk_weights(
            weights[k_key], config.num_key_value_heads
        )

    model = Llama3Wrapper(Llama3(config))
    load_state_dict(model, weights, strict=False, verbose=verbose)
    return model, config


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


def generate(
    model: Llama3Wrapper,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temp: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3,
    show_progress: bool = True,
) -> str:
    token_ids = tokenizer.encode(prompt)
    tokens = Tensor([token_ids])
    seq_len = tokens.shape[1]

    logits = model(tokens, 0)
    next_token_id = sample_top_p(
        logits[:, -1, :].flatten(),
        temp=temp,
        top_p=top_p,
        history=token_ids,
        repetition_penalty=repetition_penalty,
    )
    generated = [next_token_id]

    max_context = model.model.max_seq_len
    steps = range(1, max_tokens)
    if show_progress:
        steps = tqdm(steps)
    for i in steps:
        start_pos = seq_len - 1 + i
        logits = model.model.decode_step(
            Tensor([[generated[-1]]]),
            Variable("start_pos", 1, max_context).bind(start_pos),
        )
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

    return tokenizer.decode(token_ids + generated, skip_special=True)
