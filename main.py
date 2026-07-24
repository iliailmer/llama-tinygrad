import argparse
import random
from pathlib import Path

import numpy as np
from tinygrad.tensor import Tensor

from src.generate import generate, load_model
from src.tokenizer import Tokenizer


def set_seed(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    Tensor.manual_seed(seed_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Llama 3.2 inference with tinygrad.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="The capital of the USA is",
        help="Prompt to complete (default: %(default)r).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./llama3.2"),
        help="Directory containing config.json, model.safetensors, and the tokenizer files.",
    )
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate.")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.3,
        help="Penalty applied to already-generated tokens (1.0 disables it).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable the token-generation progress bar."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model, _ = load_model(args.model_dir, verbose=not args.no_progress)
    tokenizer = Tokenizer(args.model_dir)

    text = generate(
        model,
        tokenizer,
        args.prompt,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        show_progress=not args.no_progress,
    )
    print(text)


if __name__ == "__main__":
    main()
