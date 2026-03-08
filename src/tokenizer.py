import json
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe

SPLIT_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


DEFAULT_SPECIAL_TOKENS = {
    "<|begin_of_text|>": 128000,
    "<|end_of_text|>": 128001,
    "<|start_header_id|>": 128006,
    "<|end_header_id|>": 128007,
    "<|eot_id|>": 128009,
}


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _resolve_model_dir(model_path_or_dir: Path) -> Path:
    # Accept both:
    # - llama3.2/original/tokenizer.model
    # - llama3.2/
    if model_path_or_dir.is_dir():
        return model_path_or_dir
    if model_path_or_dir.name == "tokenizer.model":
        return model_path_or_dir.parent.parent
    raise ValueError(f"Unsupported tokenizer path: {model_path_or_dir}")


def _load_special_tokens(model_dir: Path = Path("./llama3.2/")) -> dict[str, int]:
    special_tokens = dict(DEFAULT_SPECIAL_TOKENS)

    tokenizer_config_path = model_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        cfg = _read_json(tokenizer_config_path)
        for token_id, token_meta in cfg.get("added_tokens_decoder", {}).items():
            if token_meta.get("special"):
                special_tokens[token_meta["content"]] = int(token_id)

    # Validate special token names from special_tokens_map.json when present.
    special_map_path = model_dir / "special_tokens_map.json"
    if special_map_path.exists():
        special_map = _read_json(special_map_path)
        for key in ("bos_token", "eos_token"):
            value = special_map.get(key)
            content = value.get("content") if isinstance(value, dict) else value
            if isinstance(content, str) and content not in special_tokens:
                raise KeyError(
                    f"Special token {content} from {special_map_path} "
                    "is missing from tokenizer config."
                )

    return special_tokens


class Tokenizer:
    def __init__(self, model_path: Path):
        model_dir = _resolve_model_dir(model_path)
        mergeable_ranks = load_tiktoken_bpe(
            str(model_dir / "original" / "tokenizer.model")
        )
        special_tokens = _load_special_tokens(model_dir)

        self.enc = tiktoken.Encoding(
            name="llama3",
            pat_str=SPLIT_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        self.bos_id = special_tokens["<|begin_of_text|>"]
        self.eos_id = special_tokens["<|end_of_text|>"]
        self.eot_id = special_tokens.get("<|eot_id|>", self.eos_id)

    def encode(self, text: str, bos: bool = True) -> list[int]:
        tokens = self.enc.encode(text, allowed_special="all")
        if bos:
            tokens = [self.bos_id] + tokens
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return self.enc.decode(tokens)
