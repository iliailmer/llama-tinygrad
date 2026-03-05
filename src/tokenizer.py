import tiktoken
from tiktoken.load import load_tiktoken_bpe
from pathlib import Path

SPLIT_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

SPECIAL_TOKENS = {
    "<|begin_of_text|>": 128000,
    "<|end_of_text|>":   128001,
    "<|start_header_id|>": 128006,
    "<|end_header_id|>":   128007,
    "<|eot_id|>":          128009,
}


class Tokenizer:
    def __init__(self, model_path: Path):
        mergeable_ranks = load_tiktoken_bpe(str(model_path))
        self.enc = tiktoken.Encoding(
            name="llama3",
            pat_str=SPLIT_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=SPECIAL_TOKENS,
        )
        self.bos_id = SPECIAL_TOKENS["<|begin_of_text|>"]
        self.eos_id = SPECIAL_TOKENS["<|end_of_text|>"]

    def encode(self, text: str, bos: bool = True) -> list[int]:
        tokens = self.enc.encode(text, allowed_special="all")
        if bos:
            tokens = [self.bos_id] + tokens
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return self.enc.decode(tokens)
