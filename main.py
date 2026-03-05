import random
from pathlib import Path
from pprint import pprint

import numpy as np
from tinygrad.helpers import tqdm
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save
from tinygrad.tensor import Tensor

from src.configs import load_config
from src.model import Llama3
from src.tokenizer import Tokenizer


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


def main():
    config_path = Path("./llama3.2/config.json")
    config = load_config(config_path)
    weights = safe_load("./llama3.2/model.safetensors")

    # model init
    model = Llama3Wrapper(Llama3(config))
    load_state_dict(model, weights, strict=False)

    # tiktoken
    tokenizer = Tokenizer(Path("./llama3.2/original/tokenizer.model"))

    # sample sentence
    token_ids = tokenizer.encode("The capital of the USA is ")
    tokens = Tensor([token_ids])

    seq_len = tokens.shape[1]
    logits = model(tokens, 0)
    temp = 0.5

    proba = (logits[:, -1, :].flatten() / temp).softmax()
    next_token_id = proba.multinomial()
    max_tokens_gen = 10
    generated = [next_token_id.item()]
    print(generated)
    for i in tqdm(range(1, max_tokens_gen)):
        start_pos = seq_len - 1 + i
        logits = model(Tensor([[generated[-1]]]), start_pos)
        proba = (logits[:, -1, :].flatten() / temp).softmax()
        next_token_id = proba.multinomial()
        generated.append(next_token_id.item())
        if generated[-1] == tokenizer.eos_id:
            break

    print("Generated token IDs:", generated)
    pprint(tokenizer.decode(token_ids + generated))


if __name__ == "__main__":
    main()
