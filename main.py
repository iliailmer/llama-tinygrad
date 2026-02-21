from pathlib import Path
from pprint import pprint

from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save
from tinygrad.tensor import Tensor

from src.configs import load_config
from src.model import Llama3


class Llama3Wrapper:
    def __init__(self, model: Llama3):
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def main():
    config_path = Path("./llama3.2/config.json")
    config = load_config(config_path)
    weights = safe_load("./llama3.2/model.safetensors")
    model = Llama3Wrapper(Llama3(config))
    load_state_dict(model, weights, strict=False)

    print(model(Tensor([1, 2, 3])))


if __name__ == "__main__":
    main()
