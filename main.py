from pprint import pprint

from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save

from src.model import Llama3


def main():
    weights = safe_load("./llama3.2/model.safetensors")
    pprint(Llama3(1, 1).__repr__())
    for k in weights.keys():
        if "norm" in k:
            pprint((k, weights[k]))
            break


if __name__ == "__main__":
    main()
