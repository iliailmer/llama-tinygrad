from tinygrad.tensor import Tensor


def print_keys(weight_dict: dict[str, Tensor]) -> dict:
    result = {"model": {"layers": {}}}
    for k in weight_dict.keys():
        names = k.split(".")[1:-1]

        if names[0] == "layers":
            pass
        if names[0] == "norm":
            result["model"]["norm"] = weight_dict[k]
        if names[0] == "embed_tokens":
            result["model"]["embed_tokens"] = weight_dict[k]
    return result
