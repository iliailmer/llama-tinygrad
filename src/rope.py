from loguru import logger
from tinygrad import Tensor as T


@logger.catch
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> T:
    freqs = 1.0 / (theta ** (T.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = T.arange(end)
    freqs = T.einsum("i,j->ij", t, freqs)
    return T.stack(freqs.cos(), freqs.sin(), dim=-1).reshape(1, end, 1, dim // 2, 2)


@logger.catch
def complex_mult(A, c, d):
    a, b = A[..., 0:1], A[..., 1:2]
    ro = a * c - b * d
    co = a * d + b * c
    return ro.cat(co, dim=-1)


@logger.catch
def apply_rope(xq: T, xk: T, freqs: T) -> tuple[T, T]:
    assert freqs.shape[1] == xq.shape[1] == xk.shape[1], (
        f"freqs shape mismatch {freqs.shape} xq:{xq.shape} xk:{xk.shape}"
    )
    xq = xq.reshape(*xq.shape[0:-1], -1, 2)  # make it complex type
    xk = xk.reshape(*xk.shape[0:-1], -1, 2)
    assert len(xq.shape) == len(xk.shape) == len(freqs.shape) == 5
    c, d = freqs[..., 0:1], freqs[..., 1:2]
    xq_out = complex_mult(xq, c, d)
    xk_out = complex_mult(xk, c, d)
    return xq_out.flatten(3), xk_out.flatten(3)
