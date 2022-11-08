from functools import reduce

from megengine import Tensor


def all(x: Tensor) -> Tensor:
    return x.sum() == reduce(lambda x, y: x * y, x.shape, 1)


def any(x: Tensor) -> Tensor:
    return x.sum() > 1


def allclose(x: Tensor, y: Tensor, rtol=1e-05, atol=1e-08, equal_nan=False):
    pass
