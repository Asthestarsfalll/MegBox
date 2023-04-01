from functools import reduce

from megengine import Tensor


def all(x: Tensor) -> Tensor:
    x = x.astype("bool")
    return x.sum() == reduce(lambda x, y: x * y, x.shape, 1)


def any(x: Tensor) -> Tensor:
    x = x.astype("bool")
    return x.sum() > 1


def equal(x: Tensor, y: Tensor) -> bool:
    if x.shape != y.shape:
        return False
    return not bool((x != y).sum())


# def allclose(x: Tensor, y: Tensor, rtol=1e-05, atol=1e-08, equal_nan=False):
#     pass
