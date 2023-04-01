import functools

from megengine import Tensor

from megbox.types import Number


def handle_number(value: Number) -> Tensor:
    if isinstance(value, float):
        return Tensor(value, dtype="float32")
    elif isinstance(value, int):
        return Tensor(value, dtype="int32")
    return value


def handle_negtive_aixs(axis: int, ndim: int) -> int:
    if axis < 0:
        return ndim + axis
    return axis


def index_cast(transform, idx):
    """Use as a decorator"""

    def _index_cast(func):
        @functools.wraps(func)
        def wrapper_func(*args, **kwargs):
            if len(args) < max(idx) + 1:
                raise RuntimeError()
            args = list(args)
            for i in idx:
                args[i] = transform(args[i])
            return func(*args, **kwargs)

        return wrapper_func

    return _index_cast


def assert_no_kwargs(func):
    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):
        if kwargs:
            raise RuntimeError("Expected no keyword arguments")
        return func(*args, **kwargs)

    return wrapper_func
