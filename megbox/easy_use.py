from typing import Sequence, Tuple

import megengine.functional as F
from megengine import Tensor

from .functional.logic import all, any, equal
from .functional.tensor import (exchang_axes, expand_dims_with_repeat, pad,
                                randn, where)
from .module.pooling import (AdaptiveAvgPool2d, AdaptiveMaxPool2d, AvgPool2d,
                             MaxPool2d)

__all__ = [
    "all",
    "any",
    "equal",
    "exchang_axes",
    "pad",
    "expand_dims_with_repeat",
    "randn",
    "where",
    "AdaptiveMaxPool2d",
    "AdaptiveAvgPool2d",
    "AvgPool2d",
    "MaxPool2d",
    "broadcast_to",
]


def broadcast_to(x: Tensor, axis: Sequence[int], shape: Tuple) -> Tensor:
    out = F.expand_dims(x, axis=axis)
    out = F.broadcast_to(out, shape=shape)
    return out
