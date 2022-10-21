from typing import Sequence, Tuple

import megengine.functional as F
from megengine import Tensor

from .functional.logic import all, any
from .functional.tensor import (expand_dims_with_repeat, pad, randn, exchang_axes,
                                where)
from .module.pooling import (AdaptiveAvgPool2D, AdaptiveMaxPool2D, AvgPool2d,
                             MaxPool2d)


def broadcast_to(x: Tensor, axis: Sequence[int], shape: Tuple) -> Tensor:
    out = F.expand_dims(x, axis=axis)
    out = F.broadcast_to(out, shape=shape)
    return out
