from typing import Sequence, Tuple

import megengine.functional as F
from megengine import Tensor

from .functional.logic import all, any  # noqa: F401
from .functional.tensor import exchang_axes  # noqa: F401; noqa: F401
from .functional.tensor import pad  # noqa: F401
from .functional.tensor import expand_dims_with_repeat, randn, where
from .module.pooling import AdaptiveAvgPool2d  # noqa: F401; noqa: F401
from .module.pooling import AdaptiveMaxPool2d, AvgPool2d


def broadcast_to(x: Tensor, axis: Sequence[int], shape: Tuple) -> Tensor:
    out = F.expand_dims(x, axis=axis)
    out = F.broadcast_to(out, shape=shape)
    return out
