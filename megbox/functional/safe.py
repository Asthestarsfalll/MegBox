import warnings
from typing import Tuple

import megengine.functional as F
from megengine import Tensor

from .logic import any
from .tensor import where


def sort(x: Tensor, descending: bool = False) -> Tuple[Tensor, Tensor]:
    mask = F.isnan(x)
    if any(mask):
        warnings.warn(
            "Input tensor of sort has NaN, it will be converted to Inf", stacklevel=2)
    x = where(mask, float("Inf") if descending else -float("Inf"), x)
    return F.sort(x, descending)
