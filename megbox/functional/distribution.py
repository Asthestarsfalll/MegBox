import random
from typing import Optional, Sequence

import megengine as mge
import megengine.functional as F
from megengine import Tensor


def multinomial(x: Tensor, num_samples: int, repalcement: Optional[bool] = None):
    if x.ndim != 2:
        raise ValueError(f"expected input has 2 dimension, but got {x.ndim}")
    if repalcement is not None:
        raise ValueError("Currently not support `replacement`")
    _, num_col = x.shape
    x = F.cumsum(x, axis=1)
    choices = []
    for t in x:
        t = t.numpy()
        ch = []
        for _ in range(num_samples):
            prob = random.random()
            for id in range(num_col):
                if t[id] > prob:
                    idx = id
                    break
            # NOTE: idx is possibly unbound when input cantains NaN
            ch.append(idx)
        choices.append(ch)
    return Tensor(choices, dtype="int32")


def sample_exponential(size: Sequence[int], lambd: float = 1.0, eps: float = 1e-10):
    """
    Generate random numbers from exponential distribution.
    """
    random_tensor = mge.random.uniform(0, 1, size=size)
    return -(1 / lambd) * F.log(random_tensor + eps)
