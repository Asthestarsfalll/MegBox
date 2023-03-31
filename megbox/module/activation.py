import megengine.module as M
from megengine import Tensor
from megengine.functional import tanh

from ..functional.elementwise import (celu, elu, glu, hardshrink, hardsigmoid,
                                      hardswish, hardtanh, mish, rrelu,
                                      softshrink, softsign, swish, tanhshrink)
from ..functional.elementwise import threshold as f_threshold


class CELU(M.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return celu(x)


class ELU(M.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return elu(x)


class GLU(M.Module):
    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return glu(x, self.axis)


class HardShrink(M.Module):
    def __init__(self, lambd: float = -1) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: Tensor) -> Tensor:
        return hardshrink(x, self.lambd)


class HardSigmoid(M.Module):
    def forward(self, x: Tensor) -> Tensor:
        return hardsigmoid(x)


class HardSwish(M.Module):
    def forward(self, x: Tensor) -> Tensor:
        return hardswish(x)


class HardTanH(M.Module):
    def __init__(self, max_value: float = -1.0, min_value: float = 1.0) -> None:
        super().__init__()
        self.max_value = max_value
        self.min_value = min_value

    def forward(self, x: Tensor) -> Tensor:
        return hardtanh(x, self.max_value, self.min_value)


class Mish(M.Module):
    def forward(self, x: Tensor) -> Tensor:
        return mish(x)


class RReLU(M.Module):
    def __init__(self, lower: float = 1 / 8, upper: float = 1 / 3) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper

    def forward(self, x: Tensor) -> Tensor:
        return rrelu(x)


class Swish(M.Module):
    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, x: Tensor) -> Tensor:
        return swish(x, self.beta)


class Softshrink(M.Module):
    def __init__(self, lambd: float) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: Tensor) -> Tensor:
        return softshrink(x, self.lambd)


class Softsign(M.Module):
    def forward(self, x: Tensor) -> Tensor:
        return softsign(x)


class Tanh(M.Module):
    def forward(self, x: Tensor) -> Tensor:
        return tanh(x)


class Tanhshrink(M.Module):
    def forward(self, x: Tensor) -> Tensor:
        return tanhshrink(x)


class Threshold(M.Module):
    def __init__(self, threshold: float, value: float) -> None:
        super().__init__()
        self.threshold = threshold
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        return f_threshold(x, self.threshold, self.value)
