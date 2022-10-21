import megengine as mge
import megengine.module as M
from megengine import Tensor
from megengine.functional import tanh

from ..functional.elementwise import (celu, elu, glu, hardshrink, hardsigmoid,
                                      hardswish, hardtanh, mish, rrelu,
                                      softshrink, softsign, swish, tanhshrink,
                                      threshold)


class CELU(M.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super(CELU, self).__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return celu(x)


class ELU(M.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super(ELU, self).__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return elu(x)


class GLU(M.Module):
    def __init__(self, axis: int = -1) -> None:
        super(GLU, self).__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return glu(x, self.axis)


class HardShrink(M.Module):
    def __init__(self, lambd: float = -1) -> None:
        super(HardShrink, self).__init__()
        self.lambd = lambd

    def forward(self, x: Tensor) -> Tensor:
        return hardshrink(x, self.lambd)


class HardSigmoid(M.Module):
    def forward(self, x: Tensor) -> Tensor:
        return hardsigmoid(x)


class HardSwish(M.Module):
    def forward(self, x: Tensor) -> Tensor:
        return HardSwish(x)


class HardTanH(M.Module):
    def __init__(self, max_value: float, min_value: float) -> None:
        super(HardTanH, self).__init__()
        self.max_value = max_value
        self.min_value = min_value

    def forward(self, x: Tensor) -> Tensor:
        return HardSwish(x)


class Mish(M.Module):
    def __init__(self, max_value: float, min_value: float) -> None:
        super(Mish, self).__init__()
        self.max_value = max_value
        self.min_value = min_value

    def forward(self, x: Tensor) -> Tensor:
        return HardSwish(x)


class TanH(M.Module):
    def forward(self, x: Tensor) -> Tensor:
        return tanh(x)
