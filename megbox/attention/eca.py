import megengine.module as M
from megengine import Tensor
from megengine.functional import expand_dims, squeeze

from ..functional import exchang_axes
from .init import _init_weights


class ECABlock(M.Module):
    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        self.gap = M.AdaptiveAvgPool2d(1)
        self.conv = M.Conv1d(
            1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = M.Sigmoid()

        self.apply(_init_weights)

    def forward(self, x: Tensor) -> Tensor:
        atten = self.gap(x)
        # B, C, 1, 1 -> B, 1, C
        atten = exchang_axes(squeeze(atten, -1), -1, -2)
        atten = self.sigmoid(self.conv(atten))
        # B, 1, C -> B, C, 1, 1
        return x * expand_dims(exchang_axes(atten, -1, -2), 3)
