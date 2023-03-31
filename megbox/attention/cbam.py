import megengine.functional as F
import megengine.module as M
from megengine import Tensor

from .init import _init_weights


class ChannelAttention(M.Module):
    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        inner_chan = in_channels // reduction
        self.max_pool = M.AdaptiveMaxPool2d(1)
        self.avg_pool = M.AdaptiveAvgPool2d(1)
        self.mlp = M.Sequential(
            # According official repo, set bias=True
            M.Conv2d(in_channels, inner_chan, 1, bias=True),
            M.ReLU(),
            M.Conv2d(inner_chan, in_channels, 1, bias=True),
        )
        self.sigmoid = M.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        return self.sigmoid(max_out + avg_out)


class SpatialAttention(M.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = M.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = M.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        max_res = x.max(1, True)
        avg_res = x.mean(1, True)
        concat_tensor = F.concat([max_res, avg_res], axis=1)
        return self.sigmoid(self.conv(concat_tensor))


class CBAMBlock(M.Module):
    def __init__(
        self, in_channels: int, reduction: int = 16, kernel_size: int = 7
    ) -> None:
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

        self.apply(_init_weights)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + identity
