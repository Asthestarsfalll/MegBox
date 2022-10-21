from collections import OrderedDict
from typing import Sequence, Union

import megengine.module as M
from megengine import Tensor
import megengine.functional as F

from ..module.flatten import Flatten
from .init import _init_weights


class ChannelAttention(M.Sequential):
    def __init__(self, in_channels: int, reduction: int = 16, num_layers=1):
        inner_chan = in_channels // reduction
        modules = [
            ("gap", M.AdaptiveAvgPool2d(1)),
            ("flatten", Flatten(1)),
            ("fc_0", M.Linear(in_channels, inner_chan)),
            ("bn_1", M.BatchNorm1d(inner_chan)),
            ("relu_1", M.ReLU()),
        ]

        for i in range(num_layers):
            modules.extend([
                (f"fc_{i}", M.Linear(in_channels, inner_chan)),
                (f"bn_{i+1}", M.BatchNorm1d(inner_chan)),
                (f"relu_{i+1}", M.ReLU()),
            ])

        modules.append(("fc_final", M.Linear(inner_chan, in_channels)))

        super(ChannelAttention, self).__init__(OrderedDict(modules))


class SpatialAttention(M.Sequential):
    def __init__(self, in_channels: int, reduction: int = 16, dilations: Union[int, Sequence[int]] = 4) -> None:
        inner_chan = in_channels // reduction
        modules = [
            ("conv_reduce", M.Conv2d(in_channels, inner_chan, 1)),
            ("bn_reduce", M.BatchNorm2d(inner_chan)),
            ("relu_reduce", M.ReLU()),
        ]

        if isinstance(dilations, int):
            dilations = [dilations]
        for i, d in enumerate(dilations):
            modules.extend([
                (f"conv_{i}", M.Conv2d(
                    inner_chan, inner_chan, kernel_size=3, padding=d, dilation=d)),
                (f"bn_{i}", M.BatchNorm2d(inner_chan)),
                (f"relu_{i}", M.ReLU()),
            ])

        modules.append((f"conv_final", M.Conv2d(inner_chan, 1, 1)))

        super(SpatialAttention, self).__init__(OrderedDict(modules))


class BAMBlock(M.Module):
    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        dilations: Union[int, Sequence[int]] = 4,
        num_layers: int = 1
    ) -> None:
        super(BAMBlock, self).__init__()
        self.sa = SpatialAttention(in_channels, reduction)
        self.ca = ChannelAttention(in_channels, reduction)

        self.apply(_init_weights)

    def forward(self, x: Tensor) -> Tensor:
        atten = 1 + F.sigmoid(F.expand_dims(self.ca(x), [2, 3]) * self.sa(x))
        return atten * x
