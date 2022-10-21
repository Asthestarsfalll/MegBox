import megengine.module as M
from megengine import Tensor

from .init import _init_weights


class SEBlock(M.Module):
    def __init__(self, in_channels: int, reduction: int) -> None:
        super(SEBlock, self).__init__()
        out_channels = in_channels // reduction
        self.gap = M.AdaptiveAvgPool2d(1)
        self.fc1 = M.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = M.ReLU()
        self.fc2 = M.Conv2d(out_channels, channels, kernel_size=1)
        self.sigmoid = M.Sigmoid()

        self.apply(_init_weights)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.gap(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return identity * x
