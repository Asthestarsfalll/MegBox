import megengine.module as M
from megengine import Tensor
from megengine.functional import softmax

from .init import _init_weights


class EABlock(M.Module):
    def __init__(self, channels: int, inner_channels: int = 64) -> None:
        super(EABlock, self).__init__()

        self.conv1 = M.Conv2d(channels, channels, 1)
        self.linear_0 = M.Conv1d(channels, inner_channels, 1, bias=False)
        self.linear_1 = M.Conv1d(inner_channels, channels, 1, bias=False)
        self.conv2 = M.Sequential(
            M.Conv2d(channels, channels, 1, bias=False), M.BatchNorm2d(channels)
        )
        self.relu = M.ReLU()

        self.apply(_init_weights)
        self.linear_1.weight[:] = self.linear_0.weight.transpose(1, 0, 2)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.conv1(x)

        b, c, h, w = x.shape
        h * w
        x = x.reshape(b, c, h * w)  # b * c * n

        attn = self.linear_0(x)  # b, k, n
        attn = softmax(attn, axis=-1)  # b, k, n

        attn = attn / (1e-9 + attn.sum(axis=1, keepdims=True))  # b, k, n
        x = self.linear_1(attn)  # b, c, n

        x = x.reshape(b, c, h, w)
        x = self.conv2(x)
        x = x + identity
        x = self.relu(x)
        return x
