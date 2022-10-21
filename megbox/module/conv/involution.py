from megengine.module import Module, ConvBnRelu2d, Conv2d, AvgPool2d, SlidingWindow, Identity
from megengine import Tensor
from megengine.functional import expand_dims

# TODO: add cuda implementation


class Involution(Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int,
        dilation: int = 1,
        reduction: int = 4,
        group_channels: int = 16,
    ) -> None:
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.group_channels = group_channels
        self.groups = channels // group_channels

        inner_chan = channels // reduction

        self.conv1 = ConvBnRelu2d(channels, inner_chan, 1)
        self.conv2 = Conv2d(inner_chan, kernel_size**2*self.groups, 1)
        self.avgpool = AvgPool2d(stride, stride) if stride > 1 else Identity()
        self.slide_window = SlidingWindow(
            kernel_size,
            padding=dilation*(kernel_size - 1) // 2,
            stride=stride,
            dilation=dilation
        )

    def forward(self, x: Tensor) -> Tensor:
        weight = self.conv2(self.conv1(self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.reshape(b, self.groups, self.kernel_size**2, h, w)
        weight = expand_dims(weight, 2)

        out = self.slide_window(x).reshape(
            b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(axis=3)
        out = out.reshape(b, -1, h, w)
        return out
