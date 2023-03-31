from typing import Tuple, Union

from megengine import Tensor
from megengine.module import Conv1d, Conv2d, Module

from .depthwise_conv import DepthwiseConv1d, DepthwiseConv2d


class DepthwiseSeparableConv1d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        conv_mode: str = "cross_correlation",
        compute_mode: str = "default",
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        # megengine don't support groups for conv1d
        raise RuntimeError("Currently do not support DepthwiseSeparableConv1d")
        self.depthwise_conv = DepthwiseConv1d(  # pylint: disable=unreachable
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias,
            conv_mode,
            compute_mode,
            padding_mode,
        )

        self.pointwise_conv = Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out


class DepthwiseSeparableConv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        conv_mode: str = "cross_correlation",
        compute_mode: str = "default",
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.depthwise_conv = DepthwiseConv2d(
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias,
            conv_mode,
            compute_mode,
            padding_mode,
        )

        self.pointwise_conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out
