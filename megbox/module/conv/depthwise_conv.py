from typing import Tuple, Union

from megengine.module import Conv1d, Conv2d


class DepthwiseConv1d(Conv1d):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        conv_mode: str = "cross_correlation",
        compute_mode: str = "default",
        padding_mode: str = "zeros",
    ) -> None:
        # megengine don't support groups for conv1d
        raise RuntimeError("Currently do not support DepthwiseSeparableConv1d")
        super().__init__(  # pylint: disable=unreachable
            channels,
            channels,
            kernel_size,
            stride,
            padding,
            dilation,
            channels,
            bias,
            conv_mode,
            compute_mode,
            padding_mode,
        )


class DepthwiseConv2d(Conv2d):
    def __init__(
        self,
        channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        conv_mode: str = "cross_correlation",
        compute_mode: str = "default",
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__(
            channels,
            channels,
            kernel_size,
            stride,
            padding,
            dilation,
            channels,
            bias,
            conv_mode,
            compute_mode,
            padding_mode,
        )
