import math
from typing import Callable, Optional, Tuple, Union

import megengine.functional as F
import megengine.module as M
from megengine import Tensor
from megengine.module.pooling import AvgPool2d as MgeAvgPool2d
from megengine.module.pooling import MaxPool2d as MgeMaxPool2d
from megengine.module.pooling import _PoolNd

from ..functional.tensor import pad
from ..types import pooling_type

__all__ = [
    "AdaptiveAvgPool2D",
    "AdaptiveMaxPool2D",
    "AvgPool2d",
    "MaxPool2d",
]


def _calculate_pad_size(
    h: int, w: int, padding: int, ks: int, stride: int
) -> Tuple[int, int]:
    out_h = math.ceil((h + 2 * padding - ks) / stride + 1)
    out_w = math.ceil((w + 2 * padding - ks) / stride + 1)
    pad_h = (out_h - 1) * stride + ks - 2 * padding - h
    pad_w = (out_w - 1) * stride + ks - 2 * padding - w
    return (pad_h, pad_w)


def with_ceil_mode(pool: pooling_type, **kwargs) -> _PoolNd:
    module = pool(**kwargs)

    def _process_ceil_mode(self, x):
        x = x[0]
        h, w = x.shape[2:]
        pad_h, pad_w = _calculate_pad_size(
            h, w, self.padding, self.kernel_size, self.stride
        )
        return pad(x, (0, pad_w, 0, pad_h))

    module.register_forward_pre_hook(_process_ceil_mode)

    return module


def MaxPool2d(
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    ceil_mode: bool = False,
) -> _PoolNd:
    if ceil_mode:
        return with_ceil_mode(
            MgeMaxPool2d, kernel_size=kernel_size, stride=stride, padding=padding
        )
    else:
        return MgeAvgPool2d(kernel_size, stride, padding)


def AvgPool2d(
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    ceil_mode: bool = False,
) -> _PoolNd:
    if ceil_mode:
        return with_ceil_mode(
            MgeAvgPool2d, kernel_size=kernel_size, stride=stride, padding=padding
        )
    else:
        return MgeAvgPool2d(kernel_size, stride, padding)


# TODO: refactor and implement it with CPP CUDA.
class AdaptivePool2D(M.Module):
    def __init__(
        self,
        oshp: Union[int, Tuple[int, int]],
        func: Callable,
    ) -> None:
        super(AdaptivePool2D, self).__init__()
        if isinstance(oshp, int):
            oshp = (oshp, oshp)
        self.oshp = oshp
        self.func = func

    def _calculate_kernel_size(self, ishp):
        kh = (ishp[0] + self.oshp[0] - 1) // self.oshp[0]
        kw = (ishp[1] + self.oshp[1] - 1) // self.oshp[1]
        return (kh, kw)

    @staticmethod
    def _zip(*x):
        element_length = len(x[0])
        for i in x:
            assert element_length == len(i)
        out = []
        total_length = len(x)
        for i in range(element_length):
            temp = []
            for j in range(total_length):
                temp.append(x[j][i])
            out.append(temp)
        return out

    def _get_points(self, input_size, kernel_size):
        start_points_h = (
            F.arange(self.oshp[0], dtype="float32") * (input_size[0] / self.oshp[0])
        ).astype("int32")
        end_points_h = F.ceil(
            (
                (F.arange(self.oshp[0], dtype="float32") + 1)
                * (input_size[0] / self.oshp[0])
            )
        ).astype("int32")
        start_points_w = (
            F.arange(self.oshp[1], dtype="float32") * input_size[1] / self.oshp[1]
        ).astype("int32")
        end_points_w = F.ceil(
            (
                (F.arange(self.oshp[1], dtype="float32") + 1)
                * (input_size[1] / self.oshp[1])
            )
        ).astype("int32")
        return self._zip(start_points_h, end_points_h), self._zip(
            start_points_w, end_points_w
        )

    def _get_windows(self, inp, coords):
        windows = []
        for h_s, h_e in coords[0]:
            for w_s, w_e in coords[1]:
                windows.append(self.func(inp[:, :, h_s:h_e, w_s:w_e], axis=(2, 3)))
        windows = F.stack(windows, -1)
        return windows

    def forward(self, inputs: Tensor) -> Tensor:
        assert inputs.ndim == 4, "Currently only support 4D input"
        ishp = inputs.shape[-2:]
        kernel_size = self._calculate_kernel_size(ishp)
        point_h, point_w = self._get_points(ishp, kernel_size)
        windows = self._get_windows(inputs, (point_h, point_w))
        return windows.reshape(*windows.shape[:2], *self.oshp)


def AdaptiveAvgPool2D(
    oshp: Union[int, Tuple[int, int]],
) -> AdaptivePool2D:
    return AdaptivePool2D(oshp, F.mean)


def AdaptiveMaxPool2D(
    oshp: Union[int, Tuple[int, int]],
) -> AdaptivePool2D:
    return AdaptivePool2D(oshp, F.max)
