import logging
import math
from typing import Callable, Optional, Tuple, Union

import megengine.functional as F
import megengine.module as M
from megengine import Tensor
from megengine.module.pooling import AvgPool2d as MgeAvgPool2d
from megengine.module.pooling import MaxPool2d as MgeMaxPool2d
from megengine.module.pooling import _PoolNd

from megbox.functional.tensor import pad
from megbox.types import pooling_type
from megbox.utils.msic import borrow_doc

__all__ = [
    "AdaptiveAvgPool2d",
    "AdaptiveMaxPool2d",
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


def with_ceil_mode(pool: pooling_type, value: float, **kwargs) -> _PoolNd:
    module = pool(**kwargs)
    mode = "reflect" if value == 0 else "constant"
    if value == 0:
        logging.warning(
            "The implementation of `ceil_mode` is approximate, may cause some potential problems"
        )

    def _process_ceil_mode(self, x):
        x = x[0]
        h, w = x.shape[2:]
        pad_h, pad_w = _calculate_pad_size(
            h, w, self.padding, self.kernel_size, self.stride
        )
        return pad(x, (0, pad_w, 0, pad_h), constant_value=value, mode=mode)

    module.register_forward_pre_hook(_process_ceil_mode)

    return module


@borrow_doc(MgeMaxPool2d)
def MaxPool2d(
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    ceil_mode: bool = False,
) -> _PoolNd:
    if ceil_mode:
        return with_ceil_mode(
            MgeMaxPool2d,
            value=-float("inf"),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
    else:
        return MgeMaxPool2d(kernel_size, stride, padding)


@borrow_doc(MgeAvgPool2d)
def AvgPool2d(
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    ceil_mode: bool = False,
) -> _PoolNd:
    if ceil_mode:
        return with_ceil_mode(
            MgeAvgPool2d,
            value=0,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
    else:
        return MgeAvgPool2d(kernel_size, stride, padding)


# TODO: refactor and implement it with CPP CUDA.
class AdaptivePool2d(M.Module):
    def __init__(
        self,
        oshp: Union[int, Tuple[int, int]],
        func: Callable,
    ) -> None:
        super().__init__()
        if isinstance(oshp, int):
            oshp = (oshp, oshp)
        self.oshp = oshp
        self.func = func

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

    def _get_points(self, input_size):
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
        point_h, point_w = self._get_points(ishp)
        windows = self._get_windows(inputs, (point_h, point_w))
        return windows.reshape(*windows.shape[:2], *self.oshp)


@borrow_doc(M.AdaptiveAvgPool2d)
def AdaptiveAvgPool2d(
    oshp: Union[int, Tuple[int, int]],
) -> AdaptivePool2d:
    return AdaptivePool2d(oshp, F.mean)


@borrow_doc(M.AdaptiveMaxPool2d)
def AdaptiveMaxPool2d(
    oshp: Union[int, Tuple[int, int]],
) -> AdaptivePool2d:
    return AdaptivePool2d(oshp, F.max)
