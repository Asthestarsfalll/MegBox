import math
from typing import Optional

import megengine.module as M
from megengine import Tensor
from megengine.functional import sliding_window_transpose, softmax

from ..module.pooling import AvgPool2d
from .init import _init_weights


class OutlookAttention(M.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super(OutlookAttention, self).__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = qk_scale or head_dim**-0.5

        self.v = M.Linear(dim, dim, bias=qkv_bias)
        self.attn = M.Linear(dim, kernel_size**4 * num_heads)

        self.attn_drop = M.Dropout(attn_drop)
        self.proj = M.Linear(dim, dim)
        self.proj_drop = M.Dropout(proj_drop)

        self.unfold = M.SlidingWindow(
            kernel_size=kernel_size, padding=padding, stride=stride
        )
        self.pool = AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.apply(_init_weights)

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape

        v = self.v(x).transpose(0, 3, 1, 2)  # B, C, H, W

        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = (
            self.unfold(v)
            .reshape(
                B,
                self.num_heads,
                C // self.num_heads,
                self.kernel_size * self.kernel_size,
                h * w,
            )
            .transpose(0, 1, 4, 3, 2)
        )  # B,H,N,kxk,C/H

        attn = self.pool(x.transpose(0, 3, 1, 2)).transpose(0, 2, 3, 1)
        attn = (
            self.attn(attn)
            .reshape(
                B,
                h * w,
                self.num_heads,
                self.kernel_size * self.kernel_size,
                self.kernel_size * self.kernel_size,
            )
            .transpose(0, 2, 1, 3, 4)
        )  # B,H,N,kxk,kxk
        attn = attn * self.scale
        attn = softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (
            (attn @ v)
            .transpose(0, 1, 3, 2, 4)
            .reshape(B, C, h, w, self.kernel_size, self.kernel_size)
        )
        x = sliding_window_transpose(
            x,
            output_size=(H, W),
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )

        x = self.proj(x.transpose(0, 2, 3, 1))
        x = self.proj_drop(x)

        return x
