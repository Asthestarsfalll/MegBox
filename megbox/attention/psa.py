import math
from typing import Optional

import megengine.module as M
from megengine import Tensor
import megengine.functional as F

from .init import _init_weights


class PolarizedChannelAttention(M.Module):
    def __init__(self, in_channels: int, reduction: int = 2):
        self.inner_chan = in_channels // reduction
        super(PolarizedChannelAttention, self).__init__()
        self.w_q = M.Conv2d(in_channels, 1, 1)
        self.w_v = M.Conv2d(in_channels, self.inner_chan, 1)
        self.softmax = M.Softmax(axis=1)
        self.w_z = M.Conv2d(self.inner_chan, in_channels, 1)
        self.ln = M.LayerNorm(in_channels)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        q = self.w_q(x).reshape(b, h*w, 1)
        v = self.w_v(x).reshape(b, self.inner_chan, -1)
        attn = F.expand_dims(v @ self.softmax(q), -1)
        z = F.squeeze(self.w_z(attn), -1)
        out = self.ln(z.reshape(b, c, 1).transpose(0, 2, 1))
        out = F.sigmoid(out.transpose(0, 2, 1))
        return F.expand_dims(out, axis=-1) * x


class PolarizedSpatialAttention(M.Module):
    def __init__(self, in_channels: int, reduction: int = 2) -> None:
        super(PolarizedSpatialAttention, self).__init__()
        self.inner_chan = in_channels
        self.w_q = M.Conv2d(in_channels, self.inner_chan, 1)
        self.w_v = M.Conv2d(in_channels, self.inner_chan, 1)
        self.gp = M.AdaptiveAvgPool2d((1, 1))
        self.softmax = M.Softmax(axis=-1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        q = self.gp(self.w_q(x)).reshape(b, 1, self.inner_chan)
        v = self.w_v(x).reshape(b, self.inner_chan, -1)
        z = (self.softmax(q) @ v).reshape(b, 1, h, w)
        return F.sigmoid(z) * x


class _PolarizedSelfAttention(M.Module):
    def __init__(self, in_channels: int, reduction: int = 2) -> None:
        super(_PolarizedSelfAttention, self).__init__()
        self.ca = PolarizedChannelAttention(in_channels, reduction)
        self.sa = PolarizedSpatialAttention(in_channels, reduction)


class ParallelPolarizedSelfAttention(_PolarizedSelfAttention):
    def forward(self, x: Tensor) -> Tensor:
        return self.ca(x) + self.sa(x)


class SequentialPolarizedSelfAttention(_PolarizedSelfAttention):
    def forward(self, x: Tensor) -> Tensor:
        return self.sa(self.ca(x))
