import megengine.functional as F
import megengine.module as M
from megengine import Parameter, Tensor

from ..functional.tensor import expand_dims_with_repeat
from .init import _init_weights


def _build_mask(b: int, h: int, w: int) -> Tensor:
    mask = F.diag(F.repeat(Tensor(float("Inf")), h))
    mask = expand_dims_with_repeat(mask, axis=0, repeats=b * w)
    return -mask


class CrissCrossAttention(M.Module):
    def __init__(self, in_channels: int, reduction: int = 8) -> None:
        super().__init__()
        inner_chan = in_channels // reduction
        self.query_conv = M.Conv2d(in_channels, inner_chan, 1)
        self.key_conv = M.Conv2d(in_channels, inner_chan, 1)
        self.value_conv = M.Conv2d(in_channels, in_channels, 1)
        self.softmax = M.Softmax(axis=3)
        self.gamma = Parameter(F.zeros((1)))

        self.apply(_init_weights)

    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape

        proj_query = self.query_conv(x)
        # b,c',h,w -> b,w,c',h -> b*w,c',h -> b*w,h,c'
        proj_query_H = (
            proj_query.transpose(0, 3, 1, 2).reshape(b * w, -1, h).transpose(0, 2, 1)
        )
        # b,c',h,w -> b,h,c',w -> b*h,c',w -> b*h,w,c'
        proj_query_W = (
            proj_query.transpose(0, 2, 1, 3).reshape(b * h, -1, w).transpose(0, 2, 1)
        )

        proj_key = self.key_conv(x)
        # b,c',h,w -> b,w,c',h -> b*w,c',h
        proj_key_H = proj_key.transpose(0, 3, 1, 2).reshape(b * w, -1, h)
        # b,c',h,w -> b,h,c',w -> b*h,c',w
        proj_key_W = proj_key.transpose(0, 2, 1, 3).reshape(b * h, -1, w)

        proj_value = self.value_conv(x)
        # b,c,h,w -> b,w,c,h -> b*w,c,h
        proj_value_H = proj_value.transpose(0, 3, 1, 2).reshape(b * w, -1, h)
        # b,c,h,w -> b,h,c,w -> b*h,c,w
        proj_value_W = proj_value.transpose(0, 2, 1, 3).reshape(b * h, -1, w)

        energy_H = (
            ((proj_query_H @ proj_key_H) + _build_mask(b, h, w))
            .reshape(b, w, h, h)
            .transpose(0, 2, 1, 3)
        )
        energy_W = (proj_query_W @ proj_key_W).reshape(b, h, w, w)

        concate = self.softmax(F.concat([energy_H, energy_W], 3))
        # fmt: off
        att_H = concate[:, :, :, 0: h].transpose(0, 2, 1, 3).reshape(b * w, h, h)
        att_W = concate[:, :, :, h: h + w].reshape(b * h, w, w)
        # fmt: on
        out_H = (
            (proj_value_H @ att_H.transpose(0, 2, 1))
            .reshape(b, w, -1, h)
            .transpose(0, 2, 3, 1)
        )
        out_W = (
            (proj_value_W @ att_W.transpose(0, 2, 1))
            .reshape(b, h, -1, w)
            .transpose(0, 2, 1, 3)
        )

        return self.gamma * (out_H + out_W) + x
