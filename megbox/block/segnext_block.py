from collections import OrderedDict
from typing import Optional, Sequence, Union

from megengine.module import GELU, BatchNorm2d, Conv2d, Module, Sequential

from megbox.types import ModuleType
from megbox.utils import to_2tuple

from .arch import MultiPathConv2d, TransformerArch
from .mlp import SegNeXtMlp


def get_depthwise_conv(dim, kernel_size: Union[int, Sequence[int]] = 3):
    if isinstance(kernel_size, int):
        kernel_size = to_2tuple(kernel_size)
    padding = [k // 2 for k in kernel_size]
    return Conv2d(dim, dim, kernel_size, padding=padding, groups=dim)


class SegNextAttention(MultiPathConv2d):
    def __init__(
        self,
        in_channels: int,
        kernel_sizes: Sequence[Union[Sequence[int], int]],
        out_channels: Optional[int] = None,
    ):
        super().__init__(in_channels, kernel_sizes, out_channels, MultiPathConv2d.Sum)

    def _build_convs(self):
        return [
            Sequential(
                OrderedDict(
                    [
                        (f"conv{i}_1", get_depthwise_conv(self.in_channels, (1, k))),
                        (f"conv{i}_2", get_depthwise_conv(self.in_channels, (k, 1))),
                    ]
                )
            )
            for i, k in enumerate(self.kernel_sizes)
        ]

    def _build_in_conv(self):
        return get_depthwise_conv(self.in_channels, 5)

    def _build_out_conv(self):
        return Conv2d(self.in_channels, self.in_channels, 1)


class SegNextBlock(TransformerArch):
    def __init__(
        self,
        dim: int,
        attention_kernel_sizes: Sequence[int],
        mlp_ratio: float = 4.0,
        mlp_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: ModuleType = GELU,
        norm_layer: ModuleType = BatchNorm2d,
    ) -> None:
        mlp = SegNeXtMlp(dim, int(dim * mlp_ratio), dim, act_layer, mlp_drop)
        attn = SegNextAttention(dim, kernel_sizes=attention_kernel_sizes)
        self.norm_layer = norm_layer
        super().__init__(dim, attn, mlp, [drop_path, None], None)
        delattr(self, "norm_layer")

    def _build_pre_norm(self) -> Optional[Module]:
        return self.norm_layer(self.dim)

    def _build_post_norm(self) -> Optional[Module]:
        return self.norm_layer(self.dim)
