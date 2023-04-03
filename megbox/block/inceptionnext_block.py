from typing import Optional

from megengine import module as M

from megbox.types import ModuleType

from .arch import NeXtArch, SplitConcatConv2d

__all__ = ["InceptionDWConv2d", "InceptionNeXtBlock"]


class InceptionDWConv2d(SplitConcatConv2d):
    def __init__(
        self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125
    ):
        self.branch_ratio = branch_ratio

        super().__init__(
            in_channels, in_channels, [square_kernel_size, band_kernel_size]
        )

    def _build_split_convs(self):
        gc = int(self.in_channels * self.branch_ratio)
        sk, bk = self.kernel_sizes
        convs = [
            M.Identity(),
            M.Conv2d(gc, gc, sk, padding=sk // 2, groups=gc),
            M.Conv2d(gc, gc, kernel_size=(1, bk), padding=(0, bk // 2), groups=gc),
            M.Conv2d(
                gc,
                gc,
                kernel_size=(bk, 1),
                padding=(bk // 2, 0),
                groups=gc,
            ),
        ]
        return convs

    def _get_split_channels(self):
        gc = int(self.in_channels * self.branch_ratio)
        return (self.in_channels - 3 * gc, gc, gc, gc)


class InceptionNeXtBlock(NeXtArch):
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        dropout_ratio: float = 0.0,
        mlp_expansion: float = 4.0,
        layer_scale_init_value: float = 1e-6,
        norm_layer: Optional[ModuleType] = M.BatchNorm2d,
        act_layer: Optional[ModuleType] = M.GELU,
    ):
        super().__init__(
            dim,
            drop_path,
            dropout_ratio,
            mlp_expansion,
            layer_scale_init_value,
            norm_layer,
            act_layer,
        )

    def _build_token_mixer(self):
        return InceptionDWConv2d(self.dim)
