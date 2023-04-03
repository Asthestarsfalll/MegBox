import functools

from megengine import module as M

from megbox.module.layer_norm import LayerNorm

from .arch.next_arch import NeXtArch

__all__ = ["ConvNeXtBlock"]


class ConvNeXtBlock(NeXtArch):
    r"""
    ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv ->
        GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) ->
        Linear -> GELU -> Linear; Permute back

    Args:
        dim (int): Number of input channels.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        expansion (float, optional): Expansion of channels of 1x1 convolution.
            Default: 4.0
        layer_scale_init_value (float, optional): Init value for Layer Scale.
            Default: 1e-6.
        implementation (int, optional): Which implementation used for forward,
            ChannelFirst or ChannelLast. Default: ConvNeXtBlock.ChannelLast.
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        dropout_ratio: float = 0.0,
        mlp_expansion: float = 4,
        layer_scale_init_value: float = 1e-6,
        implementation: int = 1,
    ) -> None:
        layer_norm = functools.partial(LayerNorm, data_format=implementation)
        super().__init__(
            dim,
            drop_path,
            dropout_ratio,
            mlp_expansion,
            layer_scale_init_value,
            layer_norm,
            M.GELU,
            implementation,
        )

    def _build_token_mixer(self):
        return M.Conv2d(self.dim, self.dim, kernel_size=7, padding=3, groups=self.dim)
