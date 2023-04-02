from abc import ABCMeta, abstractmethod
from typing import Optional

from megengine import Parameter, Tensor
from megengine import functional as F
from megengine import module as M

from megbox.block.mlp import ConvMlp, Mlp
from megbox.module import DropPath
from megbox.types import ModuleType

__all__ = ["NeXtArch"]


class NeXtArch(M.Module, metaclass=ABCMeta):
    ChannelFirst = 0
    ChannelLast = 1

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        mlp_expansion: float = 4,
        layer_scale_init_value: float = 1e-6,
        norm_layer: Optional[ModuleType] = None,
        act_layer: Optional[ModuleType] = None,
        implementation: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.token_mixer = self._build_token_mixer()
        self.norm = norm_layer(dim) if norm_layer else M.Identity()
        # Layer Scale
        self.gamma = (
            Parameter(layer_scale_init_value * F.ones((dim)))
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else M.Identity()
        if implementation == NeXtArch.ChannelFirst:
            self.forward = self._forward_channel_first
            self.mlp = ConvMlp(dim, int(dim * mlp_expansion), act_layer=act_layer)
        elif implementation == NeXtArch.ChannelLast:
            self.forward = self._forward_channel_last
            self.mlp = Mlp(dim, int(dim * mlp_expansion), act_layer=act_layer)
        else:
            raise ValueError(
                f"Expected {self.__class__.__name__}`.ChannelFirst` or "
                f"{self.__class__.__name__}.`ChannelLast`"
            )

    @abstractmethod
    def _build_token_mixer(self) -> M.Module:
        raise NotImplementedError()

    def _forward_channel_first(self, x: Tensor) -> Tensor:
        identity = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma.reshape(-1, 1, 1) * x
        x = identity + self.drop_path(x)
        return x

    def _forward_channel_last(self, x: Tensor) -> Tensor:
        identity = x
        x = self.token_mixer(x)
        x = x.transpose(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = identity + self.drop_path(x)
        return x

    forward = _forward_channel_first
