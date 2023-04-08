from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Union

from megengine import Tensor
from megengine import module as M

from megbox.block.layer_scale import LayerScale
from megbox.module import DropPath

from .mlp_arch import MlpArch

__all__ = ["TransformerArch"]


def _to_2tuple(value) -> Sequence[float]:
    if isinstance(value, (int, float)):
        value = (float(value),) * 2
    return value


class TransformerArch(M.Module, metaclass=ABCMeta):
    def __init__(
        self,
        dim: int,
        attention: M.Module,
        mlp: MlpArch,
        drop_path: Union[float, Sequence[float]] = 0.0,
        layer_scale_init_value: Optional[
            Optional[Union[float, Sequence[float]]]
        ] = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.attn = attention
        pre_norm = self._build_pre_norm()
        post_norm = self._build_post_norm()
        self.pre_norm = pre_norm if pre_norm else M.Identity()
        self.post_norm = post_norm if post_norm else M.Identity()

        if layer_scale_init_value:
            layer_scale_init_value = _to_2tuple(layer_scale_init_value)
            self.ls1 = LayerScale(dim, layer_scale_init_value[0])
            self.ls2 = LayerScale(dim, layer_scale_init_value[1])
        else:
            self.ls1 = M.Identity()
            self.ls2 = M.Identity()

        drop_path = _to_2tuple(drop_path)
        self.drop_path1 = DropPath(drop_path[0]) if drop_path[0] else M.Identity()
        self.drop_path2 = DropPath(drop_path[1]) if drop_path[1] else M.Identity()

        self.mlp = mlp

    @abstractmethod
    def _build_pre_norm(self) -> Optional[M.Module]:
        raise NotImplementedError()

    @abstractmethod
    def _build_post_norm(self) -> Optional[M.Module]:
        raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.pre_norm(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.post_norm(x))))
        return x
