from abc import ABCMeta, abstractmethod
from typing import Optional

from megengine import Tensor
from megengine.module import Dropout, Identity, Module

from megbox.types import ModuleType


class MlpArch(Module, metaclass=ABCMeta):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[ModuleType] = None,
        drop_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = self._build_fc1(in_features, hidden_features)
        self.act = act_layer() if act_layer else Identity()
        self.fc2 = self._build_fc2(hidden_features, out_features)
        self.drop = Dropout(drop_ratio)

    @abstractmethod
    def _build_fc1(self, in_chan: int, out_chan: int) -> Module:
        raise NotImplementedError()

    @abstractmethod
    def _build_fc2(self, in_chan: int, out_chan: int) -> Module:
        raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
