from abc import ABCMeta, abstractmethod
from typing import List, Sequence, Union

from megengine import functional as F
from megengine import module as M

__all__ = ["MultiPathConv2d"]


class MultiPathConv2d(M.Module, metaclass=ABCMeta):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[Union[Sequence[int], int]],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.convs = self._build_convs()

    @abstractmethod
    def _build_convs(self) -> List[M.Module]:
        raise NotImplementedError()

    def forward(self, x):
        return F.concat([m(x)] for m in self.convs)
