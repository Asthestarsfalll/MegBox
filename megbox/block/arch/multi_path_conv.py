import functools
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Sequence, Union

from megengine import functional as F
from megengine import module as M

__all__ = ["MultiPathConv2d"]


class MultiPathConv2d(M.Module, metaclass=ABCMeta):
    Concat = 0
    Sum = 1
    def __init__(
        self,
        in_channels: int,
        kernel_sizes: Sequence[Union[Sequence[int], int]],
        out_channels: Optional[int] = None,
        out_func: int = Concat,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.kernel_sizes = kernel_sizes

        self.in_conv = self._build_in_conv()
        self.out_conv = self._build_out_conv()
        self.convs = self._build_convs()
        if out_func == MultiPathConv2d.Concat:
            self.func = functools.partial(F.concat, axis=1)
        elif out_func == MultiPathConv2d.Sum:
            self.func = sum

    def _build_in_conv(self) -> Optional[M.Module]:
        return None

    def _build_out_conv(self) -> Optional[M.Module]:
        return None

    @abstractmethod
    def _build_convs(self) -> List[M.Module]:
        raise NotImplementedError()

    def forward(self, x):
        if self.in_conv is not None:
            x  = self.in_conv(x)

        feats = [m(x) for m in self.convs]
        out = self.func(feats)

        if self.out_conv is not None:
            out = self.out_conv(out)
        return out
