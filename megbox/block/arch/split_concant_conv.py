from abc import ABCMeta, abstractmethod
from typing import List, Sequence, Tuple, Union

from megengine import functional as F
from megengine import module as M

from megbox.functional.tensor import split

__all__ = ["SplitConcatConv2d"]


class SplitConcatConv2d(M.Module, metaclass=ABCMeta):
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

        self.convs = self._build_split_convs()
        self.split_channels = self._get_split_channels()
        if len(self.split_channels) != len(self.convs):
            raise RuntimeError(
                "Expected `split_channels` and `convs` to have same length."
            )

    @abstractmethod
    def _build_split_convs(self) -> List[M.Module]:
        raise NotImplementedError()

    @abstractmethod
    def _get_split_channels(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    def forward(self, x):
        splits = split(x, self.split_channels, axis=1)
        outs = [m(s) for m, s in zip(self.convs, splits)]
        return F.concat(outs, axis=1)
