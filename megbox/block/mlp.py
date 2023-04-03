from megengine.module import Conv2d, Linear

from .arch.mlp_arch import MlpArch

__all__ = [
    "ConvMlp",
    "Mlp",
]


class Mlp(MlpArch):
    def _build_fc1(self, in_chan, out_chan):
        return Linear(in_chan, out_chan)

    _build_fc2 = _build_fc1


class ConvMlp(MlpArch):
    def _build_fc1(self, in_chan, out_chan):
        return Conv2d(in_chan, out_chan, 1)

    _build_fc2 = _build_fc1
