from megengine.module import Conv2d, Linear, Sequential

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


class SegNeXtMlp(MlpArch):
    def _build_fc1(self, in_chan, out_chan):
        return Sequential(
            Conv2d(in_chan, out_chan, 1),
            Conv2d(out_chan, out_chan, 3, padding=1, groups=out_chan),
        )

    def _build_fc2(self, in_chan, out_chan):
        return Conv2d(in_chan, out_chan, 1)
