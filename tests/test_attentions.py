import random

import megengine as mge
from utils import _test_modules

from megbox.attention import (BAMBlock, CBAMBlock, CrissCrossAttention,
                              EABlock, ECABlock, MultiheadAttention,
                              OutlookAttention, ParallelPolarizedSelfAttention,
                              SEBlock, SequentialPolarizedSelfAttention, SimAM)

ATTENTION_KWARGS = dict(
    bam=[
        dict(in_channels=16, reduction=4),
        dict(in_channels=64, reduction=16, dilations=[2, 4], num_layers=2),
    ],
    cbam=[
        dict(in_channels=16, reduction=8, kernel_size=5),
        dict(in_channels=64, reduction=16, kernel_size=9),
    ],
    cca=[
        dict(in_channels=16, reduction=8),
        dict(in_channels=64, reduction=16),
    ],
    eca=[
        dict(in_channels=16, kernel_size=5),
        dict(in_channels=64, kernel_size=7),
    ],
    ea=[
        dict(channels=16, inner_channels=8),
        dict(channels=64, inner_channels=32),
    ],
    out_look=[
        dict(dim=64, num_heads=8, kernel_size=3),
        dict(dim=128, num_heads=8, kernel_size=7, padding=3),
        dict(dim=32, num_heads=8),
    ],
    p_psa=[
        dict(in_channels=64, reduction=16),
        dict(in_channels=32, reduction=2),
    ],
    s_psa=[
        dict(in_channels=64, reduction=16),
        dict(in_channels=32, reduction=2),
    ],
    se=[
        dict(in_channels=16, reduction=8),
        dict(in_channels=64, reduction=16),
    ],
    simam=[dict(e_lambda=1e-8)],
    mhsa=[
        dict(embed_dim=64, num_heads=8),
        dict(embed_dim=64, num_heads=4, add_bias_kv=True),
        # dict(embed_dim=64, num_heads=4, kdim=32, vdim=16),
    ],
)

ATTENTIONS = dict(
    bam=BAMBlock,
    cbam=CBAMBlock,
    cca=CrissCrossAttention,
    eca=ECABlock,
    ea=EABlock,
    out_look=OutlookAttention,
    p_psa=ParallelPolarizedSelfAttention,
    s_psa=SequentialPolarizedSelfAttention,
    se=SEBlock,
    simam=SimAM,
    mhsa=MultiheadAttention,
)


def test_attentions():
    spatial_sizes = [64, 128]
    batch_size = 2

    def get_input(kwargs, spatial_size):
        if "dim" in kwargs.keys():
            chan = kwargs["dim"]
            return mge.random.normal(
                0, 1, (batch_size, spatial_size, spatial_size, chan)
            )
        if "embed_dim" in kwargs.keys():
            chan = kwargs["embed_dim"]
            return mge.random.normal(0, 1, (batch_size, spatial_size, chan))
        if "in_channels" in kwargs.keys():
            chan = kwargs["in_channels"]
        elif "channels" in kwargs.keys():
            chan = kwargs["channels"]
        else:
            chan = random.randint(1, 256)
        return mge.random.normal(0, 1, (batch_size, chan, spatial_size, spatial_size))

    def check_func(cls, kwargs, sp_size, name, is_gpu):
        x = get_input(kwargs, sp_size)
        if cls == ECABlock:
            kwargs.pop("in_channels", None)
        module = cls(**kwargs)
        y = module(x)
        if isinstance(y, tuple):
            y = y[0]
        assert (
            x.shape == y.shape
        ), f"Wrong output shape of module {name} with kwargs {kwargs} and spatial size {sp_size}."

    _test_modules(
        module_mappers=ATTENTIONS,
        kwargs_mappers=ATTENTION_KWARGS,
        spatial_sizes=spatial_sizes,
        check_func=check_func,
    )


if __name__ == "__main__":
    test_attentions()
