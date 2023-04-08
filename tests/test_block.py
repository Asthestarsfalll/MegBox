from megbox import easy_use
from megbox.block import ConvNeXtBlock, InceptionNeXtBlock, SegNextBlock

from .utils import _test_modules

BLOCKS = dict(
    convnext=ConvNeXtBlock, inceptionnext=InceptionNeXtBlock, segnext=SegNextBlock
)

BLOCK_KWARGS = dict(
    convnext=[
        dict(dim=64, drop_path=0.0, mlp_expansion=4.0, implementation=1),
        dict(dim=64, drop_path=0.1, mlp_expansion=3.0, implementation=0),
    ],
    inceptionnext=[
        dict(dim=64, drop_path=0.0, mlp_expansion=4.0),
        dict(dim=64, drop_path=0.1, mlp_expansion=3.0),
    ],
    segnext=[
        dict(dim=64, attention_kernel_sizes=[3, 5]),
        dict(dim=64, attention_kernel_sizes=[3, 5, 7], drop_path=0.1),
    ],
)


def test_block():
    batch_size = 2
    spsize = [224, 256]

    def _get_input(kwargs, sp):
        if "dim" in kwargs:
            ch = kwargs["dim"]
        elif "in_channels" in kwargs:
            ch = kwargs["in_channels"]
        else:
            raise ValueError()
        size = (batch_size, ch, sp, sp)
        return easy_use.randn(*size)

    def check_func(cls, kwargs, sp_size, name, is_gpu):
        module = cls(**kwargs)
        inputs = _get_input(kwargs, sp_size)
        out = module(inputs)
        error_msg = f"Wrong output shape of module {name} with kwargs {kwargs} and spatial size {sp_size}."
        assert out.shape == inputs.shape, error_msg

    _test_modules(
        module_mappers=BLOCKS,
        kwargs_mappers=BLOCK_KWARGS,
        spatial_sizes=spsize,
        check_func=check_func,
    )


if __name__ == "__main__":
    test_block()
