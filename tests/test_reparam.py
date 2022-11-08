import megengine as mge
import numpy as np
from megengine.module import BatchNorm2d, Conv2d, Identity, init
from utils import _test_modules

from megbox.reparam import RepConv2d, RepLargeKernelConv2d

GLOBAL_RTOL = 1e-1
GLOBAL_ATOL = 1e-5

REPS = dict(
    rep_conv=RepConv2d,
    rep_lk_conv=RepLargeKernelConv2d,
)


REPS_KWAGS = dict(
    rep_conv=[
        dict(in_channels=32, out_channels=32),
        dict(in_channels=32, out_channels=32, bias=True),
        dict(in_channels=32, out_channels=32, groups=16),
        dict(in_channels=32, out_channels=64),
        dict(in_channels=32, out_channels=32, kernel_size=5, small_kernel_size=3),
        dict(in_channels=32, out_channels=32, dilation=2),
        dict(in_channels=32, out_channels=64, dilation=(1, 3)),
        dict(
            in_channels=32,
            out_channels=64,
            kernel_size=7,
            small_kernel_size=5,
            dilation=(1, 3),
        ),
        dict(in_channels=32, out_channels=32, stride=2),
        dict(in_channels=32, out_channels=32, stride=2, dilation=2),
        dict(in_channels=32, out_channels=32, attention=Identity()),
    ],
    rep_lk_conv=[
        dict(channels=32),
        dict(channels=32, bias=True),
        dict(channels=32, kernel_size=7, small_kernel_size=(5, 3, 1)),
        dict(channels=32, dilation=2),
        dict(channels=64, kernel_size=5, small_kernel_size=(3, 1), dilation=(1, 3, 2)),
        dict(channels=32, stride=2, dilation=2),
    ],
)


def _init_weights(m):
    if isinstance(m, Conv2d):
        init.ones_(m.weight)
        if m.bias is not None:
            init.ones_(m.bias)
    elif isinstance(m, BatchNorm2d) and m.affine:
        init.ones_(m.weight)
        init.ones_(m.bias)


def test_reparams():
    batch_size = 2
    spatial_sizes = [64, 128]

    def get_input(kwargs, spatial_size):
        if "in_channels" in kwargs.keys():
            chan = kwargs["in_channels"]
        elif "channels" in kwargs.keys():
            chan = kwargs["channels"]
        else:
            raise ValueError()
        return mge.functional.ones((batch_size, chan, spatial_size, spatial_size))

    def check_func(module, kwargs, sp_size, name):
        print(kwargs)
        module.apply(_init_weights)
        module.eval()

        x = get_input(kwargs, sp_size)
        y = module(x).numpy()

        module.switch_to_deploy()
        assert module.is_deploy

        y1 = module(x).numpy()

        error_msg = f"Different output after switch to deploy of module {name}, with kwargs {kwargs} and spatial size {sp_size}"  # noqa: E501
        np.testing.assert_allclose(
            y1, y, rtol=GLOBAL_RTOL, atol=GLOBAL_ATOL, err_msg=error_msg
        )

        new_module = type(module)(is_deploy=True, **kwargs)
        state_dict = module.state_dict()
        new_module.load_state_dict(state_dict)

        y2 = module(x).numpy()
        error_msg = f"Different output of module {name}, with kwargs {kwargs} and spatial size {sp_size}"  # noqa: E501

        np.testing.assert_allclose(
            y2, y1, rtol=GLOBAL_RTOL, atol=GLOBAL_ATOL, err_msg=error_msg
        )

    _test_modules(
        module_mapper=REPS,
        kwargs_mapper=REPS_KWAGS,
        spatial_sizes=spatial_sizes,
        check_func=check_func,
    )


if __name__ == "__main__":
    test_reparams()
