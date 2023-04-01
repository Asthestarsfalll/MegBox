import megengine as mge
import numpy as np
import torch
from torch.nn import AdaptiveAvgPool2d as TorchAdaptiveAvgPool2d
from torch.nn import AdaptiveMaxPool2d as TorchAdaptiveMaxPool2d
from torch.nn import AvgPool2d as TorchAvgPool2d
from torch.nn import MaxPool2d as TorchMaxPool2d
from utils import _test_modules

from megbox.module.pooling import (AdaptiveAvgPool2d, AdaptiveMaxPool2d,
                                   AvgPool2d, MaxPool2d)

GLOBAL_RTOL = 1e-3
GLOBAL_ATOL = 1e-7


def test_doc(*args):
    for i in args:
        if not i.__doc__:
            raise RuntimeError("Failed to add `__doc__`")


POOLINGS = dict(
    maxpool2d=MaxPool2d,
    avgpool2d=AvgPool2d,
    adpmaxpool2d=AdaptiveMaxPool2d,
    adpavgpool2d=AdaptiveAvgPool2d,
)

TORCH_POOLINGS = dict(
    maxpool2d=TorchMaxPool2d,
    avgpool2d=TorchAvgPool2d,
    adpmaxpool2d=TorchAdaptiveMaxPool2d,
    adpavgpool2d=TorchAdaptiveAvgPool2d,
)
POOLINGS_KWARGS = dict(
    maxpool2d=[
        dict(kernel_size=2, stride=2, padding=0, ceil_mode=False),
        dict(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        dict(kernel_size=3, ceil_mode=True),
    ],
    avgpool2d=[
        dict(kernel_size=2, stride=2, padding=0, ceil_mode=False),
        dict(kernel_size=3, stride=2, padding=0, ceil_mode=True),
        dict(kernel_size=2, ceil_mode=True),
    ],
    adpmaxpool2d=[{"oshp": i} for i in range(1, 12)],
    adpavgpool2d=[{"oshp": i} for i in range(1, 12)],
)
TORCHPOOLINGS_KWARGS = dict(
    adpmaxpool2d=[{"output_size": i} for i in range(1, 12)],
    adpavgpool2d=[{"output_size": i} for i in range(1, 12)],
)


def test_pooling():
    batch_size = 2
    channels = 3
    spatial_sizes = [25, 26, 33]

    def check_func(classes, kwargs, sp_size, name, is_gpu):
        cls, torch_cls = classes
        kwargs, torch_kwargs = kwargs
        torch_model = torch_cls(**torch_kwargs)
        if is_gpu:
            torch_model = torch_model.cuda()
        torch_model.eval()
        model = cls(**kwargs)
        model.eval()

        # data = np.random.randn(batch_size, channels, sp_size, sp_size)
        data = np.ones((batch_size, channels, sp_size, sp_size))

        out = model(mge.tensor(data, dtype="float32")).numpy()
        torch_data = torch.tensor(data, dtype=torch.float32)
        if is_gpu:
            torch_data = torch_data.cuda()
        torch_out = torch_model(torch_data)
        if is_gpu:
            torch_out = torch_out.cpu()
        torch_out = torch_out.numpy()

        error_msg = f"Different output of module {name}, with kwargs {kwargs} and spatial size {sp_size}"  # noqa: E501
        np.testing.assert_allclose(
            out, torch_out, rtol=GLOBAL_RTOL, atol=GLOBAL_ATOL, err_msg=error_msg
        )

    _test_modules(
        module_mappers=[POOLINGS, TORCH_POOLINGS],
        kwargs_mappers=[POOLINGS_KWARGS, TORCHPOOLINGS_KWARGS],
        spatial_sizes=spatial_sizes,
        check_func=check_func,
    )


if __name__ == "__main__":
    test_doc(AvgPool2d, MaxPool2d)
    test_pooling()
