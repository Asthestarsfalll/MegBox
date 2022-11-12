from typing import Optional, Sequence

import megengine.functional as F
from megengine import Parameter, Tensor
from megengine.module import BatchNorm2d

from ..functional.tensor import pad


def zero_padding(x: Tensor, pad_size: int) -> Tensor:
    if x is None:
        return 0
    if pad_size == 0:
        return x
    return pad(x, (pad_size,) * 4)


def create_identity_kernel(groups_channel, groups):
    # group convlution kernel shape:
    # [groups, out_channels // groups, in_channels // groups, kernel_size, kernel_size]
    kernel_value = F.zeros(
        (groups_channel * groups, groups_channel, 1, 1), dtype="float32"
    )
    for i in range(groups_channel * groups):
        kernel_value[i, i % groups_channel, 0, 0] = 1
    if groups > 1:
        kernel_value = kernel_value.reshape(
            groups, groups_channel, groups_channel, 1, 1
        )
    identity = Parameter(kernel_value)
    return identity


def merge_kernels(
    kernels: Sequence[Parameter], kernel_sizes: Sequence[int]
) -> Parameter:
    max_kernel = max(kernel_sizes)
    padding_size = [(max_kernel - k) // 2 for k in kernel_sizes]
    kernels = [zero_padding(k, p) for k, p in zip(kernels, padding_size)]

    return sum(kernels)


def fuse_conv_bn(
    conv_kernel: Parameter, bn: BatchNorm2d, conv_bias: Optional[Parameter] = None
):
    assert bn.affine
    conv_bias = 0 if conv_bias is None else conv_bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = F.sqrt(running_var + eps)
    # for broadcast
    t = (gamma / std).reshape(*conv_kernel.shape[:-3], 1, 1, 1)
    rep_kernel = conv_kernel * t
    rep_bias = beta + (conv_bias - running_mean) * gamma / std
    return rep_kernel, rep_bias


def _get_dilation_kernel_size(kernel_size, dilation):
    return kernel_size + (dilation - 1) * (kernel_size - 1)


def pad_with_dilation(kernel: Parameter, dilation: int) -> Parameter:
    if dilation == 1:
        return kernel
    kernel_size = kernel.shape[-1]
    assert kernel.shape[-2] == kernel_size and kernel_size % 2

    s = _get_dilation_kernel_size(kernel_size, dilation)
    mask = [i % dilation == 0 for i in range(s)]
    dilation_kernel = F.zeros((*kernel.shape[:-2], s, s))
    for h in range(s):
        if mask[h]:
            for w in range(s):
                if mask[w]:
                    dilation_kernel[..., h, w] = kernel[
                        ..., h // dilation, w // dilation
                    ]
    return Parameter(dilation_kernel)
