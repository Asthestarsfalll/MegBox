from typing import List, Optional, Sequence, Tuple, Union

import megengine.functional as F
import megengine.module as M
from megengine import Parameter, Tensor
from megengine.module import BatchNorm2d as MgeBatchNorm2d
from megengine.module import Conv2d
from megengine.module import ConvBn2d as MgeConvBn2d
from megengine.module import Identity, Module, ReLU

from .utils import (_get_dilation_kernel_size, create_identity_kernel,
                    fuse_conv_bn, merge_kernels, pad_with_dilation)

__all__ = [
    "BatchNorm2d",
    "ConvBn2d",
    "RepConv2d",
    "RepLargeKernelConv2d",
]


def _init_weights(m):
    if isinstance(m, ConvBn2d):
        if m.conv.bias is not None:
            M.init.normal_(m.conv.bias)


class ConvBn2d(MgeConvBn2d):
    def _get_equivalent_kernel_bias(self) -> Tuple[Parameter, Parameter]:
        kernel = pad_with_dilation(self.conv.weight, self.conv.dilation[0])
        return fuse_conv_bn(kernel, self.bn, self.conv.bias)


class BatchNorm2d(MgeBatchNorm2d):
    def _get_equivalent_kernel_bias(
        self, groups_channel: int, groups: int
    ) -> Tuple[Parameter, Parameter]:
        identity_kernel = create_identity_kernel(groups_channel, groups)
        return fuse_conv_bn(identity_kernel, self)


class RepConv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        small_kernel_size: int = 1,
        stride: int = 1,
        dilation: Union[int, Sequence[int]] = (1, 1),
        groups: int = 1,
        bias: bool = False,
        nonlinearity: Optional[Module] = ReLU(),
        attention: Optional[Module] = None,
        is_deploy: bool = False,
    ) -> None:
        super().__init__()

        if small_kernel_size > kernel_size:
            raise ValueError("`small_kernel_size` muse be smaller than `kernel_size`")

        self.kernel_size = kernel_size
        self.small_kernel_size = small_kernel_size

        self.groups = groups
        self.groups_channel = in_channels // groups

        self.nonlinearity = nonlinearity if nonlinearity is not None else Identity()
        self.attn = attention if attention is not None else Identity()

        self.is_deploy = is_deploy
        if isinstance(dilation, int):
            dilation = [dilation] * 2

        self.dilation = dilation

        if is_deploy:
            kernel_size = max(self._get_equivalent_kernel_size())
            self.reparam = Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                bias=True,
            )
        else:
            self.identity = (
                BatchNorm2d(num_features=in_channels)
                if in_channels == out_channels and stride == 1
                else None
            )
            # do not use bias preceding batch_norm
            self.large = ConvBn2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=dilation[0] * (kernel_size // 2),
                groups=groups,
                dilation=dilation[0],
                bias=bias,
            )
            self.small = ConvBn2d(
                in_channels,
                out_channels,
                kernel_size=small_kernel_size,
                stride=stride,
                padding=dilation[1] * (small_kernel_size // 2),
                groups=groups,
                dilation=dilation[1],
                bias=bias,
            )

        self.apply(_init_weights)

    def forward(self, x: Tensor) -> Tensor:
        if self.is_deploy:
            return self.nonlinearity(self.attn(self.reparam(x)))

        identity = 0 if self.identity is None else self.identity(x)

        return self.nonlinearity(self.attn(self.large(x) + self.small(x) + identity))

    def _get_equivalent_kernel_size(self) -> List[int]:
        kernel_sizes = [
            _get_dilation_kernel_size(self.kernel_size, self.dilation[0]),
            _get_dilation_kernel_size(self.small_kernel_size, self.dilation[1]),
            1,
        ]
        return kernel_sizes

    def switch_to_deploy(self) -> None:
        if self.is_deploy:
            return

        kernel_L, bias_L = self.large._get_equivalent_kernel_bias()
        kernel_S, bias_S = self.small._get_equivalent_kernel_bias()

        if self.identity is None:
            kernel_id = F.zeros((*kernel_L.shape[:-2], 1, 1))
            bias_id = F.zeros_like(bias_L)
        else:
            kernel_id, bias_id = self.identity._get_equivalent_kernel_bias(
                self.groups_channel, self.groups
            )

        kernel_sizes = self._get_equivalent_kernel_size()

        kernel = merge_kernels([kernel_L, kernel_S, kernel_id], kernel_sizes)
        bias = bias_L + bias_S + bias_id

        self.reparam = Conv2d(
            in_channels=self.large.conv.in_channels,
            out_channels=self.large.conv.out_channels,
            kernel_size=max(kernel_sizes),
            stride=self.large.conv.stride,
            padding=max(kernel_sizes) // 2,
            groups=self.large.conv.groups,
            bias=True,
        )

        self.reparam.weight[:] = kernel
        self.reparam.bias[:] = bias

        for para in self.parameters():
            para.detach()
        delattr(self, "large")
        delattr(self, "small")
        if hasattr(self, "identity"):
            delattr(self, "identity")
        if hasattr(self, "bn_identity"):
            delattr(self, "bn_identity")
        self.is_deploy = True


class RepLargeKernelConv2d(Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        small_kernel_size: Union[int, Sequence[int]] = 1,
        stride: int = 1,
        dilation: Union[int, Sequence[int]] = 1,
        bias: bool = False,
        is_deploy: bool = False,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size

        self.is_deploy = is_deploy

        if isinstance(small_kernel_size, int):
            small_kernel_size = [small_kernel_size]

        if isinstance(dilation, int):
            dilation = [dilation] * (len(small_kernel_size) + 1)
        elif len(dilation) != (len(small_kernel_size) + 1):
            raise ValueError()

        self.small_kernel_size = small_kernel_size
        self.dilation = dilation

        if is_deploy:
            kernel_size = max(self._get_equivalent_kernel_size())
            self.reparam = Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=channels,
                bias=True,
            )
        else:
            self.dw_large = ConvBn2d(
                channels,
                channels,
                kernel_size,
                stride=stride,
                padding=dilation[0] * (kernel_size // 2),
                groups=channels,
                dilation=dilation[0],
                bias=bias,
            )

            for k, d in zip(small_kernel_size, dilation[1:]):
                if k > kernel_size:
                    raise ValueError(
                        "`small_kernel_size` muse be smaller than `kernel_size`"
                    )
                self.__setattr__(
                    f"dw_small_{k}",
                    ConvBn2d(
                        channels,
                        channels,
                        k,
                        stride=stride,
                        padding=d * (k // 2),
                        groups=channels,
                        dilation=d,
                        bias=bias,
                    ),
                )

        self.apply(_init_weights)

    def forward(self, x: Tensor) -> Tensor:
        if self.is_deploy:
            return self.reparam(x)
        out = self.dw_large(x)
        for k in self.small_kernel_size:
            out += getattr(self, f"dw_small_{k}")(x)
        return out

    def _get_equivalent_kernel_size(self) -> List[int]:
        kernel_sizes = [self.kernel_size, *self.small_kernel_size]
        kernel_sizes = [
            _get_dilation_kernel_size(k, d) for k, d in zip(kernel_sizes, self.dilation)
        ]
        return kernel_sizes

    def switch_to_deploy(self) -> None:
        if self.is_deploy:
            return

        kernel_bias_pairs = [self.dw_large._get_equivalent_kernel_bias()]
        for k in self.small_kernel_size:
            kernel_bias_pairs.append(
                getattr(self, f"dw_small_{k}")._get_equivalent_kernel_bias()
            )
            delattr(self, f"dw_small_{k}")

        kernel_sizes = self._get_equivalent_kernel_size()
        kernels = [p[0] for p in kernel_bias_pairs]

        kernel = merge_kernels(kernels, kernel_sizes)
        bias = sum((p[1] for p in kernel_bias_pairs))

        self.reparam = Conv2d(
            in_channels=self.dw_large.conv.in_channels,
            out_channels=self.dw_large.conv.out_channels,
            kernel_size=max(kernel_sizes),
            stride=self.dw_large.conv.stride,
            padding=max(kernel_sizes) // 2,
            groups=self.dw_large.conv.groups,
            bias=True,
        )

        self.reparam.weight[:] = kernel
        self.reparam.bias[:] = bias

        for para in self.parameters():
            para.detach()

        delattr(self, "dw_large")
        self.is_deploy = True
