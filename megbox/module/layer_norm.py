from megengine import Parameter, Tensor
from megengine import functional as F
from megengine import module as M

from megbox.functional import pow


class LayerNorm(M.Module):
    ChannelFirst = 0
    ChannelLast = 1
    r"""
    LayerNorm that supports two data formats: channels_last and
    channels_first. The ordering of the dimensions in the inputs.
    channels_last corresponds to inputs with shape
    (batch_size, height, width, channels).
    while channels_first corresponds to inputs with shape
    (batch_size, channels, height, width).

    Args:
        normalized_shape (tuple): The shape which is specified
            to be normalizated.
        eps (float): A value added to the denominator for
            numerical stability. Default: 1e-6.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: int = ChannelLast,
        need_transpose: bool = False,
    ):
        super().__init__()
        self.weight = Parameter(F.ones(normalized_shape))
        self.bias = Parameter(F.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
        if not need_transpose:
            if data_format == LayerNorm.ChannelFirst:
                self.forward = self._forward_channel_first
            elif data_format == LayerNorm.ChannelLast:
                self.forward = self._forward_channel_last
            else:
                raise ValueError(
                    "Expected `data_format` to be one of "
                    "`LayerNorm.ChannelLast` or `LayerNorm.ChannelFitst`, "
                    f"but got {data_format}"
                )
        else:
            self.forward = self._forward_channel_first_with_transpose

    def _forward_channel_last(self, x: Tensor) -> Tensor:
        return F.layer_norm(
            x, self.normalized_shape, True, self.weight, self.bias, self.eps
        )

    def _forward_channel_first(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdims=True)
        s = pow(x - u, 2).mean(1, keepdims=True)
        x = (x - u) / F.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

    def _forward_channel_first_with_transpose(self, x: Tensor) -> Tensor:
        """x (Tensor): ChannelFirst data format"""
        x = x.transpose(0, 2, 3, 1)
        x = self._forward_channel_last(x)
        x = x.transpose(0, 3, 1, 2)
        return x

    forward = _forward_channel_last
