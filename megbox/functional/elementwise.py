import megengine as mge
import megengine.functional as F
from megengine import Tensor
from megengine.core.tensor.array_method import _elwise

from .tensor import where
from .distribution import sample_exponential


def celu(x: Tensor, alpha: float = 1.0) -> Tensor:
    return F.maximum(x, 0) + F.minimum(0, alpha*(F.exp(x) - 1))


def drop_path(
    x: Tensor,
    drop_prob: float = 0.,
    training: bool = False,
) -> Tensor:
    if drop_prob == 0. or not training:
        return x
    keep_prob = mge.tensor(1 - drop_prob, dtype=x.dtype)
    # work with diff dim tensors, not just 2D ConvNets
    size = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + mge.random.normal(mean=0, std=1, size=size)
    random_tensor = F.floor(random_tensor)  # binarize
    output = x / keep_prob * random_tensor
    return output


def erfinv(x: Tensor) -> Tensor:
    return _elwise(x, "erfinv")


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    return where(x > 0, x, alpha*(F.exp(x) - 1))


def glu(x: Tensor, axis: int = -1) -> Tensor:
    assert x.shape[axis] % 2 == 0
    a, b = F.split(x, 2)
    return a * F.sigmoid(b)


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1.,
    hard: bool = False,
    eps: float = 1e-10,
    axis: int = -1
) -> Tensor:
    """ 
        Generate gumble noise, G_i = -log(-log(U_i)), U_i \in U(0, 1)
        More details see https://arxiv.org/pdf/1611.00712.pdf
    """
    gumble_noise = -F.log(sample_exponential(logits.shape, eps=eps) + eps)

    gumbels = (logits + gumble_noise) / tau
    y_soft = F.softmax(gumbels, axis=axis)

    if hard:
        index = F.argmax(y_soft, axis=axis, keepdims=True)
        y_hard = F.scatter(F.zeros_like(logits), axis=axis,
                           index=index, source=1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


def hardshrink(x: Tensor, lambd: float = 0.5) -> Tensor:
    return where((x <= lambd and x >= -lambd), 0, x)


def hardsigmoid(x: Tensor) -> Tensor:
    out = where(x >= 3, 1, x / 6 + 1 / 2)
    return where(x <= -3, 0, out)


def hardswish(x: Tensor) -> Tensor:
    out = where(x <= -3, 0, x * (x + 3) / 6)
    return where(x >= 3, x, out)


def hardtanh(x: Tensor, max_value: float, min_value: float) -> Tensor:
    out = where(x > max_value, max_value, x)
    return where(x < min_value, min_value, out)


def mish(x: Tensor) -> Tensor:
    return x * F.tanh(F.softplus(x))


def rrelu(x: Tensor, lower: float = 1/8, upper: float = 1/3) -> Tensor:
    a = mge.random.uniform(lower, upper, size=(1,))
    return where(x >= 0, x, a * x)


def swish(x: Tensor, beta: float = 1.0) -> Tensor:
    return x * F.sigmoid(beta * x)


def softshrink(x: Tensor, lambd: float = 0.5) -> Tensor:
    out = where(x > lambd, x - lambd, 0)
    return where(x < -lambd, x + lambd, out)


def softsign(x: Tensor) -> Tensor:
    return x / 1 + F.abs(x)


def tanhshrink(x: Tensor) -> Tensor:
    return x - F.tanh(x)


def threshold(x: Tensor, threshold: float, value: float) -> Tensor:
    return where(x > threshold, x, value)
