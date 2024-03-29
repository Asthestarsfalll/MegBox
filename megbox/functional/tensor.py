import math
from typing import List, Optional, Sequence, Tuple, Union

import megengine as mge
import megengine.functional as F
import numpy as np
from megengine import Tensor

from megbox.types import Number
from megbox.utils import (assert_no_kwargs, handle_negtive_aixs, handle_number,
                          index_cast, to_1tuple)

where = assert_no_kwargs(index_cast(handle_number, (1, 2))(F.where))
pow = assert_no_kwargs(index_cast(handle_number, (0, 1))(F.pow))


def pad(
    src: Tensor,
    pad_width: Tuple[int, ...],
    mode: str = "constant",
    constant_value: float = 0.0,
) -> Tensor:
    target_length = len(src.shape)
    pad_length = len(pad_width)
    assert pad_length % 2 == 0
    pad_pairs = [(pad_width[i], pad_width[i + 1]) for i in range(0, pad_length, 2)]
    if pad_length // 2 != target_length:
        pad_pairs.extend([(0, 0) for _ in range(target_length - pad_length // 2)])
    pad_pairs.reverse()

    return F.pad(src, tuple(pad_pairs), mode, constant_value)


def _swap_elmen_of_list(lis: List, index1: int, index2: int):
    lis[index1], lis[index2] = lis[index2], lis[index1]
    return lis


def exchang_axes(x: Tensor, axis0: int, axis1: int):
    ndim = x.ndim
    axis0 = handle_negtive_aixs(axis0, ndim)
    axis1 = handle_negtive_aixs(axis1, ndim)
    permutation = _swap_elmen_of_list(list(range(ndim)), axis0, axis1)

    return x.transpose(permutation)


# TODO:
# def sliding_window_transpose(
#     x: Tensor,
#     output_size: Union[int, Tuple[int, int]],
#     kernel_size: Union[int, Tuple[int, int]],
#     padding: Union[int, Tuple[int, int]] = 0,
#     stride: Union[int, Tuple[int, int]] = 1,
#     dilation: Union[int, Tuple[int, int]] = 1,
# ) -> Tensor:
#     # [N, C * kh * kw, H*W] or [C * kh * kw, H*W]
#     if x.ndim == 3:
#         # [1, C * kh * kw, H*W]
#         x = F.expand_dims(x, axis=1)
#     N, S, L = x.shape
#     # N, C, H, W, kh, kw
#     x = x.reshape(
#         N,
#     )
#     return F.sliding_window_transpose(
#         x, output_size, kernel_size, padding, stride, dilation
#     )


def expand_dims_with_repeat(
    x: Tensor,
    axis: Union[int, Sequence[int]],
    repeats: Union[int, Sequence[int]],
) -> Tensor:
    axis = to_1tuple(axis)
    repeats = to_1tuple(repeats)
    assert len(axis) == len(repeats)
    y = F.expand_dims(x, axis)
    for a, r in zip(axis, repeats):
        y = F.repeat(y, r, axis=a)
    return y


def accuracy(output: Tensor, target: Tensor, topk: Tuple = (1,)) -> List[Number]:
    if output.ndim != 2:
        raise ValueError("The dimension of `output` must be 2")
    pred = F.topk(output, max(topk), True)[1].transpose(1, 0)
    target = F.repeat(target.reshape(1, -1), pred.shape[0], axis=0)
    correct = target == pred
    return [float(correct[:k].reshape(-1).sum(0, keepdims=True).numpy()) for k in topk]


def cosine_similarity(x: Tensor, y: Tensor, axis: int = 1, eps: float = 1e-8) -> Tensor:
    if x.ndim != y.ndim:
        raise ValueError("The inputs must have same dimension.")

    axis = handle_negtive_aixs(axis, x.ndim)

    if axis >= x.ndim:
        raise ValueError("Wrong axis was given.")
    t = F.norm(x, ord=2.0, axis=axis, keepdims=False) * F.norm(
        y, ord=2.0, axis=axis, keepdims=False
    )
    t = F.maximum(t, eps)
    return F.sum(x * y, axis=axis) / t


def differentiable_topk(x: Tensor, k: Tensor, temperature: float = 1.0) -> Tensor:
    n, dim = x.shape
    topks = []
    for i in range(k):
        prob = F.softmax(x / temperature, axis=-1)
        values, indices = F.topk(prob, 1, descending=True)
        topk = F.scatter(F.zeros_like(x), axis=-1, index=indices, source=values)
        topks.append(topk)
        if not i == k - 1:
            x = F.scatter(
                x,
                axis=-1,
                index=indices,
                source=F.full(indices.shape, value=-float("Inf")),
            )
    topks = F.concat(topks, axis=-1)
    return F.sum(topks.reshape(n, k, dim), axis=1)


def randn(*shape: Tuple[int, ...], device: Optional[str] = None) -> Tensor:
    out = mge.random.normal(0, 1, shape)
    if device is not None:
        out = out.to(device)
    return out


def split(x: Tensor, split_size_or_sections: Union[int, Sequence[int]], axis: int = 0):
    if isinstance(split_size_or_sections, int):
        nsplits_or_sections = math.floor(x.shape[axis] / split_size_or_sections)
    else:
        nsplits_or_sections = np.cumsum(split_size_or_sections).tolist()
        nsplits_or_sections.pop()
    return F.split(x, nsplits_or_sections, axis)
