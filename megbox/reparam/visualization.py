import os
from typing import Optional, Sequence, Union

import megengine.functional as F
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from megengine import Tensor

from .utils import _get_dilation_kernel_size, merge_kernels, pad_with_dilation

MAX = 1


def create_kernel(kernel_size: int) -> Tensor:
    return F.ones((1, 1, kernel_size, kernel_size))


def save_kernel_to_image(
    name: str, kernel: Tensor, annot: bool = True, **kwargs
) -> None:
    image = kernel[0, 0].numpy().astype(np.uint8)
    heatmap = sns.heatmap(
        image, linewidths=5, fmt="d", annot=annot, vmin=0, vmax=MAX, **kwargs
    )
    fig = heatmap.get_figure()
    fig.savefig(name, dpi=400)
    plt.close()


def visualize(
    kernel_sizes: Union[int, Sequence[int]],
    dilations: Union[int, Sequence[int]],
    annot: bool = True,
    max_value: Optional[int] = None,
    save_dir: str = "./",
    **kwargs,
) -> None:

    global MAX

    kernel_sizes = [kernel_sizes] if isinstance(kernel_sizes, int) else kernel_sizes
    dilations = [dilations] if isinstance(dilations, int) else dilations

    if max_value is not None and max_value < len(kernel_sizes):
        raise ValueError()

    MAX = len(kernel_sizes) if max_value is not None else max_value

    kernels = [create_kernel(k) for k in kernel_sizes]

    equivalent_kernel_size = [
        _get_dilation_kernel_size(k, d) for k, d in zip(kernel_sizes, dilations)
    ]

    pad_kernels = [pad_with_dilation(k, d) for k, d in zip(kernels, dilations)]

    merged_kernel = merge_kernels(pad_kernels, equivalent_kernel_size)

    os.makedirs(save_dir, exist_ok=True)

    for i, k, d in zip(pad_kernels, kernel_sizes, dilations):
        save_kernel_to_image(
            os.path.join(save_dir, f"kernel_size_{k}_dilation_{d}.png"),
            i,
            annot,
            **kwargs,
        )
    save_kernel_to_image(
        os.path.join(save_dir, "merged_kernel.png"), merged_kernel, annot, **kwargs
    )
