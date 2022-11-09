import os
from typing import Optional, Sequence, Union

import megengine.functional as F
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from megengine import Tensor

from .utils import _get_dilation_kernel_size, merge_kernels, pad_with_dilation, zero_padding


def create_kernel(kernel_size: int) -> Tensor:
    return F.ones((1, 1, kernel_size, kernel_size))


def save_kernel_to_image(
    name: str, kernel: Tensor, max_value: int, annot: bool = True, save_dpi: int = 400, **kwargs
) -> None:
    image = kernel[0, 0].numpy().astype(np.uint8)
    heatmap_kwargs = dict(
        data=image,
        fmt='d',
        annot=annot,
        vmin=0,
        vmax=max_value,
        linewidths=0.5,
    )
    heatmap_kwargs.update(kwargs)
    heatmap = sns.heatmap(**heatmap_kwargs)
    fig = heatmap.get_figure()
    fig.savefig(name, dpi=save_dpi)
    plt.close()


def visualize(
    kernel_sizes: Union[int, Sequence[int]],
    dilations: Union[int, Sequence[int]],
    *,
    fix_grid: bool = False,
    annot: bool = True,
    save_dir: str = "./",
    save_dpi: int = 400,
    **kwargs,
) -> None:

    kernel_sizes = [kernel_sizes] if isinstance(
        kernel_sizes, int) else kernel_sizes
    dilations = [dilations] if isinstance(dilations, int) else dilations

    max_value = len(kernel_sizes)

    kernels = [create_kernel(k) for k in kernel_sizes]

    equivalent_kernel_size = [
        _get_dilation_kernel_size(k, d) for k, d in zip(kernel_sizes, dilations)
    ]

    pad_kernels = [pad_with_dilation(k, d) for k, d in zip(kernels, dilations)]
    if fix_grid:
        max_kernel = max(equivalent_kernel_size)
        padding_size = [(max_kernel - k) // 2 for k in equivalent_kernel_size]
        pad_kernels = [zero_padding(k, p)
                       for k, p in zip(pad_kernels, padding_size)]
        equivalent_kernel_size = [max_kernel] * max_value

    merged_kernel = merge_kernels(pad_kernels, equivalent_kernel_size)

    os.makedirs(save_dir, exist_ok=True)

    for i, k, d in zip(pad_kernels, kernel_sizes, dilations):
        save_kernel_to_image(
            os.path.join(save_dir, f"kernel_size_{k}_dilation_{d}.png"),
            kernel=i,
            max_value=max_value,
            annot=annot,
            save_dpi=save_dpi,
            **kwargs,
        )
    save_kernel_to_image(
        os.path.join(save_dir, "merged_kernel.png"), merged_kernel, max_value, annot, save_dpi, **kwargs
    )
