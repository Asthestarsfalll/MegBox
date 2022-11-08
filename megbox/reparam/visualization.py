import os
from typing import Sequence, Union

import cv2
import megengine.functional as F
import numpy as np

from .utils import _get_dilation_kernel_size, merge_kernels, pad_with_dilation


def create_kernel(kernel_size):
    return F.ones((1, 1, kernel_size, kernel_size))


def convert_kernel_to_image(kernel, point_size):
    image = kernel[0, 0].numpy()
    image = image / np.max(image) * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    size = image.shape[0] * point_size
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return image


def visualize(
    kernel_sizes: Union[int, Sequence[int]],
    dilations: Union[int, Sequence[int]],
    point_size: int = 20,
    save_dir: str = "./",
):
    kernel_sizes = [kernel_sizes] if isinstance(kernel_sizes, int) else kernel_sizes
    dilations = [dilations] if isinstance(dilations, int) else dilations

    kernels = [create_kernel(k) for k in kernel_sizes]

    equivalent_kernel_size = [
        _get_dilation_kernel_size(k, d) for k, d in zip(kernel_sizes, dilations)
    ]

    pad_kernels = [pad_with_dilation(k, d) for k, d in zip(kernels, dilations)]

    merged_kernel = merge_kernels(pad_kernels, equivalent_kernel_size)

    images = [convert_kernel_to_image(k, point_size) for k in pad_kernels]
    merged_image = convert_kernel_to_image(merged_kernel, point_size)

    os.makedirs(save_dir, exist_ok=True)

    for i, k, d in zip(images, kernel_sizes, dilations):
        cv2.imwrite(os.path.join(save_dir, f"kernel_size_{k}_dilation_{d}.png"), i)
    cv2.imwrite(os.path.join(save_dir, "merged_kernel.png"), merged_image)
