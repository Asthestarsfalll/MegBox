import math

import cv2
import megengine.functional as F
import numpy as np
from megengine import Tensor


def chw_to_hwc(x: Tensor) -> np.ndarray:
    return x.detach().numpy().transpose(1, 2, 0)


def convert_tensor_to_image(image: Tensor) -> np.ndarray:
    assert image.ndim == 3
    image = image * 255 + 0.5
    image = chw_to_hwc(F.clip(image, 0, 255)).astype("uint8")[0]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def convert_segmentation_to_image(seg: Tensor, color_seed: int = 1) -> np.ndarray:
    seg = chw_to_hwc(seg)[:, :, None, :]
    colorize = np.random.RandomState(color_seed).randn(1, 1, seg.shape[-1], 3)
    colorize = colorize / colorize.sum(axis=2, keepdims=True)
    seg = seg @ colorize
    seg = seg[..., 0, :]
    seg = ((seg + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return seg


def convert_image_to_tensor(x: np.ndarray) -> Tensor:
    assert len(x.shape) == 3
    x = Tensor(x.transpose(2, 0, 1), dtype="float32")
    x = F.expand_dims(x, axis=0)
    return x


def save_images(
    images: Tensor,
    filename: str,
    nrow: int = 8,
    padding: int = 2,
    image_type: str = "image",
) -> None:
    if images.ndim == 4 and images.shape[0] != 1:
        total_num, C, h, w = images.shape
        # save images in one picture
        num_x = min(nrow, total_num)
        num_y = int(math.ceil(float(total_num) / num_x))
        H, W = int(h + padding), int(w + padding)
        grid = F.full((C, H * num_y + padding, W * num_x + padding), 0.0)
        cur_num = 0
        for y in range(num_y):
            for x in range(num_x):
                if cur_num >= total_num:
                    break
                # fmt: off
                grid[
                    :,
                    y * H + padding: (y + 1) * H,
                    x * W + padding: (x + 1) * W,
                ] = images[cur_num, ...]
                # fmt: on
                cur_num += 1
    elif images.ndim == 3:
        grid = images
    elif images.shape[0] == 1:
        grid = images[0]
    else:
        raise ValueError("The input must have 3 or 4 dimension.")
    if image_type == "image":
        image = convert_tensor_to_image(grid)
    elif image_type == "segmentation":
        image = convert_segmentation_to_image(grid)
    else:
        raise ValueError()
    cv2.imwrite(filename, image)
