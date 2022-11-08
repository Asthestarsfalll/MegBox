from typing import Optional

from megengine.functional import flatten
from megengine.module import Conv2d, Identity, Module

from ..utils.msic import to_2tuple


class PatchEmbed(Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Module] = None,
        flatten=True,
    ):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size does not match, expected {self.img_size}, got {(H, W)}"
        x = self.proj(x)
        if self.flatten:
            x = flatten(x, 2).transpose(0, 2, 1)  # BCHW -> BNC
        x = self.norm(x)
        return x
