from typing import Optional

import megengine.functional as F
from megengine import Parameter, Tensor
from megengine.module import GELU, Dropout, Identity, LayerNorm, Linear, Module

from ..attention.multi_head_self_attention import MultiheadAttention
from ..module.drop_path import DropPath


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """
    Take and adapt from huggingface/transformers.
    """

    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.shape[-1])
        filter_indices = (
            logits < F.topk(logits, top_k, descending=True)[0][..., -1, None]
        )
        logits[filter_indices] = filter_value

    if 0.0 <= top_p <= 1.0:
        sorted_logits, sorted_indices = F.sort(logits, descending=False)

        cumulative_probs = F.cumsum(F.softmax(sorted_logits, axis=-1), axis=-1)
        sorted_indices_to_filter = cumulative_probs <= 1 - top_p

        if min_tokens_to_keep > 1:
            sorted_indices_to_filter[..., -min_tokens_to_keep] = 0

        filter_indices = F.scatter(
            sorted_indices_to_filter,
            axis=1,
            index=sorted_indices,
            source=sorted_indices_to_filter,
        )

        logits[filter_indices] = filter_value

    return logits


class LayerScale(Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
    ) -> None:
        super().__init__()
        self.gamma = Parameter(init_values * F.ones((dim)))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Mlp(Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Module = GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        mlp_drop: float = 0.0,
        attn_drop: float = 0.0,
        attn_out_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Module = GELU,
        norm_layer: Module = LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiheadAttention(
            dim,
            num_heads=num_heads,
            bias=qkv_bias,
            drop_out=attn_drop,
            out_dropout=attn_out_drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else Identity()
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=mlp_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
