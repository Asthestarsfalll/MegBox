from typing import Optional

import megengine.functional as F
from megengine import Tensor
from megengine.module import GELU, LayerNorm, Module

from megbox.attention.multi_head_self_attention import MultiheadAttention
from megbox.block import Mlp
from megbox.block.arch import TransformerArch
from megbox.types import ModuleType
from megbox.utils.msic import hack_module


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


class TransformerBlock(TransformerArch):
    def __init__(
        self,
        dim: int,
        num_heads: int,  # pylint: disable=unused-argument
        mlp_ratio: float = 4.0,  # pylint: disable=unused-argument
        qkv_bias: bool = False,  # pylint: disable=unused-argument
        mlp_drop: float = 0.0,  # pylint: disable=unused-argument
        attn_drop: float = 0.0,  # pylint: disable=unused-argument
        attn_out_drop: float = 0.0,  # pylint: disable=unused-argument
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: ModuleType = GELU,  # pylint: disable=unused-argument
        norm_layer: ModuleType = LayerNorm,
    ) -> None:
        hack_module(self)
        self.norm_layer = norm_layer
        self.closure = locals()
        super().__init__(dim, drop_path, init_values)
        delattr(self, "closure")
        delattr(self, "norm_layer")

    def _get_closure_var(self, names):
        return [self.closure[n] for n in names]

    def _build_pre_norm(self) -> Optional[Module]:
        return self.norm_layer(self.dim)

    def _build_post_norm(self) -> Optional[Module]:
        return self.norm_layer(self.dim)

    def _build_mlp(self):
        var_names = ["act_layer", "mlp_drop", "mlp_ratio"]
        act_layer, mlp_drop, mlp_ratio = self._get_closure_var(var_names)
        return Mlp(self.dim, int(self.dim * mlp_ratio), self.dim, act_layer, mlp_drop)

    def _build_attention_module(self):
        var_names = ["num_heads", "qkv_bias", "attn_drop", "attn_out_drop"]
        num_heads, qkv_bias, attn_drop, attn_out_drop = self._get_closure_var(var_names)
        return MultiheadAttention(
            self.dim,
            num_heads=num_heads,
            bias=qkv_bias,
            drop_out=attn_drop,
            out_dropout=attn_out_drop,
        )
