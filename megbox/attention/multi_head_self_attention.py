from typing import Optional

import megengine as mge
import megengine.module as M
import numpy as np
from megengine import Parameter, Tensor
from megengine.functional import (broadcast_to, concat, dropout, full, linear,
                                  logical_or, matmul, repeat, softmax, split,
                                  where, zeros)


def multi_head_attention(  # pylint: disable=too-many-statements, too-many-branches
    query: Tensor,
    key: Optional[Tensor],
    value: Optional[Tensor],
    head_dim: int,
    num_heads: int,
    attn_output_weight: Tensor,
    attn_output_bias: Optional[Tensor],
    dropout_p: float = 0,
    out_dropout: float = 0.0,
    in_proj_weight: Optional[Tensor] = None,
    query_weight: Optional[Tensor] = None,
    key_weight: Optional[Tensor] = None,
    value_weight: Optional[Tensor] = None,
    in_proj_bias: Optional[Tensor] = None,
    query_bias: Optional[Tensor] = None,
    key_bias: Optional[Tensor] = None,
    value_bias: Optional[Tensor] = None,
    bias_k: Optional[Tensor] = None,
    bias_v: Optional[Tensor] = None,
    add_zero_attn: bool = False,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    compute_mode: str = "default",
):
    tgt_len = query.shape[0]
    bsz = query.shape[1]

    # Do all the linear projections in batch
    if in_proj_weight is not None:
        query, key, value = split(
            linear(query, in_proj_weight, in_proj_bias, compute_mode), 3, axis=-1
        )
    else:
        assert (
            query_weight is not None
            and query_bias is not None
            and key_weight is not None
        )
        q = query
        query = linear(q, query_weight, query_bias, compute_mode)
        key = linear(q if key is None else key, key_weight, key_bias, compute_mode)
        value = linear(
            q if value is None else value, value_weight, value_bias, compute_mode
        )
    # add bias along batch dimension
    if bias_k is not None and bias_v is not None:
        key = concat([key, repeat(bias_k, bsz, axis=1)])
        value = concat([value, repeat(bias_v, bsz, axis=1)])
        if attn_mask is not None:
            attn_mask_temp = full(
                (attn_mask.shape[0], attn_mask.shape[1], 1),
                False,
                dtype=bool,
                device=attn_mask.device,
            )
            attn_mask = concat([attn_mask, attn_mask_temp], axis=2)
        if key_padding_mask is not None:
            key_padding_mask_temp = full(
                (key_padding_mask.shape[0], 1),
                False,
                dtype=bool,
                device=key_padding_mask.device,
            )
            key_padding_mask = concat([key_padding_mask, key_padding_mask_temp], axis=1)

    query = query.reshape(-1, bsz * num_heads, head_dim).transpose(1, 0, 2)
    key = key.reshape(-1, bsz * num_heads, head_dim).transpose(1, 0, 2)
    value = value.reshape(-1, bsz * num_heads, head_dim).transpose(1, 0, 2)
    # add zero attention along batch dimension
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        key = concat(
            [key, zeros(zero_attn_shape, dtype=key.dtype)],
            axis=1,
            device=key.device,
        )
        value = concat(
            [value, zeros(zero_attn_shape, dtype=value.dtype)],
            axis=1,
            device=value.device,
        )
        if attn_mask is not None:
            attn_mask_temp = full(
                (attn_mask.shape[0], attn_mask.shape[1], 1),
                False,
                dtype=bool,
                device=attn_mask.device,
            )
            attn_mask = concat([attn_mask, attn_mask_temp], axis=2)
        if key_padding_mask is not None:
            key_padding_mask_temp = full(
                (key_padding_mask.shape[0], 1),
                False,
                dtype=bool,
                device=key_padding_mask.device,
            )
            key_padding_mask = concat([key_padding_mask, key_padding_mask_temp], axis=1)
    # update source sequence length after adjustments
    src_len = key.shape[1]

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape[0] == bsz
        assert key_padding_mask.shape[1] == src_len
        key_padding_mask = key_padding_mask.reshape(bsz, 1, 1, src_len)
        key_padding_mask = broadcast_to(
            key_padding_mask, (bsz, num_heads, 1, src_len)
        ).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = Tensor(logical_or(attn_mask, key_padding_mask))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == np.bool:
        new_attn_mask = where(
            attn_mask, full(attn_mask.shape, -1e9), full(attn_mask.shape, 0.0)
        )
        attn_mask = new_attn_mask

    # Apply attention on all the projected vectors in batch.
    attn_output_weights = matmul(
        query, key.transpose(0, 2, 1), compute_mode=compute_mode
    ) / (head_dim**0.5)
    if attn_mask is not None:
        attn_output_weights = attn_output_weights + attn_mask

    attn_output_weights = attn_output_weights.reshape(bsz * num_heads, tgt_len, src_len)
    attn_output_weights = softmax(attn_output_weights, axis=-1)
    if dropout_p > 0.0:
        attn_output_weights = dropout(attn_output_weights, dropout_p)
    attn_output = matmul(attn_output_weights, value, compute_mode=compute_mode)

    # "Concat" using a reshape and apply a final linear.
    attn_output = attn_output.transpose(1, 0, 2).reshape(
        tgt_len, bsz, num_heads * head_dim
    )
    attn_output_weights = (
        attn_output_weights.reshape(bsz, num_heads, tgt_len, src_len).sum(axis=1)
        / num_heads
    )
    attn_output = linear(
        attn_output, attn_output_weight, attn_output_bias, compute_mode
    )
    if out_dropout > 0.0:
        attn_output = dropout(attn_output, out_dropout)
    if need_weights:
        return attn_output, attn_output_weights
    else:
        return attn_output, None


class MultiheadAttention(M.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        drop_out: float = 0.0,
        out_dropout: float = 0.0,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.drop_out = drop_out
        self.out_dropout = out_dropout
        self.add_zero_attn = add_zero_attn

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim({embed_dim}) must be divisible by num_heads({num_heads})"
        if self._qkv_same_embed_dim:
            self.in_proj = M.Linear(embed_dim, 3 * embed_dim, bias=bias)
        else:
            raise NotImplementedError()
            self.q_proj = M.Linear(
                embed_dim, embed_dim, bias=bias
            )  # pylint: disable=unreachable
            self.k_proj = M.Linear(embed_dim, self.kdim, bias=bias)
            self.v_proj = M.Linear(embed_dim, self.vdim, bias=bias)

        self.out_proj = M.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(mge.random.normal(size=(1, 1, embed_dim)))
            self.bias_v = Parameter(mge.random.normal(size=(1, 1, embed_dim)))
        else:
            self.bias_k = self.bias_v = None

        self._init_parameters()

    def _init_parameters(self):
        if hasattr(self, "in_proj"):
            M.init.xavier_uniform_(self.in_proj.weight)
            if self.in_proj.bias is not None:
                M.init.zeros_(self.in_proj.bias)
        else:
            M.init.xavier_uniform_(self.q_proj.weight)
            M.init.xavier_uniform_(self.k_proj.weight)
            M.init.xavier_uniform_(self.v_proj.weight)
            if self.q_proj.bias is not None:
                M.init.zeros_(self.q_proj.bias)
                M.init.zeros_(self.k_proj.bias)
                M.init.zeros_(self.v_proj.bias)
        M.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tensor:
        if self._qkv_same_embed_dim:
            return multi_head_attention(
                q,
                k,
                v,
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                attn_output_weight=self.out_proj.weight,
                attn_output_bias=self.out_proj.bias,
                dropout_p=self.drop_out,
                in_proj_weight=self.in_proj.weight,
                in_proj_bias=self.in_proj.bias,
                bias_k=self.bias_k,
                bias_v=self.bias_v,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=need_weights,
            )
        else:
            return multi_head_attention(
                q,
                k,
                v,
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                query_weight=self.q_proj.weight,
                query_bias=self.q_proj.bias,
                key_weight=self.k_proj.weight,
                key_bias=self.k_proj.bias,
                value_weight=self.v_proj.weight,
                value_bias=self.v_proj.bias,
                attn_output_weight=self.out_proj.weight,
                attn_output_bias=self.out_proj.bias,
                dropout_p=self.drop_out,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=need_weights,
            )
