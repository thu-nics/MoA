"""
Mixture of Sparse Attention
===============

This is a Triton implementation of the Mixture of Sparse Attention (MoA) kernel.

"""
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
import math
import triton
import triton.language as tl

from MoA.attention.cache_utils import StaticCircularCache

class _mixture_of_sparse_attention(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx, q, k, v, sm_scale, head_index = None,
            attention_mask = None, attention_dropout = 0.0
        ) -> torch.Tensor:
        """
        # During Prefill
        input:
            q: (Z, H, N_IN, L)
            k, v: (Z, H, N_CTX, L)
            sm_scale: float
        output:
            o: (Z, H, N_IN, L)

        # During Decode
        input:
            q: (Z, H, N_IN, L)
            k, v: (Z, \sum_i^H N_CTX, L)
            sm_scale: float
            head_index: (H+1)
        output:
            o: (Z, H, N_IN, L)
        """
        dtype = q.dtype
        assert dtype == torch.float16

        bsz, num_heads, q_len, _ = q.shape
        _is_decode = (q_len == 1)

        # decode
        if _is_decode:
            # TODO: support contigious cache memory
            # split group and calculate for now, converting to [(Z, H, N_CTX, L)] * group for k and v
            k = StaticCircularCache.to_group_contigious(k, head_index)
            v = StaticCircularCache.to_group_contigious(v, head_index)
            attention_mask = StaticCircularCache.to_group_contigious(attention_mask, head_index)

            num_group = len(k)
            if num_group > 2:
                Warning("num_group > 2 is not recommended for efficiency reason")
            assert len(k) == len(v)
            start_index = 0
            attn_output = []
            for group_id in range(num_group):
                bsz, num_heads, cache_size, hidden_dim = k[group_id].shape
                end_index = start_index + num_heads

                this_q = q[:, start_index:end_index, :, :]
                this_k = k[group_id]
                this_v = v[group_id]
                this_attention_mask = attention_mask[group_id][:, 0, :] # attention masks are the same within each group

                this_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    this_attention_mask,
                    (bsz, q_len),
                    q,
                    past_key_values_length=cache_size - q_len,
                )

                this_attn_output = F.scaled_dot_product_attention(
                    this_q,
                    this_k,
                    this_v,
                    this_attention_mask,
                    attention_dropout,
                    is_causal=attention_mask is None and q_len > 1,
                    scale=sm_scale,
                )

                attn_output.append(this_attn_output)
                start_index = end_index
            
            attn_output = torch.cat(attn_output, dim=1)

            return attn_output

        # prefill
        else:
            # shape constraints
            Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
            assert Lq == Lk and Lk == Lv
            kv_seq_len = k.size(2)

            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (bsz, q_len),
                q,
                past_key_values_length=kv_seq_len - q_len,
            )

            return F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=attention_mask,
                dropout_p=attention_dropout,
                is_causal=attention_mask is None and q_len > 1,
                scale=sm_scale,
            )
            # return sparse_attention_prefill(q, k, v, sm_scale, lut, BLOCK_M, BLOCK_N)

def mixture_of_sparse_attention(query, key, value, sm_scale, head_index = None,
            attention_mask = None, attention_dropout = 0.0):
    """
    Wrapper for the Triton implementation of the Mixture of Sparse Attention (MoA) kernel to support keyword arguments.
    """
    # TODO: support contigious cache memory
    


    return _mixture_of_sparse_attention.apply(query, key, value, sm_scale, head_index, attention_mask, attention_dropout)