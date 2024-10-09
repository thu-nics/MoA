"""
Mixture of Sparse Attention
===============

This is the interface for the Mixture of Sparse Attention (MoA) kernels.

"""

import torch
from torch import Tensor, LongTensor
from torch import nn
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from typing import Optional, Union, List

from MoA.attention.cache_utils import StaticCircularCache

from MoA.kernels.flash_decoding_moa import _mixture_of_sparse_attention_decode
from MoA.kernels.block_sparse_attention_prefill import _sparse_attention_moa_prefill
try:
    from flashinfer import moa_prefill, moa_decode
    CUDA_KERNEL_AVAILABLE = True
except ImportError:
    print("Module 'flashinfer' is not available. Efficient sparse decoding is not supported. Switching to the Triton implementation")
    if torch.cuda.device_count() > 1:
        raise NotImplementedError("Please install the MoA CUDA kernel for inference on multiple GPUs")
    CUDA_KERNEL_AVAILABLE = False

"""
used by FlashAttention2
"""
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


def _flash_attention_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    dropout=0.0,
    softmax_scale=None,
    is_causal=True,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`int`, *optional*):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
    """

    causal = is_causal and query_length != 1

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = (
            _upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )

        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    else:
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )

    return attn_output


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _upad_input(query_layer, key_layer, value_layer, attention_mask, query_length):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
        indices_k,
    )
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
        indices_k,
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(
            query_layer.reshape(batch_size * kv_seq_len, -1, head_dim), indices_k
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
            query_layer, attention_mask
        )

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


"""
Example of how MoA attention can be implemented using FlashAttention
"""

class _adapt_mixture_of_sparse_attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        sm_scale: float,
        head_index: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        attention_dropout: float = 0.0,
        implementation: Optional[str] = "sdpa",
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
        _is_decode = q_len == 1

        # _attn_implementation: "flash_attention2" or "sdpa"
        _prefill_attn_implementation = implementation
        _decode_attn_implementation = implementation

        # decode
        if _is_decode:
            assert len(k.shape) == 3 and len(v.shape) == 3  # (Z, \sum_i^H N_CTX, L)
            # TODO: support contigious cache memory
            # split group and calculate for now, converting to [(Z, H, N_CTX, L)] * group for k and v
            k = StaticCircularCache.to_group_contigious(k, head_index)
            v = StaticCircularCache.to_group_contigious(v, head_index)
            attention_mask: Union[List[Tensor], None] = (
                StaticCircularCache.to_group_contigious(attention_mask, head_index)
                if attention_mask is not None
                else None
            )

            num_group = len(k)
            if num_group > 2:
                Warning("num_group > 2 is not recommended for efficiency reason")
            assert len(k) == len(v)
            start_index = 0
            attn_output = []
            for group_id in range(num_group):
                bsz, num_heads, cache_size, hidden_dim = k[group_id].shape
                end_index = start_index + num_heads

                this_q = q[:, start_index:end_index, :, :] # shape (bsz, num_heads, q_len, hidden_dim)
                this_k = k[group_id] # shape (bsz, num_heads, cache_size, hidden_dim)
                this_v = v[group_id] # shape (bsz, num_heads, cache_size, hidden_dim)
                this_attention_mask = (
                    attention_mask[group_id][:, 0, :]
                    if attention_mask is not None
                    else None
                )  # attention masks are the same within each group

                if _decode_attn_implementation == "sdpa":
                    # use sdpa
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
                        is_causal=this_attention_mask is None and q_len > 1,
                        scale=sm_scale,
                    )  # shape (bsz, num_heads, q_len, hidden_dim)

                elif _decode_attn_implementation == "flash_attention2":
                    # use flashAttention2
                    this_q = this_q.transpose(1, 2).contiguous()
                    this_k = this_k.transpose(1, 2).contiguous()
                    this_v = this_v.transpose(1, 2).contiguous()
                    this_attention_mask = (
                        this_attention_mask.contiguous()
                        if this_attention_mask is not None
                        else None
                    )

                    this_attn_output = _flash_attention_forward(
                        this_q,
                        this_k,
                        this_v,
                        this_attention_mask,
                        q_len,
                        dropout=attention_dropout,
                        softmax_scale=sm_scale,
                        is_causal=attention_mask is None and q_len > 1,
                    )  # shape (bsz, q_len, num_heads, hidden_dim)

                else:
                    raise NotImplementedError

                attn_output.append(this_attn_output)
                start_index = end_index

            if _decode_attn_implementation == "sdpa":
                attn_output = torch.cat(attn_output, dim=1).transpose(1, 2)
            elif _decode_attn_implementation == "flash_attention2":
                attn_output = torch.cat(attn_output, dim=2)
            else:
                raise NotImplementedError

            return attn_output

        # prefill
        else:
            # shape constraints
            Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
            assert Lq == Lk and Lk == Lv
            kv_seq_len = k.size(2)

            if _prefill_attn_implementation == "sdpa":
                # use sdpa attention
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
                ).transpose(1, 2)
            
            elif _prefill_attn_implementation == "flash_attention2":
                # use flash attention 2
                q = q.transpose(1, 2).contiguous()
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()

                return _flash_attention_forward(
                    q,
                    k,
                    v,
                    attention_mask,
                    q_len,
                    dropout=attention_dropout,
                    softmax_scale=sm_scale,
                    is_causal=True,
                )
            else:
                raise NotImplementedError
            # return sparse_attention_prefill(q, k, v, sm_scale, lut, BLOCK_M, BLOCK_N)

def mixture_of_sparse_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    sm_scale: float,
    attention_mask: LongTensor = None,
    attention_dropout: float = 0.0,
    head_start_index: LongTensor=None,
    head_valid_length: LongTensor=None,
    sink_size: LongTensor=None,
    local_size: LongTensor=None,
    implementation: str = "moa",
):
    """
    Wrapper for the different implementation of the Mixture of Sparse Attention (MoA) kernel to support keyword arguments. The default implementation is `"moa"`.
    The attention is default to be "causal".

    Args:
        query (`torch.Tensor`):
            The query tensor of shape `(batch_size, num_heads, query_length, head_dim)`.
        key, value (`torch.Tensor`):
            For prefill, the key and value tensors are of shape `(batch_size, num_heads, key_length, head_dim)`.
            For sparse decode, the key and value tensors are of shape `(batch_size, \sum_H num_cache_heads_h, head_dim)`, which means different cache heads are concatenated as a single dimention.
        sm_scale (`float`):
            The scaling factor for the attention scores.
        attention_mask (`torch.Tensor`, *optional*):
            TBD. Currently, do not support customized attention mask
        attention_dropout (`float`, *optional*):
            The dropout rate for the attention scores.
        implementation (`str`, *optional*):
            The implementation of the MoA kernel to use. Defaults to `"moa"`.
        head_start_index (`torch.Tensor`, *optional*):
            The starting index of the heads in the cache of shape `(batch_size, num_heads)`.
        head_valid_length (`torch.Tensor`, *optional*):
            The valid length of the heads in the cache of shape `(batch_size, num_heads)`.
        sink_size (`torch.Tensor`, *optional*):
            The sink size for the MoA kernel.
        local_size (`torch.Tensor`, *optional*):
            The local size for the MoA kernel.

    Returns:
        The output tensor of shape `(batch_size, query_length, num_heads, head_dim)`.
    """

    is_prefill = not (key.shape[-2] > query.shape[-2])

    if implementation in ["sdpa", "flash_attention2"]: # noqa, only used for debug purpose
        if is_prefill:
            head_index = None
        else:
            head_index = StaticCircularCache.head_start_index_valid_length_to_head_index(head_start_index, head_valid_length)
        # use dense prefill and sparse decode with flash attention 1 or 2
        return _adapt_mixture_of_sparse_attention.apply(
            query, key, value, sm_scale, head_index, attention_mask, attention_dropout, implementation
        )

    elif implementation == "moa":
        if is_prefill:
            # note that qkv are not contiguous for prefill

            BLOCK_M = 64
            BLOCK_N = 64
            """
            CUDA Implementation of MoA sparse prefill
            """
            if CUDA_KERNEL_AVAILABLE:
                return moa_prefill(
                    query.transpose(1, 2).contiguous(),
                    key,
                    value,
                    causal=True,
                    num_global_blocks=sink_size // BLOCK_N,
                    num_band_blocks=local_size // BLOCK_N,
                    kv_layout="HND",
                )

            """
            Triton Implementation of MoA sparse prefill
            """
            if not CUDA_KERNEL_AVAILABLE:
                return _sparse_attention_moa_prefill.apply(
                    query,
                    key,
                    value,
                    sm_scale,
                    sink_size // BLOCK_N,
                    local_size // BLOCK_N,
                    BLOCK_M,
                    BLOCK_N,
                ).transpose(1, 2).contiguous()

        else:
            """
            CUDA Implementation of MoA sparse decode
            """
            if CUDA_KERNEL_AVAILABLE:
                return moa_decode(
                    query.transpose(1, 2).contiguous(),
                    key,
                    value,
                    head_start_index,
                    head_valid_length,
                    sm_scale,
                )

            """
            Triton Implementation of MoA sparse decode
            """
            if not CUDA_KERNEL_AVAILABLE:
                raise NotImplementedError("Please install the MoA CUDA kernel for inference")
                causal = True
                head_index = StaticCircularCache.head_start_index_valid_length_to_head_index(head_start_index, head_valid_length)
                return _mixture_of_sparse_attention_decode.apply(
                    query, key, value, head_index, sm_scale, causal
                ).transpose(1, 2).contiguous()

    else:
        raise NotImplementedError
