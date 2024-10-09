import math
from typing import List, Optional, Tuple
import warnings

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.cache_utils import Cache

from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb, rotate_half
from transformers.utils import logging

from types import MethodType

logger = logging.get_logger(__name__)


"""
used for attention analysis
"""
from collections import defaultdict

def LlamaAttention_grad_analysis_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    ### efficient profile ###
    self.query_states = query_states.detach()
    self.key_states = key_states.detach()
    self.value_states = value_states.detach()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output.retain_grad()
    self.attn_output = attn_output

    self.causal_mask = causal_mask
    ### end efficient profile ###

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

import deepspeed
from MoA.attention.pattern import gen_causal_pattern

def LlamaModel_get_attention_matrix_log(self, max_length, take_abs=False, aggregating_block_size=1):
    """
    Get the attention matrix gradient
    """
    effect_list = []

    # effects = grad_A * mat_A
    # mat_A or grad_A shape: (batch, head, query, key)
    # print('get attention matrix log')
    for layer in self.layers:
        ### recompute mat_A and grad_A on the fly ###
        query_states = layer.self_attn.query_states
        key_states = layer.self_attn.key_states

        recomputed_attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(layer.self_attn.head_dim)
        recomputed_attn_weights = recomputed_attn_weights + layer.self_attn.causal_mask
        mat_A = nn.functional.softmax(recomputed_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        mat_A.requires_grad = True
        recomputed_attn_weights = None

        recomputed_attn_output = torch.matmul(mat_A, layer.self_attn.value_states)

        # do the backward with gradient on the output to get gradient for the attention matrix
        recomputed_attn_output.backward(layer.self_attn.attn_output.grad)

        grad_A = mat_A.grad
        mat_A = mat_A.detach()
        ### end recompute ###

        recomputed_attn_output = None

        multiply = grad_A*mat_A
        # use the following delta to avoid numerical issue
        delta = torch.finfo(grad_A.dtype).eps
        effect = mat_A*(grad_A-torch.sum(multiply, dim=-1, keepdim=True))/(delta+1-mat_A)
        
        # deal with the first row of effects
        effect[..., 0, :] = multiply[..., 0, :]
        effect = -effect

        if take_abs:
            effect = torch.abs(effect)

        effect = effect[..., :max_length, :max_length]

        if aggregating_block_size > 1:
            assert effect.shape[-1] % aggregating_block_size == 0, "the attention matrix size must be divisible by the aggerating block size"
            causal_mask = gen_causal_pattern(max_length, max_length, dtype=torch.bool).squeeze().to(effect.device)
            effect = effect * causal_mask
            effect = effect.to(torch.float32)
            effect = F.avg_pool2d(effect, aggregating_block_size) * aggregating_block_size * aggregating_block_size
            causal_mask = None 
        
        # free the GPU memory
        effect = effect.cpu()
        grad_A = None
        mat_A = None

        effect_list.append(effect)
    
    effect = torch.stack(effect_list, dim=1)
    
    # if the attention matrix is larger than 4k, flush the cache
    if max_length >= 4096:
        deepspeed.get_accelerator().empty_cache()

    return {'sum_effect': effect}


def LlamaModel_use_attention_matrix_grad_log(self):
    """
    Set the model instance to use flash attention instead of llama attention
    """
    self.attention_matrix_log = defaultdict(list)
    for layer in self.layers:
        layer.self_attn.forward = MethodType(LlamaAttention_grad_analysis_forward, layer.self_attn)
    self.get_attention_matrix_log = MethodType(LlamaModel_get_attention_matrix_log, self)
