import math
from typing import List, Optional, Tuple
import warnings

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.cache_utils import Cache

from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb, rotate_half

from types import MethodType


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
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    ### efficient profile ###
    self.query_states = query_states.detach()
    self.key_states = key_states.detach()
    self.value_states = value_states.detach()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=0.0,
        scale=1 / math.sqrt(self.head_dim),
    )

    attn_output.retain_grad()
    self.attn_output = attn_output

    self.causal_mask = attention_mask
    ### end efficient profile ###

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output_transposed = attn_output.transpose(1, 2).contiguous()

    attn_output_reshaped = attn_output_transposed.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
       raise ValueError("not implemented")
    else:
        output_values = self.o_proj(attn_output_reshaped)

    if not output_attentions:
        attn_weights = None

    return output_values, attn_weights, past_key_value

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