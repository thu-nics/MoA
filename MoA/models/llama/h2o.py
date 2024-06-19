import math
from typing import Optional, Tuple, Union
from types import MethodType

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv

def H2O_reset_masks(self):
    self.attention_masks_next = None 
    # self.heavy_budget = None
    # self.recent_budget = None
    # self.cache_budget = None
    self.previous_scores = None

def H2O_shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
    return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()


def H2O_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]

    if q_len > 1:
        # prefill
        self._reset_masks()

    if q_len > 1:
        assert self.attention_masks_next is None

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
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # assert self.num_key_value_groups == 1
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )
    
    if attention_mask is not None:
        assert attention_mask.dtype == attn_weights.dtype

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        # attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))


    if self.attention_masks_next is not None:
        attn_weights = attn_weights * self.attention_masks_next + (1 - self.attention_masks_next) * torch.finfo(attn_weights.dtype).min

    # # upcast attention to fp32
    # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # upcast attention to fp32
    if q_len < 8192:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    elif q_len < 16384:
        attn_weights[:, :self.num_heads // 2, :, :] = nn.functional.softmax(attn_weights[:, :self.num_heads // 2, :, :], dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights[:, self.num_heads // 2:, :, :] = nn.functional.softmax(attn_weights[:, self.num_heads // 2:, :, :], dim=-1, dtype=torch.float32).to(query_states.dtype)
    else:
        attn_weights[:, :self.num_heads // 3, :, :] = nn.functional.softmax(attn_weights[:, :self.num_heads // 3, :, :], dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights[:, self.num_heads // 3:2 * self.num_heads // 3, :, :] = nn.functional.softmax(attn_weights[:, self.num_heads // 3:2 * self.num_heads // 3, :, :], dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights[:, 2 * self.num_heads // 3:, :, :] = nn.functional.softmax(attn_weights[:, 2 * self.num_heads // 3:, :, :], dim=-1, dtype=torch.float32).to(query_states.dtype)

    # attn_weights (BS, heads, q-tokens, k-tokens) 16, 15, 15 // 16, 1, 16
    if q_len < 8192:
        current_scores_sum = attn_weights.sum(0).sum(1) # (heads, k-tokens)
    elif q_len < 16384:
        current_scores_sum = torch.zeros(self.num_heads, kv_seq_len, dtype=attn_weights.dtype, device=attn_weights.device)
        current_scores_sum[:self.num_heads // 2, :] = attn_weights[:, :self.num_heads // 2, :, :].sum(0).sum(1)
        current_scores_sum[self.num_heads // 2:, :] = attn_weights[:, self.num_heads // 2:, :, :].sum(0).sum(1)
    else:
        current_scores_sum = torch.zeros(self.num_heads, kv_seq_len, dtype=attn_weights.dtype, device=attn_weights.device)
        current_scores_sum[:self.num_heads // 3, :] = attn_weights[:, :self.num_heads // 3, :, :].sum(0).sum(1)
        current_scores_sum[self.num_heads // 3:2 * self.num_heads // 3, :] = attn_weights[:, self.num_heads // 3:2 * self.num_heads // 3, :, :].sum(0).sum(1)
        current_scores_sum[2 * self.num_heads // 3:, :] = attn_weights[:, 2 * self.num_heads // 3:, :, :].sum(0).sum(1)
    # offset = attn_weights.gt(0).sum(0).sum(1)
    
    # Accumulate attention scores
    if not self.previous_scores == None:
        current_scores_sum[:, :-1] += self.previous_scores #(Enlarged Sequence)
    # else:
    #     self.heavy_budget = int(self.heavy_budget_ratio * current_scores_sum.shape[-1])
    #     self.recent_budget = int(self.recent_budget_ratio * current_scores_sum.shape[-1])
    #     self.cache_budget = self.heavy_budget + self.recent_budget

        # current_scores_sum = current_scores_sum / offset
    dtype_attn_weights = attn_weights.dtype
    attn_weights_devices = attn_weights.device
    assert attn_weights.shape[0] == 1
    self.previous_scores = current_scores_sum #(heads, k-tokens)
    attn_mask = torch.ones(current_scores_sum.shape[0], current_scores_sum.shape[1]+1, dtype=dtype_attn_weights, device=attn_weights_devices)

    attn_tokens_all = self.previous_scores.shape[-1]

    if attn_tokens_all > self.cache_budget:
        # activate most recent k-cache
        if not self.recent_budget == 0:
            attn_mask[:, :-self.recent_budget] = 0
            selected_set = self.previous_scores[:, :-self.recent_budget]
        else:
            # activate historical best self.cache_budget - self.recent_budget tokens.
            # self.previous_scores # (k-Cache - 1)
            selected_set = self.previous_scores

        if not self.heavy_budget == 0:
            _, keep_topk = selected_set.topk(k=self.heavy_budget, dim=-1, largest=True)
            attn_mask = attn_mask.scatter(-1, keep_topk, 1)

    self.attention_masks_next = attn_mask.unsqueeze(0).unsqueeze(2)

    score_mask = attn_mask[:,:-1]
    score_mask[:, -self.recent_budget:] = 1
    self.previous_scores = self.previous_scores * score_mask

    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def convert_kvcache_llama_heavy_recent(model, heavy_budget, recent_budget):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_llama_heavy_recent(module, heavy_budget, recent_budget)

        if isinstance(module, LlamaAttention):
            model._modules[name].forward = MethodType(H2O_attention_forward, model._modules[name])
            model._modules[name]._reset_masks = MethodType(H2O_reset_masks, model._modules[name])
            model._modules[name]._shape = MethodType(H2O_shape, model._modules[name])
            # model._modules[name].heavy_budget_ratio = heavy_ratio
            # model._modules[name].recent_budget_ratio = recent_ratio
            model._modules[name].attention_masks_next = None
            model._modules[name].heavy_budget = heavy_budget
            model._modules[name].recent_budget = recent_budget
            model._modules[name].cache_budget = heavy_budget + recent_budget
            model._modules[name].previous_scores = None
    return model