# coding=utf-8
# Align with transformer==4.36.2
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union, Dict
import warnings

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch import Tensor

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import add_start_docstrings_to_model_forward, logging

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
import contextlib

from transformers import LlamaModel
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv, apply_rotary_pos_emb, rotate_half


from types import MethodType

from MoA.attention.convert import block_sparse_to_dense
from MoA.attention.cache_utils import StaticCircularCache, StreamingllmDynamicCache, moa_config_to_cache_config
from MoA.attention.permutation_utils import (
    permute_attention_projection, 
    permute_output_projection, 
    permute_lut, 
    lut_to_permutation, 
    get_lut_global_size,
    get_lut_band_size,
    moa_config_to_permutation,
)
from MoA.attention.density_calculation import streamingllm_attention_density, streamingllm_kv_cache_density

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


"""
efficient implementation for Mixture of Attention
"""
from MoA.kernels.mixture_of_attention import mixture_of_sparse_attention


def LlamaModel_MixtureAttention_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[StaticCircularCache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    ### prepare cache ###
    if not isinstance(past_key_values, StaticCircularCache):
        # initialize the cache
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        cache_config = moa_config_to_cache_config(
            self.moa_config,
            seq_len=seq_length,
            max_new_token=self.moa_max_new_token,
            sink_size=64,
            minimum_cache_size=128,
            split_size=64,
            verbose=self.moa_verbose if hasattr(self, "moa_verbose") else False, # set to True if you want to see the cache sizes
        )
        past_key_values = StaticCircularCache(
            **cache_config,
            batch_size=batch_size,
            head_dim=head_dim,
            device=self.device,
            dtype=self.dtype,
            update_cache_content=True
        )
    ### end of perpare cache ###

    if cache_position is None:
        past_cache_position = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_cache_position, past_cache_position + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    ### modification ###
    attention_mask = (
        attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    )  # same as flash_attention2
    if output_attentions:
        raise NotImplementedError(
            "output_attentions is not supported in mixture of attention"
        )
    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    ### end of modification ###

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

    ### only pass the hidden_states of last seq length
    # ! you can pass only the hidden_states of last seq length for better performance
    last_hidden_only = self.last_hidden_only if hasattr(self, "last_hidden_only") else False
        
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states if not last_hidden_only else hidden_states[:, -1:, :], # use this to past only the hidden_states of the last token to lm_head to reduce peak memory
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
    ######


class LlamaMixtureAttention(LlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[StaticCircularCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        ### begin modification ###
        assert (
            self.num_key_value_groups == 1
        ), "only support one key value group now, but got {}".format(
            self.num_key_value_groups
        )

        if output_attentions:
            raise NotImplementedError(
                "output_attentions is not supported in mixture of attention"
            )
        ### end of modification ###

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

        ### begin modification ###
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs=cache_kwargs)
            if q_len == 1 and key_states.shape[-2] > 1:
                # update this_attention_mask during decode
                this_attention_mask = (
                    past_key_value.mask_cache[self.layer_idx]
                    if attention_mask is not None
                    else None
                )
            else:
                # TODO: support prefill with KV-cache
                this_attention_mask = attention_mask
        else:
            this_attention_mask = attention_mask
        ### end of modification ###

        ### begin modification ###
        # the key and value states before past_key_value.update are tensors of shape (bsz, num_heads, q_len, head_dim); after updating:
        # 1. During prefill, the do not change in shape
        # 2. During decode, they become tensors of shape (bsz, \sum_h^H cache_size_of_head_h, head_dim)
        if self.num_key_value_groups > 1:
            raise NotImplementedError("only support one key value group now")
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Contiguous is necessary here because of the view call in the linear layers
        if (
            query_states.device.type == "cuda"
            and causal_mask is not None
            and isinstance(key_states, torch.Tensor)
        ):
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        ### end of modification ###

        ### begin modification ###
        assert self.training == False, "only support inference now"
        if isinstance(past_key_value, StaticCircularCache):
            # head_index = past_key_value.head_index[self.layer_idx]
            # used for prefill
            sink_size = past_key_value.static_cache_size[self.layer_idx]
            local_size = past_key_value.circular_cache_size[self.layer_idx]
            # used for decode
            head_start_index = past_key_value.cache_head_start_index[self.layer_idx]
            head_valid_length = past_key_value.cache_valid_length[self.layer_idx]
        else:
            # head_index = None
            sink_size = None
            local_size = None
            head_start_index = None
            head_valid_length = None

        # support both sparse prefill and decode
        attn_output = mixture_of_sparse_attention(
            query_states,
            key_states,
            value_states,
            sm_scale=self.head_dim**-0.5,
            attention_mask=this_attention_mask,
            attention_dropout=0.0,
            sink_size=sink_size,
            local_size=local_size,
            head_start_index=head_start_index,
            head_valid_length=head_valid_length,
        ) # shape: (bsz, q_len, num_heads, head_dim)
        ### end modification ###

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

def LlamaModel_set_mixture_of_attention(
    self,
    moa_config: Dict,
    permute_head: bool = False,
    moa_verbose: bool = False,
    moa_max_new_token: int = 1024,
    last_hidden_only: bool = True,
    sparse_prefill: bool = True,
):
    """
    Set the mixture of attention of the model

    Args:
        moa_config: a dictionary containing the configuration of the mixture of attention
            keys:
                alphas: a list of list of int, the alpha of each head in each layer
                betas: a list of list of float, the beta of each head in each layer
        permute_head: bool, whether to permute the heads to make the heads with the same cache size to be adjacent
        moa_verbose: bool, whether to print the cache size of each head
        last_hidden_only: bool, whether to pass only the hidden_states of the last token to lm_head to reduce peak memory
        sparse_prefill: bool, whether to use sparse prefill
    """
    # update forward functions
    self.forward = MethodType(LlamaModel_MixtureAttention_forward, self)

    # addtional setups
    self.moa_verbose = moa_verbose
    self.last_hidden_only = last_hidden_only
    self.moa_max_new_token = moa_max_new_token

    assert sparse_prefill, "only support sparse prefill now. modify the implementation in `mixture_of_attention` if you want to use dense prefill"

    # update functions in LlamaAttention
    for layer in self.layers:
        layer.self_attn.forward = MethodType(
            LlamaMixtureAttention.forward, layer.self_attn
        )

    alphas: Union[List[List[int]], List[Tensor]] = moa_config["alphas"]
    betas: Union[List[List[float]], List[Tensor]] = moa_config["betas"]
    # block_size: int = moa_config['block_size'] # set to 64 for now

    def permute_head_func(
        self,
        moa_config: Dict,
    ):
        alphas: Union[List[List[int]], List[Tensor]] = moa_config["alphas"]
        betas: Union[List[List[float]], List[Tensor]] = moa_config["betas"]

        permutations, clusters = moa_config_to_permutation(moa_config)
        for layer_id, layer in enumerate(self.layers):
            permutation = permutations[layer_id]
            cluster = clusters[layer_id]

            # permute alpha and beta based on the permutation
            alphas[layer_id] = [alphas[layer_id][i] for i in permutation]
            betas[layer_id] = [betas[layer_id][i] for i in permutation]

            # TODO: merge layer permute to this function
            num_heads = layer.self_attn.num_heads
            permute_attention_projection(layer.self_attn.q_proj, permutation, num_heads)
            permute_attention_projection(layer.self_attn.k_proj, permutation, num_heads)
            permute_attention_projection(layer.self_attn.v_proj, permutation, num_heads)
            permute_output_projection(layer.self_attn.o_proj, permutation, num_heads)

        moa_config["alphas"] = alphas
        moa_config["betas"] = betas

    if permute_head:
        permute_head_func(self, moa_config)

    self.use_moa = True
    self.moa_config = moa_config


"""
Deprecated
Efficient block sparse llama attention using lut
"""
from MoA.kernels.block_sparse_attention_prefill import sparse_attention

def lut_to_single_query_kv_mask(
    lut: torch.Tensor,
    query_idx: int,
    block_size: int = 64,
):
    if len(lut.size()) == 4:
        # raise warning, but proceed
        warnings.warn("The input lut has four dimensions, please make sure that lut is the same across all heads in the layer")
        lut = lut[0, 0, :, :]
    else:
        lut = lut[0]

    kv_length = query_idx + 1

    query_block_idx = query_idx // block_size
    query_block_lut = lut[query_block_idx]
    seen_id = [0 for _ in range(kv_length)]
    for i in query_block_lut:
        for j in range(i * block_size, min((i + 1) * block_size, kv_length)):
            seen_id[j] = 1

    return seen_id


def lut2layout_single_layer(lut: torch.IntTensor) -> torch.BoolTensor:
    """
    input:
        lut: (num_heads, num_block, nnz)
        num_block: the number of blocks
    output:
        layout: (num_heads, num_block, num_block)
    """
    num_block = lut.shape[1]

    assert num_block >= lut.max().item(), "The number of blocks should be larger than the maximum value in the LUT."
   
    num_head = lut.shape[0]
    layout = torch.zeros((num_head, num_block, num_block), dtype=torch.bool, device=lut.device)

    for i in range(num_head):
        for j in range(num_block):
            for k in range(lut.shape[2]):
                layout[i, j, lut[i, j, k]] = True

    return layout


def LlamaAttention_block_sparse_lut_forward_fake_sparse_decode(
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

    ## find the smallest lut in lut list that is greater than the kv_seq_len
    ## the key of self.lut_list is the supported lut len
    lut_len = min([k for k in self.lut_dict if k >= kv_seq_len])
    lut = self.lut_dict[lut_len]

    ### decoding stage: drop kv according to lut
    if q_len == 1 and self.sparse_decode:
        query_block_idx = (kv_seq_len - 1) // self.block_size
        layout = self.layout_dict[lut_len]
        current_layout = layout[:, query_block_idx:query_block_idx+1, :]
        attention_mask = current_layout.repeat_interleave(self.block_size, dim=-1)[:, :, :kv_seq_len]
        attention_mask = attention_mask.unsqueeze(0).to(torch.bool)
    ### decoding stage: drop kv according to lut

    ### efficient attention implementation ###
    attn_output = self.efficient_attention(
        # for prefill
        query_states, 
        key_states, 
        value_states, 
        self.head_dim**-0.5, 
        lut, 
        self.block_size, 
        self.block_size,
        # for decode
        attention_mask,
        self.attention_dropout if self.training else 0.0 # noqa
    )
    ### end efficient attention implementation ###

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def LlamaAttention_block_sparse_lut_forward_split_head(
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

    assert self.num_key_value_groups == 1, "only support one key value group now, but got {}".format(self.num_key_value_groups)
    divide_head = (hasattr(self, 'cluster') and (q_len == 1))

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

    # find the shortest lut that is greater than kv_seq_len
    usable_lut_len = [k for k in self.lut_dict if k >= kv_seq_len]
    if len(usable_lut_len) == 0:
        lut_len = max(self.lut_dict.keys())
        if not q_len != 1:
            raise ValueError(f"input token len exceeded at prefilling stages. kv_len: {kv_seq_len}")
        
    else:
        lut_len = min([k for k in self.lut_dict if k >= kv_seq_len])

    lut = self.lut_dict[lut_len]
    band_size = self.band_size_dict[lut_len]
    global_size = self.global_size_dict[lut_len]

    if hasattr(self, 'cluster'):
        # split the query, key, value states
        assert len(self.cluster) <= 2
        cluster0 = self.cluster[0]
        cluster1 = self.cluster[1] if len(self.cluster) == 2 else None
        query_states0 = query_states[:, cluster0[0]:cluster0[1], :, :]
        query_states1 = query_states[:, cluster1[0]:cluster1[1], :, :] if cluster1 is not None else None
        key_states0 = key_states[:, cluster0[0]:cluster0[1], :, :]
        key_states1 = key_states[:, cluster1[0]:cluster1[1], :, :] if cluster1 is not None else None
        value_states0 = value_states[:, cluster0[0]:cluster0[1], :, :]
        value_states1 = value_states[:, cluster1[0]:cluster1[1], :, :] if cluster1 is not None else None


    query_states0, key_states0 = apply_rotary_pos_emb(query_states0, key_states0, cos, sin, position_ids)
    query_states1, key_states1 = apply_rotary_pos_emb(query_states1, key_states1, cos, sin, position_ids) if cluster1 is not None else (None, None)

    if not divide_head:
        if cluster1 is None:
            query_states = query_states0
            key_states = key_states0
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
    updated_key_states, updates_value_states = past_key_value.update(
        [key_states0, key_states1], 
        [value_states0, value_states1],
        self.layer_idx,
        global_size,
        band_size,
        cache_kwargs,
    )

    if divide_head:
        attn_output0 = self.efficient_attention(
            query_states0, 
            updated_key_states[0], 
            updates_value_states[0], 
            self.head_dim**-0.5, 
            None, 
            self.block_size, 
            self.block_size,
            None,
            self.attention_dropout if self.training else 0.0
        )
        attn_output1 = self.efficient_attention(
            query_states1,
            updated_key_states[1],
            updates_value_states[1],
            self.head_dim**-0.5,
            None,
            self.block_size,
            self.block_size,
            None,
            self.attention_dropout if self.training else 0.0
        ) if cluster1 is not None else None

        # concat the output
        attn_output = torch.cat([attn_output0, attn_output1], dim=1) if cluster1 is not None else attn_output0

    else:
        attn_output = self.efficient_attention(
            # for prefill
            query_states, 
            key_states, 
            value_states, 
            self.head_dim**-0.5, 
            lut, 
            self.block_size, 
            self.block_size,
            # for decode
            attention_mask,
            self.attention_dropout if self.training else 0.0 # noqa
        )

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def apply_rotary_pos_emb_single(x, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)

    return x_embed


def LlamaDecoderLayer_set_static_attention_lut(self, lut_list, lut_for_head, block_size: int, device: Optional[str] = None, permute_head=False, sparse_decode=False):
    """
    Set the attention layout of the decoder layer

    lut: a tuple has 'layer' elements, each element has the size of [lut_num_heads, num_block, nnz]
    lut_for_head: a tuple has 'layer' elements, each element has the size of [lut_num_heads]
                  we use it as an indicator when combine heads
    block_size: int
    device: str
    """

    if isinstance(lut_list, List) is False:
        lut_list = [lut_list]

    # print(f"permute head is set to {permute_head}")

    # self.self_attn.efficient_attention = sparse_attention_prefill
    num_heads = lut_list[0].shape[0]
    self.self_attn.efficient_attention = sparse_attention
    if device is None:
        device = self.self_attn.o_proj.weight.device
        # print(device)
    lut_list = [lut.to(device) for lut in lut_list]
    lut_token_len = [lut.shape[1] * block_size for lut in lut_list]
       
    if permute_head:
        # get a single permutation and a single cluster from lut_list
        permutation, cluster = lut_to_permutation(lut_list, num_heads)
        # print("The number of different rules per layer is ", len(cluster))
        
        # assert cluster dictionary has only two records
        assert len(cluster) <= 2, f"to use permute head, you must have at most two patterns for the head, currently have {len(cluster)}"
        lut_list = [permute_lut(lut, permutation, num_heads) for lut in lut_list]
        self.self_attn.permutation = permutation
        self.self_attn.cluster = cluster

        lut0_list = [lut[cluster[0][0], :, :] for lut in lut_list]
        lut1_list = [lut[cluster[1][0], :, :] for lut in lut_list] if len(cluster) == 2 else None

        global0_list = [get_lut_global_size(lut0, block_size) for lut0 in lut0_list]
        global1_list = [get_lut_global_size(lut1, block_size) for lut1 in lut1_list] if len(cluster) == 2 else [None for _ in lut0_list]

        self.self_attn.global_size_dict = {i: (global0, global1) for i, global0, global1 in zip(lut_token_len, global0_list, global1_list)}

        band_list  = [get_lut_band_size(lut0, block_size) for lut0 in lut0_list]
        band1_list = [get_lut_band_size(lut1, block_size) for lut1 in lut1_list] if len(cluster) == 2 else [None for _ in lut0_list]

        self.self_attn.band_size_dict = {i: (band, band1) for i, band, band1 in zip(lut_token_len, band_list, band1_list)}

    lut_dict = {i: lut for i, lut in zip(lut_token_len, lut_list)}
    self.self_attn.lut_dict = lut_dict

    if not permute_head and sparse_decode:
        layout_list = [lut2layout_single_layer(lut.to('cpu')).to('cuda') for lut in lut_list]
        self.self_attn.layout_dict = {i: layout for i, layout in zip(lut_token_len, layout_list)}
        print("successfully convert lut to layout")

    self.self_attn.lut_for_head = lut_for_head
    self.self_attn.block_size = block_size

    if permute_head:
        permute_attention_projection(self.self_attn.q_proj, permutation, num_heads)
        permute_attention_projection(self.self_attn.k_proj, permutation, num_heads)
        permute_attention_projection(self.self_attn.v_proj, permutation, num_heads)
        permute_output_projection(self.self_attn.o_proj, permutation, num_heads)


def LlamaModel_use_block_sparse_attention_lut(self, permute_head=False, sparse_decode=False):
    """
    Overall interface
    Set the model instance to use efficient attention instead of llama attention
    """

    # update functions in LlamaModel
    if permute_head:
        self.set_mixture_of_attention = MethodType(LlamaModel_set_mixture_of_attention, self)
        self.forward = MethodType(LlamaModel_MixtureAttention_forward, self)

    # update functions in LlamaAttention
    for layer in self.layers:
        if permute_head:
            layer.self_attn.forward = MethodType(LlamaAttention_block_sparse_lut_forward_split_head, layer.self_attn)
        else:
            layer.self_attn.forward = MethodType(LlamaAttention_block_sparse_lut_forward_fake_sparse_decode, layer.self_attn)
            layer.self_attn.sparse_decode = sparse_decode

        layer.set_static_attention_lut = MethodType(LlamaDecoderLayer_set_static_attention_lut, layer)

"""
streamingLLM implementation
"""

def LlamaModel_Streamingllm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0
    if use_cache:
        # use_legacy_cache = not isinstance(past_key_values, Cache)
        # if use_legacy_cache:
        if past_key_values is None:
            past_key_values = StreamingllmDynamicCache()
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    elif self._use_sdpa and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    ## changes ###
    if use_cache:
        next_cache = next_decoder_cache
    ### end ###
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    ### !you can just past the last hidden states in generation mode
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states[:, -1:, :],
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def LlamaAttention_streamingllm_forward(
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

        is_decode = (q_len == 1)
        assert bsz == 1, "only support batch size 1 now"

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

        # ## the key inside the kv cache does not contain position embedding here
        # if past_key_value is not None:
        #     cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        band_size = self.band_size
        global_size = self.global_size

        if is_decode:
            ## the key inside the kv cache does not contain position embedding here
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs, band_size=band_size, global_size=global_size)
            
            kv_seq_len = key_states.shape[2]
            
            # create position_ids again, using kv_seq_len, possibly with batch_size
            position_ids = torch.tensor(kv_seq_len-1, dtype=torch.long, device=position_ids.device).unsqueeze(0).unsqueeze(0)

            ### add position embedding ###
            key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)

            query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
            ### end of position embedding ###

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            # attention_mask = None

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
        
        else:
            # the key inside the kv cache does not contain position embedding here
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            if key_states.shape[2] > band_size + global_size:
                concatenated_key_states = torch.cat([key_states[:, :, :global_size, :], key_states[:, :, -band_size:, :]], dim=2)
                concatenated_value_states = torch.cat([value_states[:, :, :global_size, :], value_states[:, :, -band_size:, :]], dim=2)
                past_key_value.update(concatenated_key_states, concatenated_value_states, self.layer_idx, cache_kwargs, update_cache_position=key_states.shape[2])
            else:
                past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)



            # save a copy of original states
            original_query_states = query_states.clone().detach()
            original_key_states = key_states.clone().detach()

            # first do normal attention, which calculates the global_size area
            query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            # we will add causal mask after the attention weights are calculated
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            # assert attention_mask is not None, "causal mask is required for streaming attention"

            if band_size + global_size < kv_seq_len:
                # further masking
                attn_weights  = attn_weights + self.global_mask[:, :, :kv_seq_len, :kv_seq_len]

                # rope for key
                new_key_states = original_key_states[:, :, :global_size, :]
                new_query_states = original_query_states[:, :, band_size + global_size:, :]

                new_key_position_ids = torch.arange(global_size, device=position_ids.device).unsqueeze(0)
                new_key_states = apply_rotary_pos_emb_single(new_key_states, cos, sin, new_key_position_ids)

                # rope for query
                new_query_position_ids = torch.full((1, kv_seq_len - band_size - global_size), band_size + global_size - 1, device=position_ids.device)
                new_query_states = apply_rotary_pos_emb_single(new_query_states, cos, sin, new_query_position_ids)

                # repeat k/v heads if n_kv_heads < n_heads
                new_key_states = repeat_kv(new_key_states, self.num_key_value_groups)
                new_query_states = repeat_kv(new_query_states, self.num_key_value_groups)

                new_attn_weights = torch.matmul(new_query_states, new_key_states.transpose(2, 3)) / math.sqrt(
                    self.head_dim
                )

                # copy new_attn_weights to attn_weights
                attn_weights[:, :, band_size + global_size:, :global_size] = new_attn_weights[:, :, :, :]


            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            # upcast attention to fp32
            if kv_seq_len < 12000:
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            else:
                attn_weights[:, :self.num_heads // 2, :, :] = nn.functional.softmax(attn_weights[:, :self.num_heads // 2, :, :], dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights[:, self.num_heads // 2:, :, :] = nn.functional.softmax(attn_weights[:, self.num_heads // 2:, :, :], dim=-1, dtype=torch.float32).to(query_states.dtype)

            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def create_streaming_attention_mask(token_len, global_size, band_size):
    # Start by creating an empty mask filled with False (0)
    mask = torch.zeros(1, 1, token_len, token_len, dtype=torch.bool)
    
    # Apply causal mask with band pattern
    for i in range(token_len):
        # Ensuring causality and the band pattern
        start = max(i - band_size + 1, 0)
        end = min(i + 1, token_len)  # Causal limit
        mask[:, :, i, start:end] = True
    
    # Overwrite the first `global_size` columns to True, only up to the diagonal for causality
    if global_size > 0:
        for i in range(token_len):
            mask[:, :, i, :min(global_size, i+1)] = True
        
    return mask


def LlamaModel_use_streamingllm_attention(model, global_size, band_size, device='cuda', max_length=16384):
    """
    Set the model instance to use streamingllm like attention instead of llama attention
    """

    # create a mask in advance
    attention_mask = create_streaming_attention_mask(max_length, global_size, band_size)
    attention_mask = (~attention_mask).to(torch.float16) * torch.finfo(torch.float16).min

    # get the device name of all layers. they may be on different cuda devices
    device_list = set([str(layer.self_attn.o_proj.weight.device) for layer in model.layers])
    attention_mask_dict = {device: attention_mask.to(device) for device in device_list}
    print(f"parameters are on devices: {device_list}")

    for layer in model.layers:
        layer.self_attn.forward = MethodType(LlamaAttention_streamingllm_forward, layer.self_attn)
        layer.self_attn.global_size = global_size
        layer.self_attn.band_size = band_size
        layer.self_attn.global_mask = attention_mask_dict[str(layer.self_attn.o_proj.weight.device)]

    attention_density = streamingllm_attention_density(global_size, band_size, max_length)
    kv_cache_deisnty = streamingllm_kv_cache_density(global_size, band_size, max_length)

    print(f"streamingllm: global_size = {global_size}, band_size = {band_size}, max_length = {max_length}\n attention_density: {attention_density}\n kv_cache_density: {kv_cache_deisnty}")

    model.forward = MethodType(LlamaModel_Streamingllm_forward, model)
