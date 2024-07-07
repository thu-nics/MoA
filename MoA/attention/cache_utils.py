from transformers.cache_utils import Cache
from transformers.cache_utils import DynamicCache as OriginalCache
from typing import Any, Dict, List, Optional, Tuple
import torch

class NewDynamicCache(Cache):
    def __init__(
        self,
        pattern_num: List[int],
        global_size: List[List[int]],
        band_size: List[List[int]],
        pattern_index: List[List[int]],
    ) -> None:
        self.seen_tokens = 0
        self.layer_cache = []
        for i in range(len(pattern_num)):
            self.layer_cache.append(LayerCache(pattern_num[i], global_size[i], band_size[i], pattern_index[i]))

    def __len__(self):
        return len(self.layer_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]

        # Update the cache
        key_cache, value_cache = self.layer_cache[layer_idx].update(key_states, value_states)

        return key_cache, value_cache
    
    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        raise NotImplementedError

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            raise NotImplementedError("Reordering the cache is not implemented currently!")
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))


class StreamingllmDynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        update_seen_tokens: Optional[int] = None,
        global_size: Optional[int] = None,
        band_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            if update_seen_tokens is not None:
                self.seen_tokens += update_seen_tokens
            else:
                self.seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            if band_size is not None and global_size is not None and (band_size + global_size < self.key_cache[layer_idx].shape[2]):
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx][:, :, :global_size, :], self.key_cache[layer_idx][:, :, -band_size:, :]], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx][:, :, :global_size, :], self.value_cache[layer_idx][:, :, -band_size:, :]], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))


class LayerCache(Cache):
    def __init__(
        self,
        pattern_num: int,
        global_size: List[int],
        band_size: List[int],
        pattern_index: List[int],
    ) -> None:
        self.pattern_num = pattern_num
        self.global_size = global_size
        self.band_size = band_size
        self.pattern_index = pattern_index
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self.replace_index = [self.global_size[i] for i in range(self.pattern_num)]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        '''
        key_states, value_states : [bz, num_heads, seq_len, head_dim]
        currently, after the first insert, the size of key_cache and value_cache will be fixed
        '''
        if len(self.key_cache) == 0:
            # only keep the first self.global_size and the last self.band_size of seq_len dimension
            # if key_states.size(2) < self.global_size + self.band_size:
            #     raise ValueError(f"seq_len of key_states should be at least {self.global_size + self.band_size}")
            seperated_key_states = self.seperate_states(key_states)
            seperated_value_states = self.seperate_states(value_states)


            for i in range(self.pattern_num):
                self.key_cache.append(torch.cat([seperated_key_states[i][:, :, :self.global_size[i], :], seperated_key_states[i][:, :, -self.band_size[i]:, :]], dim=2))
                self.value_cache.append(torch.cat([seperated_value_states[i][:, :, :self.global_size[i], :], seperated_value_states[i][:, :, -self.band_size[i]:, :]], dim=2))

                # whe the first insert, return the original key_states and value_states for prefilling
                return key_states, value_states

        else:
            assert key_states.size(2) == 1, "in decoding mode, key_states should only have one token"
            seperated_key_states = self.seperate_states(key_states)
            seperated_value_states = self.seperate_states(value_states)
            # print(f"replace_index: {self.replace_index}")
            for i in range(self.pattern_num):
                replace_index = self.replace_index[i]
                self.key_cache[i][:, :, replace_index:replace_index+1, :] = seperated_key_states[i]
                self.value_cache[i][:, :, replace_index:replace_index+1, :] = seperated_value_states[i]
                
                if replace_index == self.global_size[i] + self.band_size[i] - 1:
                    self.replace_index[i] = self.global_size[i]
                else:
                    self.replace_index[i] += 1
            
            return self.key_cache, self.value_cache

    def seperate_states(self, states: torch.Tensor) -> List[torch.Tensor]:
        '''
        states: [bz, num_heads, seq_len, head_dim]
        '''
        result = []

        for i in range(self.pattern_num):
            result.append(states[:, self.pattern_index[i]:self.pattern_index[i+1], :, :])

        return result


class CircularCacheSingle(Cache):
    def __init__(
        self,
    ) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen\
        self.global_size = []
        self.band_size = []
        self.replace_index = []
        self.kv_len = []

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")
        
    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)
    

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        global_size: Optional[int] = None,
        band_size: Optional[int] = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `CircularSingleCache`.
            global_size (`int`, `optional`):
                The size of the global part of the this layer head pattern.
            band_size (`int`, `optional`):
                The size of the band part of the this layer head pattern.

        Return:
            A tuple containing the updated key and value states.
        """

        # # Update the number of seen tokens
        # if layer_idx == 0 and key_states is not None:
        #     self.seen_tokens += key_states.shape[-2]

        # # Update the cache
        # if len(self.key_cache) <= layer_idx:
            
        #     if key_states is None:
        #         self.key_cache.append(None)
        #         self.value_cache.append(None)
        #         self.global_size.append(None)
        #         self.band_size.append(None)
        #         self.replace_index.append(None)
        #         return None, None
            
        #     assert global_size is not None and band_size is not None, "In the first update, global_size and band_size must be provided"

        #     self.global_size.append(global_size)
        #     self.band_size.append(band_size)

        #     assert len(self.key_cache) == layer_idx
        #     seq_len = key_states.shape[2]
        #     if seq_len >= global_size + band_size:
        #         self.key_cache.append(torch.cat([key_states[:, :, :global_size, :], key_states[:, :, -band_size:, :]], dim=2))
        #         self.value_cache.append(torch.cat([value_states[:, :, :global_size, :], value_states[:, :, -band_size:, :]], dim=2))
        #     else:
        #         # when the seq len is too short, just use the original key_states and value_states'
        #         # update global size
        #         # ! this shortens the band_size
        #         self.key_cache.append(key_states)
        #         self.value_cache.append(value_states)

        #         self.global_size[layer_idx] = seq_len - band_size

        #     if seq_len < global_size:
        #         raise ValueError(f"seq_len of key_states should be at least {global_size}")

        #     self.replace_index.append(global_size)
        # else:

        #     if key_states is None:
        #         return None, None
            
        #     assert key_states.size(2) == 1, "in decoding mode, key_states should only have one token"

        #     replace_index = self.replace_index[layer_idx]
        #     self.key_cache[layer_idx][:, :, replace_index:replace_index+1, :] = key_states
        #     self.value_cache[layer_idx][:, :, replace_index:replace_index+1, :] = value_states

        #     if replace_index == self.global_size[layer_idx] + self.band_size[layer_idx] - 1:
        #         self.replace_index[layer_idx] = self.global_size[layer_idx]
        #     else:
        #         self.replace_index[layer_idx] += 1

        # return self.key_cache[layer_idx], self.value_cache[layer_idx]

        ### new implementation ###


        # Update the number of seen tokens
        if layer_idx == 0 and key_states is not None:
            self.seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            # initialize the cache
            
            if key_states is None:
                self.key_cache.append(None)
                self.value_cache.append(None)
                self.global_size.append(None)
                self.band_size.append(None)
                self.replace_index.append(None)
                self.kv_len.append(None)
                return None, None
            
            assert global_size is not None and band_size is not None, "In the first update, global_size and band_size must be provided"

            self.global_size.append(global_size)
            self.band_size.append(band_size)

            assert len(self.key_cache) == layer_idx
            seq_len = key_states.shape[2]
            if seq_len >= global_size + band_size:

                self.key_cache.append(torch.cat([key_states[:, :, :global_size, :], key_states[:, :, -band_size:, :]], dim=2))
                self.value_cache.append(torch.cat([value_states[:, :, :global_size, :], value_states[:, :, -band_size:, :]], dim=2))

                self.kv_len.append(global_size + band_size)
                self.replace_index.append(global_size)

            else:
                # preallocate global_size + band_size space
                longer_allocated_key = torch.empty(key_states.shape[0], key_states.shape[1], global_size + band_size, key_states.shape[3], device=key_states.device, dtype=key_states.dtype)
                longer_allocated_value = torch.empty(value_states.shape[0], value_states.shape[1], global_size + band_size, value_states.shape[3], device=value_states.device, dtype=value_states.dtype)
                longer_allocated_key[:, :, :seq_len, :] = key_states
                longer_allocated_value[:, :, :seq_len, :] = value_states
                self.key_cache.append(longer_allocated_key)
                self.value_cache.append(longer_allocated_value)

                self.kv_len.append(seq_len)
                self.replace_index.append(seq_len)

            if seq_len < global_size:
                raise ValueError(f"seq_len of key_states should be at least {global_size}")

        else:

            if key_states is None:
                return None, None
            
            assert key_states.size(2) == 1, "in decoding mode, key_states should only have one token"

            replace_index = self.replace_index[layer_idx]
            self.key_cache[layer_idx][:, :, replace_index:replace_index+1, :] = key_states
            self.value_cache[layer_idx][:, :, replace_index:replace_index+1, :] = value_states

            if replace_index == self.global_size[layer_idx] + self.band_size[layer_idx] - 1:
                self.replace_index[layer_idx] = self.global_size[layer_idx]
            else:
                self.replace_index[layer_idx] += 1
            
            if self.kv_len[layer_idx] < self.global_size[layer_idx] + self.band_size[layer_idx]:
                self.kv_len[layer_idx] += 1

        if self.kv_len[layer_idx] == self.global_size[layer_idx] + self.band_size[layer_idx]:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            return self.key_cache[layer_idx][:, :, :self.kv_len[layer_idx], :], self.value_cache[layer_idx][:, :, :self.kv_len[layer_idx], :]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # if len(self.key_cache) <= layer_idx:
        #     return 0
        # return self.key_cache[layer_idx].shape[-2]

        ### new implementation ###
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.kv_len[layer_idx]
    
    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        raise NotImplementedError("Reordering the cache is not implemented currently!")

class CircularCache(Cache):
    def __init__(self, pattern_num: int = 1, num_layers = 32) -> None:
        self.cache = []
        for _ in range(pattern_num):
            self.cache.append(CircularCacheSingle())
        
        self.num_layers = num_layers

        self.seen_tokens = 0
        self.current_tokens = 0

    def __len__(self):

        return len(self.cache[0])
    
    def update(
        self,
        key_states: List[torch.Tensor],
        value_states: List[torch.Tensor],
        layer_idx: int,
        global_size: Optional[List[int]] = None,
        band_size: Optional[List[int]] = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        assert layer_idx < self.num_layers

        if layer_idx == 0:
            self.seen_tokens += key_states[0].shape[-2]
        if layer_idx == self.num_layers - 1:
            self.current_tokens = self.seen_tokens

        if global_size is None:
            global_size = [None for _ in range(len(key_states))]
        if band_size is None:
            band_size = [None for _ in range(len(key_states))]

        assert len(key_states) == len(self.cache)
        assert len(value_states) == len(self.cache)
        assert len(global_size) == len(self.cache)
        assert len(band_size) == len(self.cache)

        updated_key_states = []
        updated_value_states = []

        for i in range(len(key_states)):
            key_state, value_state = self.cache[i].update(key_states[i], value_states[i], layer_idx, global_size[i], band_size[i], cache_kwargs)
            updated_key_states.append(key_state)
            updated_value_states.append(value_state)

        return updated_key_states, updated_value_states

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # ! warning: this function is meaningless, just a place holder
        return self.cache[0].get_seq_length(layer_idx)
        # raise NotImplementedError
    
    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None
    
    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        raise NotImplementedError("Reordering the cache is not implemented currently!")

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        # return self.get_seq_length(layer_idx)
        raise NotImplementedError        

class StaticCircularCache(Cache):
    """
    A cache that do not grow dynamically as more tokens are generated. This is the default for mixture-of-sparse-attention (MoA) method.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, \sum_h^H cache_size_of_head_h , head_dim]`. Note that when using the `update` function, the shape of each layer is `[batch_size, num_heads, seq_len, head_dim]`.
    Each head has its own total cache size consisting two parts: static and circular. The static part is the first `global_size` tokens and the circular part is the last `band_size` tokens. When updating the cache, the circular part will be replaced by the new tokens, while the static part will be kept unchanged.
    """
    def __init__(
            self, 
            cache_size: List[List[int]],
            batch_size: int = 1,
            head_dim: int = 64,
            static_size: Optional[List[List[int]]] = None,
            dtype=None,
        ) -> None:
        """
        Parameters:
            cache_size (`List[List[int]]`):
                The total cache size for each head in each layer. The cache size is the sum of the global part and the band part.
            static_size (*optional*, `List[List[int]]`):
                The size of the static part for each head in each layer. If not provided, the static part will be set to 0.
            dtype (*optional*, `torch.dtype`):
                The data type of the cache. If not provided, it will be set to `torch.float32`.
            batch_size (*optional*, `int`):
                The batch size of the input data. If not provided, it will be set to 1.
            head_dim (*optional*, `int`):
                The dimension of the head. If not provided, it will be set to 64.
        """
        self.dtype = dtype if dtype is not None else torch.float32

        self.num_layers = len(cache_size)
        self.num_head_for_each_layer = [len(cache_size[layer_id]) for layer_id in range(self.num_layers)]

        if static_size is None:
            static_size = [[0 for _ in range(len(cache_size[layer_id]))] for layer_id in range(self.num_layers)]

        self.cache_size: List[torch.Tensor] = [torch.tensor(cache_size_this_layer) for cache_size_this_layer in cache_size]
        self.static_cache_size: List[torch.Tensor] = [torch.tensor(static_size_this_layer) for static_size_this_layer in static_size]
        self.circular_cache_size: List[torch.Tensor] = [cache_size_this_layer - static_size_this_layer for cache_size_this_layer, static_size_this_layer in zip(self.cache_size, self.static_cache_size)]

        # static size should be the same shape as cache size
        assert len(static_size) == self.num_layers
        for layer_id in range(self.num_layers):
            assert len(static_size[layer_id]) == len(cache_size[layer_id])

        self.total_cache_size: List[int] = [sum(this_cache_size) for this_cache_size in cache_size] # total cache size for each layer
        head_start_index = [[] for layer_id in range(self.num_layers)] # the starting index of each head in each layer
        head_end_index = [[] for layer_id in range(self.num_layers)] # the ending index of each head in each layer
        for layer_id in range(self.num_layers):
            for head_id in range(len(cache_size[layer_id])):
                if head_id == 0:
                    head_start_index[layer_id].append(0)
                else:
                    head_start_index[layer_id].append(head_start_index[layer_id][-1] + cache_size[layer_id][head_id-1])
                head_end_index[layer_id].append(head_start_index[layer_id][head_id] + cache_size[layer_id][head_id])

        self.cache_head_start_index = [torch.tensor(head_start_index[layer_id], dtype=int) for layer_id in range(self.num_layers)]    
        self.cache_head_end_index = [torch.tensor(head_end_index[layer_id], dtype=int) for layer_id in range(self.num_layers)]

        self.circular_cache_head_index: List[torch.tensor] = [torch.tensor([head_start_index[layer_id][head_id]+static_size[layer_id][head_id] for head_id in range(self.num_head_for_each_layer[layer_id])]) for layer_id in range(self.num_layers)] # the starting index of the circular part of each head in each layer
        
        # self._update_index: List[torch.Tensor] = [torch.tensor([head_start_index[layer_id][head_id]+static_size[layer_id][head_id] for head_id in range(self.num_head_for_each_layer[layer_id])]) for layer_id in range(self.num_layers)] # initialize the update index to the beginning of the circular part
        self._update_index: List[torch.Tensor] = [torch.tensor(head_start_index[layer_id], dtype=int) for layer_id in range(self.num_layers)] # initialize the update index to the beginning of the cache, it will later circulate within the circular part after filling the cache

        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self._kv_len = [0 for _ in range(self.num_layers)] # the length of the key and value cache for each layer

        self.key_cache: List[torch.Tensor] = [torch.zeros(batch_size, total_cache_size_this, head_dim, dtype=self.dtype) for total_cache_size_this in self.total_cache_size]
        self.value_cache: List[torch.Tensor] = [torch.zeros(batch_size, total_cache_size_this, head_dim, dtype=self.dtype) for total_cache_size_this in self.total_cache_size]

        self._current_tokens = 0 # 

    @staticmethod
    def _cache_uncontigious(
        tensor: torch.Tensor, 
        head_start_index: torch.Tensor, 
        cache_size: torch.Tensor, 
    ) -> List[torch.Tensor]:
        """
        Split the tensor to each head according to the head_start_index, cache_size.

        Parameters:
            tensor (`torch.Tensor`):
                The expected shape for each tensor is `[batch_size, \sum_h^H cache_size_of_head_h , head_dim]`.
            head_start_index (`torch.Tensor`):
                The starting index of each head.
            cache_size (`torch.Tensor`):
                The total cache size for each head.
        
        Return:
            A list of tensors, each tensor is the cache for each head of shape `[batch_size, cache_size_of_head_h, head_dim]`.
        """
        return [tensor[:, head_start_index[head_id]:head_start_index[head_id]+cache_size[head_id], :] for head_id in range(len(cache_size))]
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        # position_ids: torch.Tensor, # can be used to move the sink part to the left side
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
            key_states (`torch.Tensor`):
                The expected shapes for each key_states or value_states are
                    `[batch_size, num_heads, seq_len, head_dim]`.

        Return:
            A tuple containing the updated key and value states. The expected shapes for each key_states or value_states are `[batch_size, \sum_h^H cache_size_of_head_h , head_dim]`.
        """
        assert layer_idx < self.num_layers
        assert key_states.shape[1] == self.num_head_for_each_layer[layer_idx]

        # TODO: move the sink part to the left side
        # padding - sink - circular 

        # Update the number of seen tokens
        batch_size = key_states.shape[0]
        seq_len = key_states[0].shape[-2]
        head_dim = key_states[0].shape[-1]
        if layer_idx == 0:
            self._seen_tokens += seq_len

        # calculate the update index of the cache
        update_index_circular = ((torch.stack([self._update_index[layer_idx] + i for i in range(seq_len+1)], dim=0) - self.circular_cache_head_index[layer_idx]) % self.circular_cache_size[layer_idx]) + self.circular_cache_head_index[layer_idx] # shape: (seq_len+1, num_heads), ending at _kv_len > static_len
        update_index_static = torch.stack([self._update_index[layer_idx] + i for i in range(seq_len+1)], dim=0) # shape: (seq_len+1, num_heads), ending at _kv_len <= static_len
        update_index = torch.where(self._kv_len[layer_idx] + torch.arange(0, seq_len+1).reshape(-1,1).expand(-1, num_head) > self.static_cache_size[layer_idx], update_index_circular, update_index_static) # shape: (seq_len+1, num_heads)

        self._kv_len[layer_idx] += seq_len

        self._update_index[layer_idx] = update_index[-1, :] # the starting point for next update
        update_index = update_index[:seq_len, :].permute(1, 0) # shape: (num_heads, seq_len)

        # If the seq_len is so long that it > cache size, will cause scatter error (ideally, only the latter indexed key/value should be reserved)
        # keep the index where 1. the seq_len is less than the cache size or 2. index within static cache
        valid_index_map = (torch.arange(seq_len, 0, -1, device=key_states.device).unsqueeze(0).expand(num_head, -1) <= self.circular_cache_size[layer_idx].reshape(-1,1)) | (update_index < self.circular_cache_head_index[layer_idx].reshape(-1, 1)) # shape (num_heads, seq_len)
    
        valid_update_index = update_index[valid_index_map] # shape: (num_heads * seq_len)
        
        valid_key_states = key_states.reshape(batch_size, -1, head_dim)[:, valid_index_map.reshape(-1), :] # shape (batch_size, num_heads * seq_len, head_dim)
        valid_value_states = value_states.reshape(batch_size, -1, head_dim)[:, valid_index_map.reshape(-1), :] # shape (batch_size, num_heads * seq_len, head_dim)
        
        valid_update_index = valid_update_index.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, head_dim) # shape: [batch_size, num_heads * seq_len, head_dim]

        # assign the new key_states and value_states to the cache by the update_index
        self.key_cache[layer_idx] = torch.scatter(self.key_cache[layer_idx], -2, valid_update_index, valid_key_states)
        self.value_cache[layer_idx] = torch.scatter(self.value_cache[layer_idx], -2, valid_update_index, valid_value_states)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def __len__(self):
        return len(self.key_cache)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # ! warning: this function is meaningless, just a place holder
        return self._kv_len[layer_idx]
    
    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. This Cache does not have a maximum length."""
        return None
    
    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        raise NotImplementedError("Reordering the cache is not implemented currently!")


def moa_config_to_cache_config(
        moa_config,
        seq_len,
        max_new_token: int = 1024,
        sink_size: int = 64,
        minimum_cache_size: int = 128,
    ):
    """
    Convert the MoA configuration to the cache configuration
    
    Parameters:
        moa_config (`Dict`):
            The MoA configuration.
    
    Return:
        A dictionary containing the cache configuration.
    """
    cache_size_dict = []
    static_size_dict = []

    alphas = moa_config["alphas"]
    betas = moa_config["betas"]
    
    for layer_id in range(len(alphas)):
        cache_size_this_layer = []
        static_size_this_layer = []
        for head_id in range(len(alphas[layer_id])):
            cache_size_this_head = int(alphas[layer_id][head_id] + (seq_len+max_new_token) * betas[layer_id][head_id])
            cache_size_this_head = max(cache_size_this_head, minimum_cache_size)
            cache_size_this_layer.append(cache_size_this_head)
            static_size_this_layer.append(min(cache_size_this_head, sink_size))
        cache_size_dict.append(cache_size_this_layer)
        static_size_dict.append(static_size_this_layer)

    return {
        "cache_size": cache_size_dict,
        "static_size": static_size_dict,
    }

if __name__ == "__main__":
    batch_size = 2
    head_dim = 4
    num_head = 2

    cache = StaticCircularCache(
        cache_size=[[5, 10], [5, 5]],
        static_size=[[2, 3], [2, 3]],
        batch_size=batch_size,
        head_dim=head_dim,
        dtype=torch.float16,
    )

    acc = 0
    for step in range(0, 10):
        seq_len = 7

        key_states = torch.arange(acc, batch_size * num_head * seq_len * head_dim + acc, dtype=torch.float16).reshape(batch_size, num_head, seq_len, head_dim)
        value_states = torch.arange(acc, batch_size * num_head * seq_len * head_dim + acc, dtype=torch.float16).reshape(batch_size, num_head, seq_len, head_dim)

        acc += batch_size * num_head * seq_len * head_dim

        key_cache_new, value_cache_new = cache.update(key_states, value_states, 0)

        list_key_cache_new = cache._cache_uncontigious(key_cache_new, cache.cache_head_start_index[0], cache.cache_size[0])

        # batch = 1
        print(key_states[1])
        print(list_key_cache_new[0][1]) # head 0, total size 5, static size 2
        print(list_key_cache_new[1][1]) # head 1, total size 10, static size 3

        print("done")


