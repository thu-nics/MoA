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