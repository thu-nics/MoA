from transformers.cache_utils import Cache
from transformers.cache_utils import DynamicCache as OriginalCache
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import torch
import math
from torch import Tensor

class NewDynamicCache(Cache):
    def __init__(
        self,
        pattern_num: List[int],
        global_size: List[List[int]],
        band_size: List[List[int]],
        pattern_index: List[List[int]],
    ) -> None:
        self.cache_position = 0
        self.layer_cache = []
        for i in range(len(pattern_num)):
            self.layer_cache.append(
                LayerCache(
                    pattern_num[i], global_size[i], band_size[i], pattern_index[i]
                )
            )

    def __len__(self):
        return len(self.layer_cache)

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self.cache_position += key_states.shape[-2]

        # Update the cache
        key_cache, value_cache = self.layer_cache[layer_idx].update(
            key_states, value_states
        )

        return key_cache, value_cache

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        raise NotImplementedError

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            raise NotImplementedError(
                "Reordering the cache is not implemented currently!"
            )
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )


class StreamingllmDynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache: List[Tensor] = []
        self.value_cache: List[Tensor] = []
        self.cache_position = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )

    def __getitem__(self, layer_idx: int) -> List[Tuple[Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}"
            )

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
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        update_cache_position: Optional[int] = None,
        global_size: Optional[int] = None,
        band_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`Tensor`):
                The new key states to cache.
            value_states (`Tensor`):
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
            if update_cache_position is not None:
                self.cache_position += update_cache_position
            else:
                self.cache_position += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )
            if (
                band_size is not None
                and global_size is not None
                and (band_size + global_size < self.key_cache[layer_idx].shape[2])
            ):
                self.key_cache[layer_idx] = torch.cat(
                    [
                        self.key_cache[layer_idx][:, :, :global_size, :],
                        self.key_cache[layer_idx][:, :, -band_size:, :],
                    ],
                    dim=-2,
                )
                self.value_cache[layer_idx] = torch.cat(
                    [
                        self.value_cache[layer_idx][:, :, :global_size, :],
                        self.value_cache[layer_idx][:, :, -band_size:, :],
                    ],
                    dim=-2,
                )

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
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )


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
        self.key_cache: List[Tensor] = []
        self.value_cache: List[Tensor] = []

        self.replace_index = [self.global_size[i] for i in range(self.pattern_num)]

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        key_states, value_states : [bz, num_heads, seq_len, head_dim]
        currently, after the first insert, the size of key_cache and value_cache will be fixed
        """
        if len(self.key_cache) == 0:
            # only keep the first self.global_size and the last self.band_size of seq_len dimension
            # if key_states.size(2) < self.global_size + self.band_size:
            #     raise ValueError(f"seq_len of key_states should be at least {self.global_size + self.band_size}")
            seperated_key_states = self.seperate_states(key_states)
            seperated_value_states = self.seperate_states(value_states)

            for i in range(self.pattern_num):
                self.key_cache.append(
                    torch.cat(
                        [
                            seperated_key_states[i][:, :, : self.global_size[i], :],
                            seperated_key_states[i][:, :, -self.band_size[i] :, :],
                        ],
                        dim=2,
                    )
                )
                self.value_cache.append(
                    torch.cat(
                        [
                            seperated_value_states[i][:, :, : self.global_size[i], :],
                            seperated_value_states[i][:, :, -self.band_size[i] :, :],
                        ],
                        dim=2,
                    )
                )

                # whe the first insert, return the original key_states and value_states for prefilling
                return key_states, value_states

        else:
            assert (
                key_states.size(2) == 1
            ), "in decoding mode, key_states should only have one token"
            seperated_key_states = self.seperate_states(key_states)
            seperated_value_states = self.seperate_states(value_states)
            # print(f"replace_index: {self.replace_index}")
            for i in range(self.pattern_num):
                replace_index = self.replace_index[i]
                self.key_cache[i][:, :, replace_index : replace_index + 1, :] = (
                    seperated_key_states[i]
                )
                self.value_cache[i][:, :, replace_index : replace_index + 1, :] = (
                    seperated_value_states[i]
                )

                if replace_index == self.global_size[i] + self.band_size[i] - 1:
                    self.replace_index[i] = self.global_size[i]
                else:
                    self.replace_index[i] += 1

            return self.key_cache, self.value_cache

    def seperate_states(self, states: Tensor) -> List[Tensor]:
        """
        states: [bz, num_heads, seq_len, head_dim]
        """
        result = []

        for i in range(self.pattern_num):
            result.append(
                states[:, self.pattern_index[i] : self.pattern_index[i + 1], :, :]
            )

        return result


class CircularCacheSingle(Cache):
    def __init__(
        self,
    ) -> None:
        self.key_cache: List[Tensor] = []
        self.value_cache: List[Tensor] = []
        self.cache_position = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen\
        )
        self.global_size = []
        self.band_size = []
        self.replace_index = []
        self.kv_len = []

    def __getitem__(self, layer_idx: int) -> List[Tuple[Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}"
            )

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
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        global_size: Optional[int] = None,
        band_size: Optional[int] = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`Tensor`):
                The new key states to cache.
            value_states (`Tensor`):
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
            self.cache_position += key_states.shape[-2]

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

            assert (
                global_size is not None and band_size is not None
            ), "In the first update, global_size and band_size must be provided"

            self.global_size.append(global_size)
            self.band_size.append(band_size)

            assert len(self.key_cache) == layer_idx
            seq_len = key_states.shape[2]
            if seq_len >= global_size + band_size:

                self.key_cache.append(
                    torch.cat(
                        [
                            key_states[:, :, :global_size, :],
                            key_states[:, :, -band_size:, :],
                        ],
                        dim=2,
                    )
                )
                self.value_cache.append(
                    torch.cat(
                        [
                            value_states[:, :, :global_size, :],
                            value_states[:, :, -band_size:, :],
                        ],
                        dim=2,
                    )
                )

                self.kv_len.append(global_size + band_size)
                self.replace_index.append(global_size)

            else:
                # preallocate global_size + band_size space
                longer_allocated_key = torch.empty(
                    key_states.shape[0],
                    key_states.shape[1],
                    global_size + band_size,
                    key_states.shape[3],
                    device=key_states.device,
                    dtype=key_states.dtype,
                )
                longer_allocated_value = torch.empty(
                    value_states.shape[0],
                    value_states.shape[1],
                    global_size + band_size,
                    value_states.shape[3],
                    device=value_states.device,
                    dtype=value_states.dtype,
                )
                longer_allocated_key[:, :, :seq_len, :] = key_states
                longer_allocated_value[:, :, :seq_len, :] = value_states
                self.key_cache.append(longer_allocated_key)
                self.value_cache.append(longer_allocated_value)

                self.kv_len.append(seq_len)
                self.replace_index.append(seq_len)

            if seq_len < global_size:
                raise ValueError(
                    f"seq_len of key_states should be at least {global_size}"
                )

        else:

            if key_states is None:
                return None, None

            assert (
                key_states.size(2) == 1
            ), "in decoding mode, key_states should only have one token"

            replace_index = self.replace_index[layer_idx]
            self.key_cache[layer_idx][
                :, :, replace_index : replace_index + 1, :
            ] = key_states
            self.value_cache[layer_idx][
                :, :, replace_index : replace_index + 1, :
            ] = value_states

            if (
                replace_index
                == self.global_size[layer_idx] + self.band_size[layer_idx] - 1
            ):
                self.replace_index[layer_idx] = self.global_size[layer_idx]
            else:
                self.replace_index[layer_idx] += 1

            if (
                self.kv_len[layer_idx]
                < self.global_size[layer_idx] + self.band_size[layer_idx]
            ):
                self.kv_len[layer_idx] += 1

        if (
            self.kv_len[layer_idx]
            == self.global_size[layer_idx] + self.band_size[layer_idx]
        ):
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            return (
                self.key_cache[layer_idx][:, :, : self.kv_len[layer_idx], :],
                self.value_cache[layer_idx][:, :, : self.kv_len[layer_idx], :],
            )

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
    def __init__(self, pattern_num: int = 1, num_layers=32) -> None:
        self.cache = []
        for _ in range(pattern_num):
            self.cache.append(CircularCacheSingle())

        self.num_layers = num_layers

        # self.seen_tokens = 0
        self.cache_position = 0
        self.current_tokens = 0

    def __len__(self):

        return len(self.cache[0])

    def update(
        self,
        key_states: List[Tensor],
        value_states: List[Tensor],
        layer_idx: int,
        global_size: Optional[List[int]] = None,
        band_size: Optional[List[int]] = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        assert layer_idx < self.num_layers

        if layer_idx == 0:
            self.cache_position += key_states[0].shape[-2]
        if layer_idx == self.num_layers - 1:
            self.current_tokens = self.cache_position

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
            key_state, value_state = self.cache[i].update(
                key_states[i],
                value_states[i],
                layer_idx,
                global_size[i],
                band_size[i],
                cache_kwargs,
            )
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

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
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
        head_dim: int = 128,
        static_size: Optional[List[List[int]]] = None,
        device: torch.device = "cpu",
        dtype: torch.dtype = None,
        update_cache_content: bool = True,
    ) -> None:
        """
        Parameters:
            cache_size (`List[List[int]]`):
                The total cache size for each head in each layer. The cache size is the sum of the global part and the band part.
            batch_size (*optional*, `int`):
                The batch size of the input data. If not provided, it will be set to 1.
            head_dim (*optional*, `int`):
                The dimension of the head. If not provided, it will be set to 64.
            static_size (*optional*, `List[List[int]]`):
                The size of the static part for each head in each layer. If not provided, the static part will be set to 0.
            device (*optional*, `torch.device`):
                The device of the cache. If not provided, it will be set to `cpu`.
            dtype (*optional*, `torch.dtype`):
                The data type of the cache. If not provided, it will be set to `torch.float16`.
            update_cache_content (*optional*, `bool`):
                Whether to update the cache content when calling `.update`. If not provided, it will be set to `True`. Set to False if the kernel update the content
        """
        super().__init__()
        self.dtype = dtype if dtype is not None else torch.float16
        self.device = device
        self.update_cache_content = update_cache_content
        self.batch_size = batch_size
        self.head_dim = head_dim

        # initialize the cache sizes and indices
        self.num_layers = len(cache_size)
        self.num_head_for_each_layer = [
            len(cache_size[layer_id]) for layer_id in range(self.num_layers)
        ]
        if static_size is None:
            static_size = [
                [0 for _ in range(len(cache_size[layer_id]))]
                for layer_id in range(self.num_layers)
            ]

        self.cache_max_length: List[torch.LongTensor] = [
            torch.tensor(cache_size_this_layer, device=device, dtype=torch.int64)
            for cache_size_this_layer in cache_size
        ]
        self.static_cache_size: List[torch.LongTensor] = [
            torch.tensor(static_size_this_layer, device=device, dtype=torch.int64)
            for static_size_this_layer in static_size
        ]
        self.circular_cache_size: List[torch.LongTensor] = [
            torch.tensor(
                cache_size_this_layer - static_size_this_layer,
                device=device,
                dtype=torch.int64,
            )
            for cache_size_this_layer, static_size_this_layer in zip(
                self.cache_max_length, self.static_cache_size
            )
        ]

        self.layer_cache_size: List[int] = [
            sum(this_cache_size) for this_cache_size in cache_size
        ]  # total cache size for each layer
        head_start_index = [
            [] for layer_id in range(self.num_layers)
        ]  # the starting index of each head in each layer

        for layer_id in range(self.num_layers):
            for head_id in range(len(cache_size[layer_id])):
                if head_id == 0:
                    head_start_index[layer_id].append(0)
                else:
                    head_start_index[layer_id].append(
                        head_start_index[layer_id][-1]
                        + cache_size[layer_id][head_id - 1]
                    )
        self.cache_head_start_index = [
            torch.tensor(
                head_start_index[layer_id], device=self.device, dtype=torch.int64
            )[None, :]
            .expand(batch_size, self.num_head_for_each_layer[layer_id])
            .contiguous()
            for layer_id in range(self.num_layers)
        ]  # shape: (batch_size, num_heads) * num_layers, the starting index of each head in each layer

        for layer_id in range(self.num_layers):
            # add another start index to indicate the end of the last head
            head_start_index[layer_id].append(
                head_start_index[layer_id][-1] + cache_size[layer_id][-1]
            )
        self.head_index = [
            torch.tensor(head_start_index[layer_id], dtype=torch.int64, device=device)
            for layer_id in range(self.num_layers)
        ]  # shape: (num_heads+1) * num_layers, the starting index of each head in each layer; contains the additional index to show the end of the cache

        self.circular_cache_head_index: List[torch.LongTensor] = [
            torch.tensor(
                [
                    head_start_index[layer_id][head_id] + static_size[layer_id][head_id]
                    for head_id in range(self.num_head_for_each_layer[layer_id])
                ],
                device=device,
                dtype=torch.int64,
            )
            for layer_id in range(self.num_layers)
        ]  # the starting index of the circular part of each head in each layer

        self.cache_update_index: List[torch.LongTensor] = [
            torch.tensor(
                head_start_index[layer_id][:-1], dtype=torch.int64, device=device
            ).expand(batch_size, -1)
            for layer_id in range(self.num_layers)
        ]  # initialize the update index to the beginning of the cache, it will later circulate within the circular part after filling the cache, shape (batch_size, num_heads) * num_layers

        # parameter check
        assert len(static_size) == self.num_layers
        for layer_id in range(self.num_layers):
            assert len(static_size[layer_id]) == len(cache_size[layer_id])

        # initialize the cache
        self.cache_position = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )
        self._kv_len = [
            0 for _ in range(self.num_layers)
        ]  # the length of the key and value cache for each layer

        # initialize as meta tensor to avoid extensive space on single gpu for multi-gpu inference
        self.key_cache: List[Tensor] = [
            torch.zeros(
                batch_size,
                total_cache_size_this,
                head_dim,
                dtype=self.dtype,
                device='meta',
            )
            for total_cache_size_this in self.layer_cache_size
        ]
        self.value_cache: List[Tensor] = [
            torch.zeros(
                batch_size,
                total_cache_size_this,
                head_dim,
                dtype=self.dtype,
                device='meta',
            )
            for total_cache_size_this in self.layer_cache_size
        ]
        self.mask_cache: List[torch.LongTensor] = [
            torch.zeros(
                batch_size, total_cache_size_this, dtype=torch.int64, device=device
            )
            for total_cache_size_this in self.layer_cache_size
        ]  # 1 means not masked, 0 means masked
        self.cache_valid_length: List[torch.LongTensor] = [
            torch.zeros(
                (batch_size, self.num_head_for_each_layer[layer_id]),
                dtype=torch.int64,
                device=device,
            )
            for layer_id in range(self.num_layers)
        ]

    @staticmethod
    def head_start_index_valid_length_to_head_index(
        head_start_index: Tensor,
        head_valid_length: Tensor,
    ) -> Tensor:
        """
        Convert the starting index and valid length of each head to the head index.
        """
        assert (head_start_index == head_start_index[0]).all().item()
        assert (head_valid_length == head_valid_length[0]).all().item()
        head_index = torch.cat((head_start_index[0], (head_start_index[0][-1]+head_valid_length[0][-1]).reshape(-1))) # head_index = past_key_value.head_index[self.layer_idx]
        return head_index.contiguous()

    @staticmethod
    def head_index_to_head_start_index_valid_length(
        head_index: Tensor,
        batch_size: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        """
        Convert the head index to the starting index and valid length of each head.
        """
        # valid length is more complex than head_index
        raise NotImplementedError
    
        # Calculate head_start_index from head_index
        head_start_index = head_index[:-1]

        # Calculate head_valid_length from head_index
        # The length of each segment is the difference between consecutive entries in head_index
        head_valid_length = head_index[1:] - head_index[:-1]

        # expand by batch size
        head_start_index = head_start_index[None, :].expand(batch_size, -1)
        head_valid_length = head_valid_length[None, :].expand(batch_size, -1)

        return head_start_index.contiguous(), head_valid_length.contiguous()


    @staticmethod
    def to_uncontigious(
        tensor: Tensor,
        head_index: Tensor,
    ) -> List[Tensor]:
        """
        Split the tensor to each head according to the head_index

        Parameters:
            tensor (`Tensor`):
                The expected shape for each tensor is `[batch_size, \sum_h^H cache_size_of_head_h , head_dim]`.
            head_index (`Tensor`):
                The starting index of each head, shape (num_heads+1).

        Return:
            A list of tensors, each tensor is the cache for each head of shape `[batch_size, cache_size_of_head_h, head_dim]`.
        """
        return [
            tensor[:, head_index[head_id] : head_index[head_id + 1], :]
            for head_id in range(len(head_index) - 1)
        ]

    @staticmethod
    def to_group_contigious(
        tensor: Tensor,
        head_index: Tensor,
    ) -> List[Tensor]:
        """
        Split the tensor into each group, where heads within the group share the same cache size.

        Parameters:
            tensor (`Tensor`):
                The expected shape for each tensor is either `[batch_size, \sum_h^H cache_size_of_head_h, head_dim]`
                or `[batch_size, \sum_h^H cache_size_of_head_h]`.
            head_index (`Tensor`):
                The starting index of each head, shape (num_heads+1).

        Return:
            A list of tensors, each tensor is the cache for each group. The shape of each tensor will be
            `[batch_size, num_heads_in_group, cache_size_of_head, head_dim]` if head_dim is present,
            otherwise `[batch_size, num_heads_in_group, cache_size_of_head]`.
        """
        groups = []
        batch_size = tensor.shape[0]
        has_head_dim = len(tensor.shape) == 3
        if has_head_dim:
            head_dim = tensor.shape[2]

        cache_size = head_index[1:] - head_index[:-1]  # shape: (num_heads)

        # Identify unique consecutive cache sizes and their boundaries
        unique_sizes, inverse_indices, counts = torch.unique_consecutive(
            cache_size, return_inverse=True, return_counts=True
        )

        # Prepare to group
        current_idx = 0
        for size, count in zip(unique_sizes, counts):
            # Start and end indices in head_index
            group_start_index = current_idx
            group_end_index = group_start_index + count

            # Slicing tensor according to the computed start and end
            start = head_index[group_start_index].item()
            end = head_index[group_end_index - 1].item() + size.item()

            # Extract the relevant slice from the tensor
            if has_head_dim:
                group_tensor = tensor[:, start:end, :]
                # Reshape to add the head dimension: [batch_size, num_heads_in_group, cache_size_of_head, head_dim]
                group_tensor = group_tensor.view(
                    batch_size, count, size.item(), head_dim
                )
            else:
                group_tensor = tensor[:, start:end]
                # Reshape to add the head dimension: [batch_size, num_heads_in_group, cache_size_of_head]
                group_tensor = group_tensor.view(batch_size, count, size.item())

            # Append to groups
            groups.append(group_tensor.contiguous())

            # Update the current index
            current_idx = group_end_index

        return groups

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        # position_ids: Tensor, # can be used to move the sink part to the left side
        attention_mask: Optional[torch.IntTensor] = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters:
            key_states (`Tensor`):
                The expected shapes for each key_states or value_states are
                    `[batch_size, num_heads, seq_len, head_dim]`.
            value_states (`Tensor`):
                The expected shapes for each key_states or value_states are
                    `[batch_size, num_heads, seq_len, head_dim]`.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            attention_mask (`torch.BoolTensor`, `optional`):
                The attention mask to apply to the cache. If not provided, no mask will be applied. The mask is a int64 tensor, where 1 means preverse the value and 0 means masked. The expected shape is `[batch_size, seq_len]`.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `StaticCircularCache`.
        Return:
            A tuple containing the updated key and value states. The expected shapes for each key_states or value_states are `[batch_size, \sum_h^H cache_size_of_head_h , head_dim]`.
        """
        assert layer_idx < self.num_layers
        if key_states is not None:
            assert key_states.shape[1] == self.num_head_for_each_layer[layer_idx]

        # move the tensors to correct devices for multi-gpu inference
        device = key_states.device
        assert value_states.device == device, "key_states and value_states should be on the same device"
        if attention_mask is not None:
            assert attention_mask.device == device, "attention_mask should be on the same device as key_states"
        
        # instantiate meta tensor to the correct device
        if self.key_cache[layer_idx].device == torch.device('meta'):
            self.key_cache[layer_idx] = torch.zeros_like(self.key_cache[layer_idx], device=device)
        if self.value_cache[layer_idx].device == torch.device('meta'):
            self.value_cache[layer_idx] = torch.zeros_like(self.value_cache[layer_idx], device=device)
        self.mask_cache[layer_idx] = self.mask_cache[layer_idx].to(device=device)
        self.cache_update_index[layer_idx] = self.cache_update_index[layer_idx].to(device=device)
        self.cache_head_start_index[layer_idx] = self.cache_head_start_index[layer_idx].to(device=device)
        self.circular_cache_head_index[layer_idx] = self.circular_cache_head_index[layer_idx].to(device=device)
        self.cache_valid_length[layer_idx] = self.cache_valid_length[layer_idx].to(device=device)
        self.cache_max_length[layer_idx] = self.cache_max_length[layer_idx].to(device=device)
        self.circular_cache_size[layer_idx] = self.circular_cache_size[layer_idx].to(device=device)
        self.static_cache_size[layer_idx] = self.static_cache_size[layer_idx].to(device=device)

        # if do not update cache content, key and value may be None
        if self.update_cache_content:
            batch_size = key_states.shape[0]
            seq_len = key_states.shape[-2]
            head_dim = key_states.shape[-1]
        else:
            batch_size = self.batch_size
            seq_len = key_states.shape[-2] if key_states is not None else 1
            head_dim = self.head_dim
        num_head = self.num_head_for_each_layer[layer_idx]

        # Update the number of seen tokens
        if layer_idx == 0:
            self.cache_position += seq_len

        is_decode = seq_len == 1

        if is_decode:
            # only update the last token of the cache, use simpler logic to speedup
            update_hctx_index = self.cache_update_index[
                layer_idx
            ]  # shape (batch_size, num_heads)
            batch_index = torch.arange(
                start=0, end=batch_size, device=device, dtype=torch.int64
            ).unsqueeze(
                1
            )  # to index shape (batch_size, num_heads, ...)

            self.mask_cache[layer_idx][batch_index, update_hctx_index] = 1
            if self.update_cache_content:
                self.value_cache[layer_idx][batch_index, update_hctx_index] = value_states[:, :, -1, :]
                self.key_cache[layer_idx][batch_index, update_hctx_index] = key_states[:, :, -1, :]

            # update valid cache length
            self.cache_valid_length[layer_idx] += (
                self.cache_valid_length[layer_idx] < self.cache_max_length[layer_idx]
            )

            is_circular = (
                self._kv_len[layer_idx] + 1 > self.static_cache_size[layer_idx]
            )[
                None, :
            ]  # shape (1, num_heads)
            self.cache_update_index[layer_idx] = (
                (
                    self.cache_update_index[layer_idx]
                    + 1
                    - self.circular_cache_head_index[layer_idx][None, :]
                )
                % self.circular_cache_size[layer_idx][None, :]
                + self.circular_cache_head_index[layer_idx][None, :]
            ) * is_circular + (self.cache_update_index[layer_idx] + 1) * (~is_circular)

            self._kv_len[layer_idx] += 1

            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        else:
            # Use the attention mask to find the starting point of the valid inputs
            if attention_mask is not None:
                assert (
                    len(attention_mask.shape) == 2
                ), "The shape of the attention mask should be [batch_size, kv_len] and the dtype should be torch.int64."
                # attention_mask shape is (batch_size, self._kv_len[layer_idx])
                # take the newly added part of the attention mask
                attention_mask = attention_mask[
                    :, -seq_len:
                ]  # shape: [batch_size, seq_len]
            else:
                attention_mask = torch.ones(
                    (batch_size, seq_len), dtype=torch.int64, device=device
                )  # shape: [batch_size, seq_len]
            advancing_pos = (
                torch.cumsum(attention_mask, dim=-1, dtype=torch.int64) - 1
            )  # shape: [batch_size, seq_len]; For masked positions, they do not store in the cache nor advance the pos
            advancing_pos = torch.cat(
                [advancing_pos, advancing_pos[:, -1, None] + 1], dim=-1
            )  # shape: [batch_size, seq_len+1]
            reversed_advancing_pos = torch.flip(
                torch.cumsum(
                    torch.flip(attention_mask, dims=[-1]), dim=-1, dtype=torch.int64
                ),
                dims=[-1],
            )  # shape: [batch_size, seq_len]; For masked positions, they do not store in the cache nor advance the pos

            update_index_circular = (
                (
                    self.cache_update_index[layer_idx][:, None, :]
                    + advancing_pos[:, :, None]
                    - self.circular_cache_head_index[layer_idx][None, None, :]
                )
                % self.circular_cache_size[layer_idx][None, None, :]
            ) + self.circular_cache_head_index[layer_idx][
                None, None, :
            ]  # shape: (batch_size, seq_len+1, num_heads), ending at _kv_len > static_len
            update_index_static = (
                self.cache_update_index[layer_idx][:, None, :]
                + advancing_pos[:, :, None]
            )  # shape: (batch_size, seq_len+1, num_heads), ending at _kv_len <= static_len
            update_hctx_index = torch.where(
                self._kv_len[layer_idx] + advancing_pos[:, :, None]
                > self.static_cache_size[layer_idx][None, None, :],
                update_index_circular,
                update_index_static,
            )  # shape: (batch_size, seq_len+1, num_heads)

            # record index
            self.cache_update_index[layer_idx] = update_hctx_index[:, -1, :]
            update_hctx_index = update_hctx_index[:, :-1, :].permute(
                0, 2, 1
            )  # shape: (batch_size, num_heads, seq_len), value are in the realm of num_heads * seq_len
            batch_index = (
                torch.arange(batch_size, device=device)[:, None, None]
                .expand(-1, num_head, seq_len)
                .contiguous()
            )  # shape: (batch_size, num_heads, seq_len)

            # If the seq_len is so long that it > cache size, only the latter indexed key/value should be reserved
            # keep the index where ((A) the final size is less or equal than the circular cache size OR (B) index within static cache) AND (C) not masked
            valid_index_map = (
                (
                    reversed_advancing_pos[:, None, :]
                    <= self.circular_cache_size[layer_idx][None, :, None]
                )
                | (
                    update_hctx_index
                    < self.circular_cache_head_index[layer_idx][None, :, None]
                )
            ) & (
                attention_mask[:, None, :] == 1
            )  # shape (batch_size, num_heads, seq_len)

            # update valid cache length based on the valid index map
            self.cache_valid_length[layer_idx] += torch.sum(
                valid_index_map, dim=-1, dtype=torch.int64
            )
            self._kv_len[layer_idx] += seq_len

            # update cache
            valid_update_index = update_hctx_index[
                valid_index_map
            ].contiguous()  # shape: (batch_size * num_heads * seq_len)
            valid_batch_index = batch_index[
                valid_index_map
            ].contiguous()  # shape: (batch_size * num_heads * seq_len)

            self.mask_cache[layer_idx][valid_batch_index, valid_update_index] = (
                attention_mask[:, None, :].expand(-1, num_head, -1)[valid_index_map]
            )  # assign tensor of shape (batch_size * num_heads * seq_len)
            self.key_cache[layer_idx][valid_batch_index, valid_update_index, :] = (
                key_states[valid_index_map, :]
            )  # assign tensor of shape (batch_size * num_heads * seq_len, head_dim)
            self.value_cache[layer_idx][valid_batch_index, valid_update_index, :] = (
                value_states[valid_index_map, :]
            )  # assign tensor of shape (batch_size * num_heads * seq_len, head_dim)

            assert self.key_cache[layer_idx].is_contiguous(), "key_cache is not contiguous"
            assert self.value_cache[layer_idx].is_contiguous(), "value_cache is not contiguous"
            assert self.mask_cache[layer_idx].is_contiguous(), "mask_cache is not contiguous"

            # TODO: for multi-round conversation, how to deal with the caches from the previous rounds whose lengths are different. Ignoring all history for now
            return key_states, value_states

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
    split_size: bool = 64,
    verbose: bool = True,
):
    """
    Convert the MoA configuration to the cache configuration

    Parameters:
        moa_config (`Dict`):
            The MoA configuration.
        seq_len (int):
            The sequence length.
        max_new_token (int, optional):
            The maximum number of new tokens. The N used to calculate cache size for each head equals seq_len + max_new_token. Defaults to 1024.
        sink_size (int, optional):
            The sink size. Defaults to 64.
        minimum_cache_size (int, optional):
            The minimum cache size. Defaults to 128.
        split_size (int, optional):
            The cache size of each head should be a multiple of this number. Defaults to 64.
        verbose (bool, optional):
            Whether to print the cache configuration summary. Defaults to True.

    Returns:
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
            cache_size_this_head = int(
                alphas[layer_id][head_id]
                + (seq_len + max_new_token) * betas[layer_id][head_id]
            )
            cache_size_this_head = min(
                math.ceil(max(cache_size_this_head, minimum_cache_size) / split_size)
                * split_size,
                seq_len + max_new_token,
            )
            cache_size_this_layer.append(cache_size_this_head)
            static_size_this_layer.append(min(cache_size_this_head, sink_size))
        cache_size_dict.append(cache_size_this_layer)
        static_size_dict.append(static_size_this_layer)

    if verbose:
        print("Cache configuration")
        summary = []
        for layer_id in range(len(alphas)):
            for head_id in range(len(alphas[layer_id])):
                summary.append(
                    {
                        "layer_id": layer_id,
                        "head_id": head_id,
                        "raw_cache_size": seq_len + max_new_token,
                        "cache_size": cache_size_dict[layer_id][head_id],
                        "static_size": static_size_dict[layer_id][head_id],
                        "circular_size": cache_size_dict[layer_id][head_id]
                        - static_size_dict[layer_id][head_id],
                        "ratio": cache_size_dict[layer_id][head_id] / (seq_len + max_new_token),
                    }
                )
        summary = pd.DataFrame(summary)
        pd.options.display.float_format = (
            "{:.2f}".format
        )  # keep two digits for all values during printing
        # reduce the summary for each layer
        layer_summary = (
            summary.groupby("layer_id")
            .agg(
                {
                    "raw_cache_size": "mean",
                    "cache_size": ["mean", "min", "max"],
                    "static_size": ["mean", "min", "max"],
                    "circular_size": ["mean", "min", "max"],
                    "ratio": ["mean", "min", "max"],
                }
            )
            .reset_index()
        )
        print(layer_summary)
        # reduce the summary for the whole model
        model_summary = summary.agg(
            {
                "raw_cache_size": ["mean", "min", "max"],
                "cache_size": ["mean", "min", "max"],
                "static_size": ["mean", "min", "max"],
                "circular_size": ["mean", "min", "max"],
                "ratio": ["mean", "min", "max"],
            }
        ).T
        model_summary.columns = ["mean", "min", "max"]
        model_summary = model_summary.unstack().reset_index()
        model_summary.columns = ["metric", "stat", "value"]
        model_summary = model_summary.pivot(
            index="metric", columns="stat", values="value"
        ).reset_index()
        print(model_summary)

    return {
        "cache_size": cache_size_dict,
        "static_size": static_size_dict,
    }


if __name__ == "__main__":
    batch_size = 3
    head_dim = 4
    num_head = 2
    device = "cuda"

    cache = StaticCircularCache(
        cache_size=[[5, 10], [5, 5]],
        static_size=[[2, 3], [2, 3]],
        batch_size=batch_size,
        head_dim=head_dim,
        device=device,
        dtype=torch.float16,
    )

    acc = 0
    seq_lens = [14, 9, 5]
    seq_len = max(seq_lens)
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.int64, device=device)
    for i, this_seq_len in enumerate(seq_lens):
        attention_mask[i, -this_seq_len:] = 1

    for step in range(0, 10):
        key_states = (
            torch.arange(
                acc, num_head * seq_len + acc, dtype=torch.float16, device=device
            )[None, :, None]
            .expand(batch_size, -1, head_dim)
            .reshape(batch_size, num_head, seq_len, head_dim)
        )
        value_states = (
            torch.arange(
                acc, num_head * seq_len + acc, dtype=torch.float16, device=device
            )[None, :, None]
            .expand(batch_size, -1, head_dim)
            .reshape(batch_size, num_head, seq_len, head_dim)
        )

        acc += num_head * seq_len

        print(key_states)
        print(attention_mask)

        key_cache_new, value_cache_new = cache.update(
            key_states, value_states, 0, attention_mask
        )

        # if seq_len == 1:
        # list_key_cache_new = StaticCircularCache.to_uncontigious(key_cache_new, cache.head_index[0])
        group_key_cache_new = StaticCircularCache.to_group_contigious(
            cache.key_cache[0], cache.head_index[0]
        )
        group_attention_mask = StaticCircularCache.to_group_contigious(
            cache.mask_cache[0], cache.head_index[0]
        )
        # print(list_key_cache_new[0][1]) # head 0, total size 5, static size 2
        # print(list_key_cache_new[1][1]) # head 1, total size 10, static size 3
        for group in range(2):
            print(group_key_cache_new[group])
            print(group_attention_mask[group])

        print("done")

        seq_len = 1
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(batch_size, seq_len, dtype=torch.int64, device=device),
            ],
            dim=-1,
        )
