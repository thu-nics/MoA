import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple

def permute_attention_projection(
    layer: nn.Linear, 
    permutation: Optional[torch.Tensor] = None,
    num_heads: Optional[int] = None,
    num_key_value_groups: Optional[int] = None,
) -> None:
    '''
    permute the weight of the q, k, v projection layer so that the head order is permuted
    '''
    
    if permutation is None:
        return

    assert permutation.dim() == 1

    assert permutation.shape[0] == num_heads
    if num_key_value_groups is not None and num_key_value_groups != 1:
        num_heads = num_heads * num_key_value_groups
        original_permutation_tensor = permutation
        original_permutation = permutation.tolist()
        permutation = []
        for index in original_permutation:
            permutation.extend([index * num_key_value_groups + i for i in range(num_key_value_groups)])

        # turn permutation to tensor
        permutation = torch.tensor(permutation, device=original_permutation_tensor.device, dtype=original_permutation_tensor.dtype)

    input_dim = layer.in_features
    output_dim = layer.out_features
    head_dim = output_dim // num_heads

    original_weights = layer.weight.data
    weights_reshaped = original_weights.reshape(num_heads, head_dim, input_dim)
    weights_permuted = weights_reshaped[permutation].reshape_as(original_weights)

    assert weights_permuted.is_contiguous()
        
    layer.weight.data = weights_permuted

    if layer.bias is not None:
        original_bias = layer.bias.data
        bias_reshaped = original_bias.reshape(num_heads, head_dim)
        bias_permuted = bias_reshaped[permutation].reshape_as(original_bias)

        assert bias_permuted.is_contiguous()

        layer.bias.data = bias_permuted
                                                                            
    return

def permute_output_projection(
    layer: nn.Linear, 
    permutation: Optional[torch.Tensor] = None,
    num_heads: Optional[int] = None,
    num_key_value_groups: Optional[int] = None,
) -> None:
    '''
    permute the weight of o_proj so that it can take in the permuted hidden states
    '''
    
    if permutation is None:
        return

    assert permutation.dim() == 1

    assert permutation.shape[0] == num_heads

    if num_key_value_groups is not None and num_key_value_groups != 1:
        num_heads = num_heads * num_key_value_groups
        original_permutation_tensor = permutation
        original_permutation = permutation.tolist()
        permutation = []
        for index in original_permutation:
            permutation.extend([index * num_key_value_groups + i for i in range(num_key_value_groups)])

        # turn permutation to tensor
        permutation = torch.tensor(permutation, device=original_permutation_tensor.device, dtype=original_permutation_tensor.dtype)


    input_dim = layer.in_features
    output_dim = layer.out_features
    head_dim = input_dim // num_heads

    original_weights = layer.weight.data
    weights_reshaped = original_weights.reshape(output_dim, num_heads, head_dim)
    weights_permuted = weights_reshaped.transpose(0,1)[permutation].transpose(0,1).reshape_as(original_weights)

    assert weights_permuted.is_contiguous()
        
    layer.weight.data = weights_permuted
                                                                            
    return
  

def permute_lut(
    lut: torch.Tensor,
    permutation: Optional[torch.Tensor] = None,
    num_heads: int = 32,
):

    if permutation is None:
        return lut
    
    assert permutation.dim() == 1
    assert permutation.shape[0] == num_heads
    assert lut.shape[0] == num_heads

    permuted_lut = lut[permutation]

    if not permuted_lut.is_contiguous():
        permuted_lut = permuted_lut.contiguous()

    return permuted_lut


def lut_to_permutation(
    lut_list: List[torch.Tensor],
    num_heads: int = 32,
    num_key_value_groups: int = 1,
):
    '''
    Find a permutation, such that after applying the permutation, the heads with same hyperparameters are clustered together

    Parameters:
        lut_list: (`List[torch.Tensor]`):
            A list of lut table
    
    Return:
        Permutation: (`torch.Tensor`):
            A 1D Tensor of shape `(num_heads,)` representing the original index of the heads before permutation
        Cluster: (`Dict[int, Tuple[int, int]]`):
            A dictionary of the form `{cluster_index: (start, end)}` where `start` and `end` are the range of the cluster after permutation
    '''

    assert lut_list[0].shape[0] == num_heads
    assert lut_list[0].dim() == 3
    assert num_heads % num_key_value_groups == 0

    # return a permutation, such that after applying the permutation, same heads clustered together
    sorted_indices_list = []
    cluster_list = []

    if num_key_value_groups > 1:
        # check that the pattern within a group is the same
        for i in range(num_key_value_groups):
            raise NotImplementedError

    for lut in lut_list:
        serialized_heads = [tuple(head.reshape(-1).tolist()) for head in lut]



        sorted_indices = sorted(range(len(serialized_heads)), key=lambda i: (serialized_heads[i], i))
        
        # get the cluster, in the form of dictionary. like {1:(0,4), 2:(4, 32)}. (0, 4) is the range
        cluster = {}
        current_cluster = 0
        start = 0
        for i in range(1, len(sorted_indices)):
            if serialized_heads[sorted_indices[i]] != serialized_heads[sorted_indices[i-1]]:
                cluster[current_cluster] = (start, i)
                start = i
                current_cluster += 1

        cluster[current_cluster] = (start, len(sorted_indices))

        sorted_indices_list.append(sorted_indices)
        cluster_list.append(cluster)

    # check those cluster
    two_pattern_idx = []
    for i in range(0, len(cluster_list)):
        if len(cluster_list[i]) == 2:
            two_pattern_idx.append(i)
        elif len(cluster_list[i]) > 2:
            print(f"cluster_list[{i}] = {cluster_list[i]}")
            raise NotImplementedError
                
    # all lut have just one pattern, return arbitrary one
    if len(two_pattern_idx) == 0:
        # manually make it two
        return_cluster = {0: (0, num_heads // 2), 1: (num_heads // 2, num_heads)}
        # return torch.tensor(sorted_indices_list[0]), cluster_list[0]
        return torch.tensor(sorted_indices_list[0]), return_cluster

    cluster = cluster_list[two_pattern_idx[0]]
    sorted_indices = sorted_indices_list[two_pattern_idx[0]]

    for i in two_pattern_idx:
        checked_lut = lut_list[i]
        checked_serialized_heads = [tuple(head.reshape(-1).tolist()) for head in checked_lut]
        # permute the head with sorted_indices
        permuted_serialized_heads = [checked_serialized_heads[j] for j in sorted_indices]

        # cluster the permuted heads
        current_cluster = 0
        start = 0
        checked_cluster = {}

        for j in range(1, len(permuted_serialized_heads)):
            if permuted_serialized_heads[j] != permuted_serialized_heads[j-1]:
                checked_cluster[current_cluster] = (start, j)
                start = j
                current_cluster += 1
        
        checked_cluster[current_cluster] = (start, len(permuted_serialized_heads))

        # check after applying the permuation, whether
        if checked_cluster != cluster:
            raise NotImplementedError

    # return any of the two pattern stuff    
    return torch.tensor(sorted_indices_list[two_pattern_idx[0]]), cluster_list[two_pattern_idx[0]]
    
def moa_config_to_permutation(moa_config) -> Tuple[List[torch.Tensor], List[Dict[int, Tuple[int, int]]]]:
    '''
    Find a permutation, such that after applying the permutation, the heads with same hyperparameters are clustered together

    Parameters:
        moa_config: (`Dict`):
            A dictionary containing the MoA config. Two keys are used:
                'alphas': (`Union[List[torch.Tensor], List[List[int]]]`): A list of alpha values for each layer
                'betas': (`Union[List[torch.Tensor], List[List[int]]]`): A list of beta values for each layer
    
    Return:
        Permutation: (`List[torch.Tensor]`):
            Each element the permutation of a layer. Each element is a 1D Tensor of shape `(num_heads,)` representing the original index of the heads before permutation
        Cluster: (`List[Dict[int, Tuple[int, int]]]`):
            Each element is a dictionary of the form `{cluster_index: (start, end)}` where `start` and `end` are the range of the cluster after permutation
    '''
    
    permutations = []
    clusters = []
    
    # Iterate over each layer
    for layer_index, (layer_alphas, layer_betas) in enumerate(zip(moa_config['alphas'], moa_config['betas'])):
        num_heads = len(layer_alphas)
        
        # Pair each alpha and beta with its index
        combined = list(zip(layer_alphas, layer_betas, range(num_heads)))
        
        # Sort based on alpha and beta values
        sorted_combined = sorted(combined, key=lambda x: (x[0], x[1]))
        
        # Extract the sorted indices
        sorted_indices = [x[2] for x in sorted_combined]
        
        # Create permutation tensor
        permutation_tensor = torch.tensor(sorted_indices)
        permutations.append(permutation_tensor)
        
        # Track clusters
        cluster_dict = {}
        current_start = 0
        
        # Iterate through sorted heads to determine clusters
        for i in range(1, num_heads):
            if sorted_combined[i-1][:2] != sorted_combined[i][:2]:  # Check if current and previous differ
                cluster_dict[len(cluster_dict)] = (current_start, i - 1)
                current_start = i
        cluster_dict[len(cluster_dict)] = (current_start, num_heads - 1)  # Last cluster
        
        clusters.append(cluster_dict)
    
    return permutations, clusters


def lut_to_permutation_single_layer(
    lut: torch.Tensor,
    num_heads: int = 32,
):
    '''
    find a permutation, such that after applying the permutation, same heads clustered together
    also return the clustered index
    '''

    assert lut.shape[0] == num_heads
    assert lut.dim() == 3

    # return a permutation, such that after applying the permutation, same heads clustered together
    serialized_heads = [tuple(head.reshape(-1).tolist()) for head in lut]

    sorted_indices = sorted(range(len(serialized_heads)), key=lambda i: (serialized_heads[i], i))
    
    # get the cluster, in the form of dictionary. like {1:(0,4), 2:(4, 32)}. (0, 4) is the range
    cluster = {}
    current_cluster = 0
    start = 0
    for i in range(1, len(sorted_indices)):
        if serialized_heads[sorted_indices[i]] != serialized_heads[sorted_indices[i-1]]:
            cluster[current_cluster] = (start, i)
            start = i
            current_cluster += 1

    cluster[current_cluster] = (start, len(sorted_indices))

    # check those cluster
    two_pattern_idx = []

    if len(cluster) == 2:
        two_pattern_idx.append(0)
    elif len(cluster) > 2:
        print(f"cluster = {cluster}")
        raise NotImplementedError
                
    # all lut have just one pattern, return arbitrary one
    if len(two_pattern_idx) == 0:
        return torch.tensor(sorted_indices), cluster


    # return any of the two pattern stuff    
    return torch.tensor(sorted_indices), cluster



def get_lut_global_size(lut: torch.Tensor, block_size: int = 64):
    # !!! assume global size is just one block!!
    return block_size
    assert lut.dim() == 2

    last_line_lut = lut[-1]
    
    global_block_num = 0
    for i in range(len(last_line_lut)):
        if last_line_lut[i] == i:
            global_block_num += 1
            continue
        else:
            break
    
    return global_block_num * block_size

def get_lut_band_size(lut: torch.Tensor, block_size: int = 64):
    assert lut.dim() == 2

    last_line_lut = lut[-1]
        
    last_line_lut_unique = set(last_line_lut.tolist())

    return block_size * len(last_line_lut_unique) - get_lut_global_size(lut, block_size)