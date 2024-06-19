import torch
from torch.nn import ModuleList
from torch import Tensor, BoolTensor
from typing import Union, List
from transformers import PreTrainedModel
import math
from tqdm import tqdm
import torch.nn.functional as F

from MoA.attention.pattern import gen_causal_pattern

### utils for block sparse attention

def block_sparse_to_dense(sparse_matrix, layout, batch_size, num_heads, token_length, block_size):
    '''
    sparse_matrix: shape: (batch_size, num_non_zero_blocks, block_size, block_size)
    layout: shape: (num_heads, num_blocks, num_blocks)
    '''
    precision = sparse_matrix.dtype
    device = sparse_matrix.device

    layout_flatten = layout.reshape(-1) # shape: (num_heads * num_blocks * num_blocks)
    num_blocks = layout.shape[1]
    # insert zero matrix to sparse matrix
    num_non_zero_blocks = sparse_matrix.shape[1]
    block_fill_index = torch.cumsum(layout_flatten, dim=0) - 1 # shape: (num_heads * num_blocks * num_blocks)
    block_fill_index[layout_flatten==0] = num_non_zero_blocks
    zero_block = torch.zeros((batch_size, 1, block_size, block_size), dtype=precision, device=device)
    # fill in the zero blocks into the sparse matrix based on the layout
    unfold_dense_matrix = torch.cat([sparse_matrix, zero_block], dim=1) # shape: (batch_size, num_non_zero_blocks + num_zero_blocks, block_size, block_size)
    dense_matrix = unfold_dense_matrix[:, block_fill_index] # shape: (batch_size, num_heads * num_blocks * num_blocks, block_size, block_size)

    # reshape the dense matrix to the dense attention weights
    dense_matrix = dense_matrix.view(batch_size, num_heads, num_blocks, num_blocks, block_size, block_size)
    dense_matrix = dense_matrix.permute(0, 1, 2, 4, 3, 5) # shape: (batch_size, num_heads, block_size, num_blocks, block_size, num_blocks)
    dense_matrix = dense_matrix.reshape(batch_size, num_heads, token_length, token_length) # shape: (batch_size, num_heads, token_length, token_length)

    return dense_matrix

"""
Convert beteen all kinds of formats
"""

"""
Convert from mask
"""

def mask_to_layout(mask: Tensor, block_size: int) -> BoolTensor:
    '''
    Input:
        mask: Tensor of shape (..., token_length, token_length)
    Output:
        layout: Boolean tensor of shape (..., num_block_x, num_block_y)
    '''
    avg = F.avg_pool2d(mask.float(), block_size)
    return avg > 0.5

"""
Convert from layout
"""

def layout_to_lut(layout: torch.BoolTensor):
    """
    input:
        layout: (layer, num_heads, num_block, num_block)
    output:
        lut: a tuple has 'layer' elements, each element has the size of (lut_num_heads, num_block, nnz)
        lut_for_head: a tuple has 'layer' elements, each element has the size of (lut_num_heads)
                      we use it as an indicator when combine heads 
        copy_times: (layer, num_heads, 1)
    """
    DeprecationWarning("This function is deprecated. Use layout_to_lut_single_densitys instead.")

    assert layout.dim() == 4, "The layout should have 4 dimensions: (layer, num_heads, num_block, num_block)"

    layer = layout.shape[0]
    num_heads = layout.shape[1]
    num_block = layout.shape[2]

    nnz_min_num = torch.full((layer,1), num_block)
    copy_times = torch.empty(layer, num_heads, 1)

    lut = ()
    lut_for_head = ()

    for i in range(layer):
        copy = torch.zeros(num_heads, 1)
        for j in range(num_heads):
            # to find out the head with the most density
            line = layout[i, j, -1, :]
            nnz = torch.sum(line).item()
            if(nnz <= nnz_min_num[i]):
                nnz_min_num[i] = nnz

        cnt = 0
        for j in range(num_heads):
            line = layout[i, j, -1, :]
            nnz = torch.sum(line).item()
            cnt += nnz/nnz_min_num[i]
            copy[j] = nnz/nnz_min_num[i]
            copy_times[i, j] = int(copy[j].item())

        lut_num_heads = cnt
        head_lut = torch.empty((int(lut_num_heads.item()), num_block, nnz_min_num[i]))
        indicator = torch.empty(int(lut_num_heads.item()))

        for j in range(num_heads):
            if(j == 0):
                sum = 0
            else:
                sum = int(torch.sum(copy[:j]).item())
            for k in range(int(copy[j].item())):
                for l in range(num_block):
                    for m in range(nnz_min_num[i].item()):
                        index = k*nnz_min_num[i] + m + 1
                        line = layout[i, j, l, :]
                        nnz_indices = torch.nonzero(line).squeeze()
                        nnz_index = nnz_indices[index-1].item()
                        head_lut[sum+k, l, m] = nnz_index
                indicator[sum+k] = j

        lut = lut + (head_lut.to(torch.int64),)
        lut_for_head = lut_for_head + (indicator.to(torch.int64),)

    return lut, lut_for_head, copy_times.to(torch.int64)

def layout_to_lut_single_density(layout: torch.BoolTensor):
    """
    input:
        layout: (layer, num_heads, num_block, num_block)
    output:
        lut: a tuple has 'layer' elements, each element has the size of (num_heads, num_block, nnz)
    """
    assert layout.dim() == 4, "The layout should have 4 dimensions: (layer, num_heads, num_block, num_block)"

    layer = layout.shape[0]
    num_heads = layout.shape[1]
    num_block = layout.shape[2]

    lut = ()

    for i in range(layer):
        layer_mask = layout[i]
        max_nnz = torch.sum(layer_mask, dim=-1).max().cpu().item()

        one_matrix = torch.ones_like(layer_mask, dtype=torch.int, device=layer_mask.device)
        cum_matrix = torch.cumsum(one_matrix, dim=-1)
        masked_cum_matrix = cum_matrix * layer_mask # keep only entries that are True in attention mask. The value of each entry is column index plus one.
        max_matrix = masked_cum_matrix.max(dim=-1, keepdim=True)[0].repeat(1, 1, num_block)
        filled_matrix = masked_cum_matrix.detach().clone()
        filled_matrix[~layer_mask] = max_matrix[~layer_mask] # fill missing entries with largest value in the row.
        lut_layer = torch.sort(filled_matrix, dim=-1)[0] - 1 # make the index start from zero instead of one.

        lut_layer = lut_layer[:, :, :max_nnz]
        lut += (lut_layer.to(torch.int64), )

    return lut

def layout_to_mask(layout: torch.BoolTensor, block_size: int = 64, is_causal: bool = True) -> torch.BoolTensor:
    """
    Convert the layout to mask
    input:
        layout: (num_layer, num_head, num_block_x, num_block_y) or (num_head, num_block_x, num_block_y)
        block_size: int, the block size
        is_causal: bool, whether the attention is causal
    output:
        mask: (num_layer, num_head, num_block_x*block_size, num_block_y*block_size) or (num_head, num_block_x*block_size, num_block_y*block_size)
    """
    # Check the number of dimensions in layout and adjust accordingly
    dim = layout.dim()
    if dim == 4:
        num_layer, num_head, num_block_x, num_block_y = layout.shape
    elif dim == 3:
        num_layer = 1  # Set num_layer to 1 if it's missing
        num_head, num_block_x, num_block_y = layout.shape
        layout = layout.unsqueeze(0)  # Add a dimension for num_layer
    else:
        raise ValueError("Invalid layout shape. Expected 3 or 4 dimensions, got {}.".format(dim))

    mask = layout.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, block_size, block_size)
    mask = mask.permute(0, 1, 2, 4, 3, 5)
    mask = mask.reshape(num_layer, num_head, num_block_x*block_size, num_block_y*block_size).contiguous()

    if is_causal:
        # Assume gen_causal_pattern is defined elsewhere and handles num_layer correctly
        causal_mask = gen_causal_pattern(num_query=mask.size(-2), num_key=mask.size(-1), dtype=mask.dtype, num_layer=num_layer, num_head=num_head).to(mask.device)
        mask = mask & causal_mask
    
    if dim == 3:
        mask = mask.squeeze(0)

    return mask

"""
Convert from lut
"""

def lut2layout(lut: List[torch.IntTensor]) -> torch.BoolTensor:
    layout = []
    pbar = tqdm(len(lut), desc="Converting LUT to layout")
    for i in range(len(lut)):
        layout.append(lut2layout_single_layer(lut[i]))
        pbar.update(1)

    pbar.close()

    return torch.stack(layout, dim=0)

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

### lut to density ###
def lut_to_density(lut_path: str) -> list:
    """
    input: 
        lut_path: the path to load lut
    output:
        density_list: a list of density for each layer
    """

    density_list = []
    lut = torch.load(lut_path)
    layer = len(lut)
    for i in range (layer):
        N = lut[i].shape[1]
        n = lut[i].shape[2]
        density = (n*(n+1)/2+n*(N-n))/(N*(N+1)/2)
        density_list.append(density)

    return density_list


"""
Convert from nnz
"""

### nnz & num_block to density ###
def n_to_density(n: Union[int, Tensor], N: int) -> Union[float, Tensor]:
    """
    Convert the number of non-zero blocks out of the total number of blocks to density under casual setting
    input: 
        n: nnz
        N: num_block
    output:
        density: the density of the layout
    """
    density = (n*(n+1)/2+n*(N-n))/(N*(N+1)/2)
    return density
