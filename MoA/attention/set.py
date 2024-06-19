import torch
from torch.nn import ModuleList
from typing import List
from MoA.models.llama.density_calculation import lut_kv_cache_density, lut_attention_density

def set_static_attention_lut(attention_lut_path_list: List[str], attention_lut_for_head_path: str, model_layers: ModuleList = None, block_size: int = 64, permute_head=False, sparse_decode=False):
    """
    Apply the efficient attention
    """

    if permute_head:
        print("Permute the head of the LUT and attention layer")

    if isinstance(attention_lut_path_list, str):
        attention_lut_path_list = [attention_lut_path_list]

    for lut_str in attention_lut_path_list:
        print(f"{lut_str}: \n   attention density: {lut_attention_density(lut_str, block_size)[1]} \n   kv cache density: {lut_kv_cache_density(lut_str, block_size)[1]}")

    # load the attention mask from the file
    lut_list = [torch.load(attention_lut_path) for attention_lut_path in attention_lut_path_list]

    # set the attention lut and lut_for_head for each layer
    for layer_index, layer in enumerate(model_layers):
        layer.set_static_attention_lut([lut[layer_index].to('cuda') for lut in lut_list], None, block_size, 'cuda', permute_head, sparse_decode)