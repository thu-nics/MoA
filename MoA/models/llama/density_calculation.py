import torch
from typing import List

def streamingllm_attention_density(
    global_size: int = 4,
    band_size: int = 1024,
    kv_seq_len: int = 8192,
):  
    num_total = 0
    num_attended = 0

    for i in range(kv_seq_len):
        for j in range(kv_seq_len):
            if i < j:
                continue
            num_total += 1

            if (j < global_size) or (i - j < band_size):
                num_attended += 1
    
    return num_attended / num_total

def streamingllm_kv_cache_density(
    global_size: int = 4,
    band_size: int = 1024,
    kv_seq_len: int = 8192,
):
    if global_size + band_size > kv_seq_len:
        return 1.0

    else:
        return (global_size + band_size) / kv_seq_len

### calculate lut density directly using block. This may incur small error since on diagonal, the number of attended elements is not exactly the whole block
def count_distinct_elements_per_row(
    lut: torch.IntTensor,
):
    assert lut.dim() == 2
    M, N = lut.shape
    distinct_counts = torch.zeros(M, dtype=torch.long)

    for i in range(M):
        row_unique = torch.unique(lut[i])
        distinct_counts[i] = row_unique.numel()

    return distinct_counts

def lut_single_layer_attention_density(
    lut: torch.IntTensor,
    block_size: int = 64
):
    assert lut.dim() == 3

    kv_seq_block_len = lut.shape[1]

    num_total = lut.shape[0] * (kv_seq_block_len * (kv_seq_block_len + 1)) // 2
    num_attended = 0

    for i in range(lut.shape[0]):
        num_attended += count_distinct_elements_per_row(lut[i]).sum().item()

    return num_attended / num_total

def lut_single_layer_kv_cache_density(
    lut: torch.IntTensor,
    block_size: int = 64
):
    assert lut.dim() == 3

    kv_seq_block_len = lut.shape[1]

    num_total = lut.shape[0] * kv_seq_block_len
    num_kv_used = 0

    for i in range(lut.shape[0]):
        num_kv_used += count_distinct_elements_per_row(lut[i, -1:]).sum().item()

    return num_kv_used / num_total

def lut_attention_density(
    lut: List[torch.IntTensor],
    block_size: int = 64
):
    if isinstance(lut, str):
        lut = torch.load(lut)

    density_list = []
    for i in range(len(lut)):
        density_list.append(lut_single_layer_attention_density(lut[i], block_size))

    return density_list, sum(density_list) / len(density_list)

def lut_kv_cache_density(
    lut: List[torch.IntTensor],
    block_size: int = 64
):
    if isinstance(lut, str):
        lut = torch.load(lut)

    density_list = []
    for i in range(len(lut)):
        density_list.append(lut_single_layer_kv_cache_density(lut[i], block_size))

    return density_list, sum(density_list) / len(density_list)


if __name__ == '__main__':
    print(streamingllm_attention_density(4, 1024, 8192))
    print(streamingllm_kv_cache_density(4, 1024, 8192))

    lut_path = "/mnt/public/autofit/universal/vicuna-7b-v1.5/q_long/half_density/lut_4096_plan_2.pt"

    attention_density_list, attention_density = lut_attention_density(lut_path)
    kv_cache_density_list, kv_cache_density = lut_kv_cache_density(lut_path)

    print(attention_density, kv_cache_density)