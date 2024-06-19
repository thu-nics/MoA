import torch
from torch import Tensor
import matplotlib.pyplot as plt
from tqdm import tqdm

from MoA.attention.pattern import gen_causal_pattern
from MoA.attention.convert import layout_to_lut_single_density

# load data
# layout_path = 'local/universal/test-model/profile_4096/generate_2048/optimize/ratio_0.50/layout_original_2048.pt'
# layout_path = 'local/universal/test-model/profile_4096/generate_2048/optimize/ratio_0.50/layout_original_2048_anchor_4096.pt'
layout_path='local/universal/test-model/profile_4096/layout_8192.pt'


layout = torch.load(layout_path)
num_layer, num_head, num_block, _ = layout.shape

causal_layout = gen_causal_pattern(num_block, num_block, dtype=torch.bool)
density = torch.sum(layout) / torch.sum(causal_layout) / num_layer / num_head

print('density', density.item())

row_num = 32
size_for_each_subfigure=2

fig, axs = plt.subplots(num_layer * num_head // row_num, row_num, figsize=(size_for_each_subfigure*row_num, size_for_each_subfigure*(num_layer*num_head//row_num)))

for i in range(num_layer * num_head):
    axs[i//row_num, i%row_num].imshow(layout[i//num_head, i%num_head], cmap='hot')
# 调整布局以防止子图之间的重叠
plt.tight_layout()
# 保存图片
plt.savefig('layout.png')