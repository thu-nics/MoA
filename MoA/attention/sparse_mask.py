import torch
from torch import Tensor
from typing import Optional, Tuple

from tqdm import tqdm
import cvxpy as cp
import numpy as np

def gen_block_sparse_layout_mask(matrix: Tensor, num_block_x: int, num_block_y: int, block_keep_num: Tensor, row_balance: bool = False, block_aggregation_func: Optional[callable] = None, is_causal: bool = False) -> Tuple[Tensor, Tensor]:    
    """
    Make a block sparse layout and mask by keeping the top block_keep_num blocks in each matrix. True for blocks to keep, False for blocks to prune.
    Note that each row may have different number of blocks to keep.
    Args:
        - matrix: shape: [keep_dim, matrix_size_x, matrix_size_y] or any size of [..., matrix_size_x, matrix_size_y]
        - num_block_x: number of blocks in x direction
        - num_block_y: number of blocks in y direction
        - block_keep_num: number of blocks to keep in each keep dimention, can be reshape to keep_dim. If row_balance is True, block_keep_num is the number of blocks to keep in each row. If row_balance is False, block_keep_num is the number of blocks to keep in each keep dimention.
        - row_balance: whether to balance the number of blocks to keep in each row. If True, block_keep_num is the number of blocks to keep in each row. If False, block_keep_num is the number of blocks to keep in each keep dimention.
        - block_aggregation_func: function to aggregate the block matrix, default to torch.mean. The input is a block matrix of shape [keep_dim, num_block_x, num_block_y, block_size_x, block_size_y], and the output is a matrix of shape [keep_dim, num_block_x, num_block_y]
    """
    matrix_size_x = matrix.shape[-2]
    matrix_size_y = matrix.shape[-1]
    original_shape = list(matrix.shape)
    matrix = matrix.reshape(-1, matrix_size_x, matrix_size_y)
    keep_dim = matrix.shape[0]
    
    block_size_x = matrix_size_x // num_block_x
    block_size_y = matrix_size_y // num_block_y
    assert block_size_x * num_block_x == matrix_size_x
    assert block_size_y * num_block_y == matrix_size_y
    
    # to block matrix
    block_matrix = matrix.reshape(keep_dim, num_block_x, block_size_x, num_block_y, block_size_y).contiguous() # reshape the dense matrix to the dense attention weights
    block_matrix = block_matrix.permute(0, 1, 3, 2, 4) # shape: [keep_dim, num_block_x, num_block_y, block_size_x, block_size_y]

    if block_aggregation_func is not None:
        aggregate_block_matrix = block_aggregation_func(block_matrix).to(torch.float32) # shape: [keep_dim, num_block_x, num_block_y]
    else:
        aggregate_block_matrix = torch.mean(block_matrix, dim=[-1, -2]).to(torch.float32) # shape: [keep_dim, num_block_x, num_block_y]

    # generate layout for block weight pruning
    device = matrix.device
    layout = torch.zeros((keep_dim, num_block_x, num_block_y), dtype=bool, device=device)

    # take the top head_percent of the attention map
    block_keep_num = block_keep_num.reshape(keep_dim)
    view_shape = [num_block_x, num_block_y] if row_balance else [-1]

    for i in range(keep_dim):
        _, indices = torch.topk(aggregate_block_matrix[i].view(*view_shape), int(block_keep_num[i]), dim=-1)
        layout[i].view(*view_shape).scatter_(-1, indices, 1)

    # repeat the layout to match the shape of the dense matrix; expand layout to dense matrix of shape (keep_dim, num_block_x, num_block_y, block_size_x, block_size_y)
    matrix_mask = layout.unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, block_size_x, block_size_y)

    matrix_mask = matrix_mask.permute(0, 1, 3, 2, 4) # transpose the matrix to match the shape of the dense matrix
    matrix_mask = matrix_mask.reshape(*original_shape).contiguous() # reshape to dense matrix

    output_layout_shape = original_shape
    output_layout_shape[-2] = num_block_x
    output_layout_shape[-1] = num_block_y
    layout = layout.reshape(*output_layout_shape).contiguous()

    if is_causal:
        # set the attention mask rule for each layer, mask the upper triangle
        causal_mask = torch.zeros((matrix_size_x,matrix_size_y), dtype=bool, device=device)
        mask_cond = torch.arange(causal_mask.size(-1), device=device)
        causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 1)
        matrix_mask = matrix_mask * causal_mask
    
        causal_layout = torch.zeros((matrix_size_x//block_size_x, matrix_size_y//block_size_y), dtype=bool, device=device)
        mask_cond = torch.arange(causal_layout.size(-1), device=device)
        causal_layout.masked_fill_(mask_cond < (mask_cond + 1).view(causal_layout.size(-1), 1), 1)
        layout = layout * causal_layout

    return layout, matrix_mask

def matrix_reserve_score(matrix: Tensor, mask: Tensor, is_causal: bool, norm_score: bool = False) -> Tensor:
    """
    Compute the reserve score of a matrix with shape [keep_dim] or any size of [...], which is the sum of the values of the **unmasked** elements. The more score, the more values are reserved.
    Args:
        - matrix: shape: [keep_dim, matrix_size_x, matrix_size_y] or any size of [..., matrix_size_x, matrix_size_y]
        - mask: shape: [keep_dim, matrix_size_x, matrix_size_y] or any size of [..., matrix_size_x, matrix_size_y]
        - is_causal: whether the matrix is causal, i.e. the upper triangular matrix is all -inf
        - norm_score: whether to normalize the score by the by the summation of original score
    """
    assert matrix.shape == mask.shape, "matrix and mask should have the same shape"
    if is_causal:
        # mask out the upper triangular matrix
        mask = torch.tril(mask, diagonal=0)
        matrix = torch.tril(matrix, diagonal=0)
    wo_norm_return = torch.sum(matrix * mask, dim=[-2, -1])
    if norm_score:
        w_norm_return = wo_norm_return / torch.sum(matrix, dim=[-2, -1])
        fairness_factor = torch.sum(wo_norm_return) / torch.sum(w_norm_return) # a number
        # if fairness_factor is nan, it means the original score is 0, then the fairness_factor is set to 0
        if torch.isnan(fairness_factor):
            fairness_factor = 0
        return w_norm_return * fairness_factor
    else:
        return wo_norm_return

# sum and scale
def _block_causal_matrix_mean_scale(block_matrix: Tensor):

    keep_dim = block_matrix.shape[0]
    num_block_x = block_matrix.shape[1]
    num_block_y = block_matrix.shape[2]
    block_size_x = block_matrix.shape[3]
    block_size_y = block_matrix.shape[4]

    # apply causal mask to block matrix
    aggregate_block_matrix = torch.mean(block_matrix, dim=[-1, -2]) # shape: [keep_dim, num_block_x, num_block_y]
            
    assert block_size_x == block_size_y, "block_size_x and block_size_y are not equal"

    # diagonal rescale
    # aggregate_block_matrix[:, torch.eye(num_block_x, dtype=bool)] = torch.tril(aggregate_block_matrix[:, torch.eye(num_block_x, dtype=bool)], diagonal=0)
    aggregate_block_matrix[:, torch.eye(num_block_x, dtype=bool)] *= 2 * block_size_x * block_size_x / (block_size_x * (block_size_x+1))

    # make upper triangular matrix to -inf
    aggregate_block_matrix = torch.triu(torch.ones_like(aggregate_block_matrix), diagonal=1) * torch.finfo(aggregate_block_matrix.dtype).min + aggregate_block_matrix
    
    return aggregate_block_matrix

from transformers.utils import is_peft_available
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
if is_peft_available():
    from peft import PeftModel

def Trainer_attention_grad_compute_loss(self, model, inputs, return_outputs=False):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    if self.label_smoother is not None and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None

    ### begin only used for attention matrix grad log ###
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions
    attention_grads = [attention.grad for attention in attentions]
    if len(attentions[0].shape) == 4:
        # attentions: (batch_size, num_heads, seq_length, seq_length)
        atten_map = torch.stack(outputs.attentions, dim=1).to('cpu') # shape: (batch_size, num_layers, num_heads, seq_length, seq_length)
    elif len(attentions[0].shape) == 3:
        # attentions: (num_heads, seq_length, seq_length)
        atten_map = torch.stack(outputs.attentions, dim=0).unsqueeze(0).to('cpu')
    else:
        raise ValueError('The shape of attention is not correct.')
        # the shape of atten_map is (batch_size, num_layers, num_heads, seq_length, seq_length)
    self.atten_map = atten_map
    self.attention_grads = attention_grads
    ### end only used for attention matrix grad log ###

    # Save past state if it exists
    # TODO: this needs to be fixed and made cleaner later.
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    if labels is not None:
        if is_peft_available() and isinstance(model, PeftModel):
            model_name = unwrap_model(model.base_model)._get_name()
        else:
            model_name = unwrap_model(model)._get_name()
        if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = self.label_smoother(outputs, labels)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    return (loss, outputs) if return_outputs else loss

def optimize_block_nnz(z_score: Tensor, density: Tensor, avg_density: int) -> Tensor:
    """
    Optimize the block sparsity of a matrix by maximizing the sum of the z_score of the blocks with the constraint that the number of non-zero blocks is less than or equal to nnz_bound.
    Args:
        - z_score: shape: [num_layers, num_heads, num_density]
        - density: shape: [num_layers, num_heads, num_density]
        - nnz_bound: number of non-zero blocks
    """
    num_layer, num_head, num_density = z_score.shape
    
    keep_dim = num_layer * num_head
    z_score = z_score.reshape(keep_dim, num_density)
    density = density.reshape(keep_dim, num_density)

    # Objective. Construct a CVXPY problem
    
    # x_{keep_dim, density} in 1 dimension
    x = cp.Variable((keep_dim * num_density, 1), integer=True) 

    # Z_{keep_dim, density} in 1 dimension
    Z = z_score.reshape(1, keep_dim * num_density).cpu().numpy()

    objective = cp.Maximize(Z @ x)

    # Constrain, choose one density for each head
    # H_{keep_dim, density * keep_dim} in 2 dimension. H @ x = 1
    # [[1,..,1,0,...,0,0,...,0],
    #  [0,..,0,1,...,1,0,...,0],
    #  [0,..,0,0,...,0,1,...,1]]
    H = np.zeros((keep_dim, keep_dim * num_density))
    for i in range(keep_dim):
        H[i, i*num_density:(i+1)*num_density] = 1

    # D_{keep_dim, density} in 1 dimension
    D = density.reshape(1, keep_dim * num_density).cpu().numpy()

    constraints = [D @ x <= avg_density * keep_dim, H @ x == 1, x >= 0, x <= 1]

    # Define and solve the CVXPY problem. (MIP_solver=cp.ECOS_BB)
    prob = cp.Problem(objective, constraints)
    prob.solve(MIP_solver=cp.MOSEK)

    print("Status: ", prob.status)
    # print("The optimal value is", prob.value)
    # print("A solution x is")
    # print(np.round(x.value))

    onehot_select = np.round(x.value).reshape(keep_dim, num_density)

    # select the keep_dim density with onehot_select
    return density[onehot_select == 1].reshape(num_layer, num_head)

def causal_density_to_topk(num_block: int, density: float):
    """
    Given the density, return the topk to keep
    Args:
        - num_block: number of blocks
        - density: density of the matrix
    """
    topk_table = torch.arange(1, num_block + 1)
    topk_to_density = ((2*num_block + 1) * topk_table - topk_table*topk_table) / (num_block * (num_block + 1))  # given topk, get density
    return(int(torch.searchsorted(topk_to_density, density, right=True)))

def mip_gen_block_sparse_layout_mask(attention_effect: Tensor, avg_density: float, block_size: int = 32, use_norm_z_score: bool = True, is_causal: bool = True, profile_dict: dict = dict(),nnz_list: list = None, density_list: list=None) -> Tuple[Tensor, Tensor]:
    """
    Generate a block sparse layout and mask by keeping the top block_keep_num blocks in each matrix. True for blocks to keep, False for blocks to prune. The block layout is generated by solving a mixed integer programming problem. The objective is to maximize the sum of the z_score of the blocks with the constraint that the average density is less than or equal to avg_density. 
    Args:
        - attention_effect: shape: [num_layer, num_head, num_query, num_key]
        - density_list: list of density to try
        - avg_density: average density constraint
        - block_size: size of the block
        - use_norm_z_score: whether to normalize the z_score by the sum of elements in the matrix
        - is_causal: whether the matrix is causal, i.e. the upper triangular matrix is all -inf
        - profile_dict: dict to store the profile information. The keys can be 'z_score' and 'opt_density'
    Returns:
        - layout: shape: [num_layer, num_head, num_query//block_size, num_key//block_size]
        - mask: shape: [num_layer, num_head, num_query, num_key]
    """
    num_layer, num_head, num_query, num_key = attention_effect.shape

    num_block = num_key // block_size

    attention_effect = torch.abs(attention_effect.detach())
    # mean_attention_matrix = -mean_attention_effect.detach() 

    block_keep_num_ones = torch.ones(num_layer, num_head)

    if (nnz_list is not None) and (density_list is not None):
        raise ValueError("nnz_list and density_list cannot be both not None")
    
    if nnz_list is not None:
        num_density = len(nnz_list)
        block_keep_num_table = nnz_list
        nnz_tensor = torch.tensor(nnz_list)
        density_tensor = (nnz_tensor*(nnz_tensor+1)/2+nnz_tensor*(num_block-nnz_tensor))/(num_block*(num_block+1)/2)
        density_list = density_tensor.tolist()
    elif density_list is not None: 
        if is_causal:
            block_keep_num_table = [causal_density_to_topk(num_block, density) for density in density_list]
        else:
            block_keep_num_table = [int(num_block * density) for density in density_list]
        num_density = len(density_list)
    else:
        raise ValueError("nnz_list and density_list cannot be both None")
    print("the block_keep_num_table is {}".format(block_keep_num_table))

    z_score = torch.empty(num_density, num_layer, num_head)

    for i, keep_num in enumerate(tqdm(block_keep_num_table)):
        _, mask = gen_block_sparse_layout_mask(attention_effect, num_block, num_block, block_keep_num_ones * keep_num, row_balance=True, block_aggregation_func=_block_causal_matrix_mean_scale)
        z_score[i] = matrix_reserve_score(attention_effect, mask, is_causal=True, norm_score=use_norm_z_score)
        print("the average z_score of density {:.2f} ({}) is {:.2f}".format(density_list[i], nnz_list[i], torch.mean(z_score[i]).item()))

    z_score = torch.permute(z_score, (1, 2, 0)) # shape (num_layers, num_heads, num_density)
    density_tensor = torch.tensor(density_list).reshape(1, 1, -1).repeat(num_layer, num_head, 1) # shape (num_layers, num_heads, num_density)

    print("optimizing block nnz ...")
    opt_density = optimize_block_nnz(z_score=z_score, density=density_tensor, avg_density=avg_density)
    opt_topk = opt_density.clone().apply_(lambda x: causal_density_to_topk(num_block, x))
    print("the optimized objective average density is {}".format(torch.mean(opt_density).item()))

    layout, mask = gen_block_sparse_layout_mask(attention_effect, num_block, num_block, opt_topk, row_balance=True, block_aggregation_func=_block_causal_matrix_mean_scale, is_causal=is_causal)
    score = matrix_reserve_score(attention_effect, mask, is_causal=True, norm_score=use_norm_z_score)
    
    print("the optimal score is {}".format(torch.mean(score).item()))

    if is_causal:
        real_average_density = torch.sum(mask) / (num_layer * num_head * torch.sum(causal_mask))
    else:
        real_average_density = torch.sum(mask) / (num_layer * num_head * num_query * num_key)
    print("the final average density is {}".format(real_average_density.item()))

    mask = mask.cpu().detach().to(bool)
    layout = layout.cpu().detach().to(bool)

    if 'z_score' in profile_dict.keys():
        profile_dict['z_score'] = z_score
    if 'opt_density' in profile_dict.keys():
        profile_dict['opt_density'] = opt_density
    if 'block_keep_num_table' in profile_dict.keys():
        profile_dict['block_keep_num_table'] = block_keep_num_table
    if 'opt_block_keep_num_table' in profile_dict.keys():
        profile_dict['opt_block_keep_num_table'] = opt_topk
    if 'opt_z_score' in profile_dict.keys():
        # show the position of opt_block_keep_num in block_keep_num_table
        opt_choice = torch.zeros(num_layer, num_head, dtype=torch.long)
        for i, keep_num in enumerate(block_keep_num_table):
            index = torch.where(opt_topk == keep_num)
            opt_choice[index] = i
        opt_z_score = torch.gather(z_score, dim=-1, index=opt_choice.unsqueeze(-1)).squeeze(-1)
        profile_dict['opt_z_score'] = opt_z_score

    return layout, mask

def mip_gen_layer_block_sparse_layout_mask(attention_effect: Tensor, density_list: list, avg_density: float, block_size: int = 32, use_norm_z_score: bool = False, is_causal: bool = True, profile_dict: dict = dict()) -> Tuple[Tensor, Tensor]:
    """
    Generate a block sparse layout and mask with same optimized density within each layer(used for ablation experiment). The optimization involves solving a mixed-integer programming problem, aiming to maximize the sum of z-scores for selected blocks while maintaining an average density constraint.
    
    Args:
        - attention_effect: Tensor of shape [num_layer, num_head, num_query, num_key]
        - density_list: List of densities to experiment with
        - avg_density: Average density constraint for the entire layer
        - block_size: Size of each block
        - use_norm_z_score: Flag indicating whether to normalize z-scores by the sum of elements in the matrix
        - is_causal: Flag indicating whether the matrix is causal, i.e., the upper triangular matrix has all elements set to -inf
        - profile_dict: Dictionary to store profiling information. Keys can include 'z_score' and 'opt_density'

    Returns:
        - layout: Tensor of shape [num_layer, num_head, num_query//block_size, num_key//block_size]
        - mask: Tensor of shape [num_layer, num_head, num_query, num_key]
    """
    num_layer, num_head, num_query, num_key = attention_effect.shape

    num_block = num_key // block_size

    attention_effect = torch.abs(attention_effect.detach())
    # mean_attention_matrix = -mean_attention_effect.detach() 

    block_keep_num_ones = torch.ones(num_layer, num_head)

    if is_causal:
        block_keep_num_table = [causal_density_to_topk(num_block, density) for density in density_list]
    else:
        block_keep_num_table = [int(num_block * density) for density in density_list]
    num_density = len(density_list)
    print("the block_keep_num_table is {}".format(block_keep_num_table))

    z_score = torch.empty(num_density, num_layer, num_head)

    for i, keep_num in enumerate(block_keep_num_table):
        _, mask = gen_block_sparse_layout_mask(attention_effect, num_block, num_block, block_keep_num_ones * keep_num, row_balance=True, block_aggregation_func=_block_causal_matrix_mean_scale)
        z_score[i] = matrix_reserve_score(attention_effect, mask, is_causal=True, norm_score=use_norm_z_score)
        print("the average z_score of density {} is {}".format(density_list[i], torch.mean(z_score[i]).item()))

    z_score = torch.permute(z_score, (1, 2, 0)) # shape (num_layers, num_heads, num_density)
    
    z_score = z_score.sum(dim=1).unsqueeze(1) # shape (num_layers, 1, num_density)
    
    density_tensor = torch.tensor(density_list).reshape(1, 1, -1).repeat(num_layer, 1, 1) # shape (num_layers, 1, num_density)

    print("optimizing block nnz ...")
    opt_density = optimize_block_nnz(z_score=z_score, density=density_tensor, avg_density=avg_density)
    opt_topk = opt_density.clone().apply_(lambda x: causal_density_to_topk(num_block, x))
    print("the optimized objective average density is {}".format(torch.mean(opt_density).item()))

    # TODO:
    opt_topk = opt_topk.repeat(1, num_head)
    
    layout, mask = gen_block_sparse_layout_mask(attention_effect, num_block, num_block, opt_topk, row_balance=True, block_aggregation_func=_block_causal_matrix_mean_scale)
    score = matrix_reserve_score(attention_effect, mask, is_causal=True, norm_score=use_norm_z_score)
    
    print("the optimal score is {}".format(torch.mean(score).item()))

    if is_causal:
        # set the attention mask rule for each layer, mask the upper triangle
        causal_mask = torch.zeros((num_query,num_key))
        mask_cond = torch.arange(causal_mask.size(-1))
        causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 1)
        mask = mask * causal_mask
    
    causal_layout = torch.zeros((num_query//block_size, num_key//block_size))
    mask_cond = torch.arange(causal_layout.size(-1))
    causal_layout.masked_fill_(mask_cond < (mask_cond + 1).view(causal_layout.size(-1), 1), 1)

    layout = layout * causal_layout

    if is_causal:
        real_average_density = torch.sum(mask) / (num_layer * num_head * torch.sum(causal_mask))
    else:
        real_average_density = torch.sum(mask) / (num_layer * num_head * num_query * num_key)
    print("the final average density is {}".format(real_average_density.item()))

    mask = mask.cpu().detach().to(bool)
    layout = layout.cpu().detach().to(bool)

    if 'z_score' in profile_dict.keys():
        profile_dict['z_score'] = z_score
    if 'opt_density' in profile_dict.keys():
        profile_dict['opt_density'] = opt_density
    if 'block_keep_num_table' in profile_dict.keys():
        profile_dict['block_keep_num_table'] = block_keep_num_table
    if 'opt_block_keep_num_table' in profile_dict.keys():
        profile_dict['opt_block_keep_num_table'] = opt_topk
    if 'opt_z_score' in profile_dict.keys():
        # show the position of opt_block_keep_num in block_keep_num_table
        opt_choice = torch.zeros(num_layer, num_head, dtype=torch.long)
        for i, keep_num in enumerate(block_keep_num_table):
            index = torch.where(opt_topk == keep_num)
            opt_choice[index] = i
        opt_z_score = torch.gather(z_score, dim=-1, index=opt_choice.unsqueeze(-1)).squeeze(-1)
        profile_dict['opt_z_score'] = opt_z_score

    return layout, mask


if __name__ == "__main__":
    # use argparse to get the arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention_effect_path", type=str)
    parser.add_argument("--avg_density", type=float)
    parser.add_argument("--output_path", type=str)

    parser.add_argument("--block_size", type=int, default=64)
    
    parser.add_argument("--ablation_layer", action="store_true", help="same density for all layers")
    parser.add_argument("--ablation_head", action="store_true", help="same density for all heads in one layer")
    # parser.add_argument("--density_list", type=str)
    density_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 

    args = parser.parse_args()

    attention_effect = torch.load(args.attention_effect_path)

    layout, mask = mip_gen_block_sparse_layout_mask(attention_effect, density_list, args.avg_density, block_size=args.block_size, use_norm_z_score=True, is_causal=True)

    torch.save(layout, args.output_path + "/layout.pt")
    torch.save(mask, args.output_path + "/mask.pt")
