import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import cvxpy as cp
import argparse

from MoA.universal.elastic_block_sparse import ElasticBlockSparse

from MoA.universal.hardware_zoo.naive import NaiveKVHardware
from MoA.universal.optimizer import ElasticOptimizer
from MoA.attention.convert import layout_to_lut_single_density
from MoA.attention.pattern import gen_causal_pattern

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, required=True, help='output directory')
parser.add_argument('--elastic_length', type=int, nargs='+', default=[1024, 2048, 3072, 4096], help='elastic length')
parser.add_argument('--extend_length', type=int, nargs='+', default=[8192, 16384], help='extend length')
parser.add_argument('--density_bounds', type=float, nargs='+', default=[1.00, 0.77, 0.58, 0.46, 0.25, 0.13], help='density bounds')
parser.add_argument('--importance_tensor_dir', type=str, required=True, help='importance tensor directory')
parser.add_argument('--output_length', type=int, nargs='+', default=[2048, 4096, 8192, 16384], help='output length')
parser.add_argument('--num_plan_limit', type=int, default=None, help='number of plans to construct')
parser.add_argument('--num_alphas', type=int, default=None, help='number of alphas')
parser.add_argument('--alpha', type=float, default=None, help='for uniform extend')
parser.add_argument('--alpha_interval', type=int, default=1024, help='interval of alpha')
parser.add_argument('--num_betas', type=int, default=9, help='number of alphas')
parser.add_argument('--beta', type=float, default=None, help="for uniform extend")
parser.add_argument('--latency_lower_bound_ratio', type=float, default=None, help='the ratio of the lower bound for latency optimization') 

parser.add_argument('--device', type=str, default='cpu', help='device to run the code')
parser.add_argument('--same_per_layer', action='store_true', help='whether to sum the importance tensor per layer')

parser.add_argument('--block_size', type=int, default=64)
parser.add_argument('--aggregating_block_size', type=int, default=1, help='aggregating block size for importance tensor')

parser.add_argument('--normalize_by_head', action='store_true', help='whether to make loss sum on every head the same')
parser.add_argument('--normalize_by_layer', action='store_true', help='whether to make the loss sum on every layer the same')

parser.add_argument('--num_key_value_groups', type=int, default=1, help="for group query")

parser.add_argument('--save_layout', action='store_true', help='whether to save layout')

parser.add_argument('--time_limit', type=int, default=None)

args = parser.parse_args()

print(args)

"""
prepare the hardware and block sparse generator
"""
block_size = args.block_size
aggregating_block_size = args.aggregating_block_size
hardware = NaiveKVHardware()

output_dir = args.output_dir

os.makedirs(output_dir, exist_ok=True)

same_per_layer = args.same_per_layer
print(f"same_per_layer: {same_per_layer}")

block_sparse_generator = ElasticBlockSparse(
    hardware=hardware,
    block_size=block_size,
    aggregating_block_size=aggregating_block_size,
    use_norm_z_score=False, 
    is_causal=True,
    device=args.device,
    accuracy_loss_ratio=1.0
)

elastic_length = args.elastic_length
extend_length = args.extend_length
density_bounds = args.density_bounds
output_length = args.output_length

max_profile_length=max(elastic_length)
max_token_length=max(elastic_length + extend_length + output_length)
min_token_length=min(elastic_length + extend_length + output_length)

num_alphas=args.num_alphas
num_betas=args.num_betas

if num_alphas is None:
    num_alphas = (max_profile_length + min_token_length) // args.alpha_interval + 1

alpha = torch.linspace(-min_token_length, max_profile_length, num_alphas)
beta = torch.linspace(0.0, 1.0, num_betas)

if args.alpha is not None:
    alpha = torch.tensor([args.alpha])
    print(f'user provided single alpha: {alpha}')

print(f"alpha: {alpha}")
print(f"beta: {beta}")

block_sparse_generator.prepare_config(alpha=alpha, beta=beta, max_profile_length=max_profile_length, max_token_length=max_token_length, min_token_length=min_token_length)
for k,v in block_sparse_generator.config.items():
    print(k)
    print(v)

num_elastic_length = len(elastic_length)
    
elastic_importance_loss = []
elastic_latency_cost = []

# load
importance_tensor_dir = args.importance_tensor_dir

importance_tensor_name_dict = {
    1024: 'profile_1k',
    2048: 'profile_2k',
    3072: 'profile_3k',
    4096: 'profile_4k',
    5120: 'profile_5k',
    6144: 'profile_6k',
    7168: 'profile_7k',
    8192: 'profile_8k',
    9216: 'profile_9k',
}

latency_bound_token_length = elastic_length+extend_length
density_bound_dict = {token_length: density for token_length, density in zip(latency_bound_token_length, density_bounds)}

# remove the duplicated key and value pairs
latency_bound_token_length = list(set(latency_bound_token_length))
density_bound_dict = {token_length:density_bound_dict[token_length] for token_length in latency_bound_token_length}

print("density_bound_dict")
print(density_bound_dict)

"""
Generate plans for different token length
"""
for token_length in tqdm(elastic_length):
    importance_tensor_path = os.path.join(importance_tensor_dir, importance_tensor_name_dict[token_length], f'grad_attn_tensor_{token_length}.pt')
    importance_tensor = torch.load(importance_tensor_path).to(torch.float32).abs()

    if args.num_key_value_groups > 1:
        importance_tensor = importance_tensor.reshape(importance_tensor.shape[0], args.num_key_value_groups, -1, importance_tensor.shape[-2], importance_tensor.shape[-1])
        importance_tensor = importance_tensor.max(dim=1, keepdim=False).values

    if args.normalize_by_head:
        sums = importance_tensor.sum(dim=(-1, -2), keepdim=True)
        importance_tensor = importance_tensor / sums
    if args.normalize_by_layer:
        sums = importance_tensor.sum(dim=(-1, -2, -3), keepdim=True)
        importance_tensor = importance_tensor / sums

    profile_num_layer, profile_num_head, profile_inner_dim, _ = importance_tensor.shape
    print(importance_tensor.shape)
    if same_per_layer:
        importance_tensor = importance_tensor.sum(dim=1, keepdim=False) # shape: [profile_num_layer, profile_inner_dim, profile_inner_dim]
    else:
        importance_tensor = importance_tensor.reshape(profile_num_layer * profile_num_head, profile_inner_dim, profile_inner_dim)
    keep_dim = importance_tensor.shape[0]

    importance_loss = block_sparse_generator.importance_to_accuracy_loss(importance_tensor=importance_tensor, shape=(keep_dim, token_length, token_length))
    elastic_importance_loss.append(importance_loss)
elastic_importance_loss = torch.stack(elastic_importance_loss, dim=0) # shape: [elastic_length, keep_dim, num_plan_config]

for token_length in latency_bound_token_length:
    latency_cost = block_sparse_generator.plan_config_to_latency(shape=(keep_dim, token_length, token_length))
    elastic_latency_cost.append(latency_cost)

elastic_latency_cost = torch.stack(elastic_latency_cost, dim=0) # shape: [elastic_length + extend_length, keep_dim, num_plan_config]

"""
Get the latency bounds
"""
latency_bounds = []
for token_length, density in density_bound_dict.items():
    latency_bound = hardware.get_latency_bound(avg_latency_ratio=density, shape=(token_length, token_length)) * keep_dim
    latency_bounds.append(latency_bound)
print("latency_bounds by each dim")
print(torch.tensor(latency_bounds) / keep_dim)

sum_elastic_latency_cost = torch.sum(elastic_latency_cost, dim=1)
sum_elastic_importance_loss = torch.sum(elastic_importance_loss, dim=1)

print("sum_elastic_latency_cost")
print(sum_elastic_latency_cost)
print("sum_elastic_importance_tensor")
print(sum_elastic_importance_loss)

"""
Optimization
"""

def elastic_optimize(elastic_importance_loss, elastic_latency_cost, latency_bounds):
    """
    Optimize the elastic importance tensor
    Input:
        elastic_importance_loss: torch.tensor, shape: [elastic_length, keep_dim, num_plan_config]
        elastic_latency_cost: torch.tensor, shape: [elastic_length + extend_length, keep_dim, num_plan_config]
        latency_bounds: list of torch.tensor, shape: [elastic_length + extend_length]
    Output:
        optimize_config_ids: torch.tensor, shape: [num_plan, keep_dim]
    """

    solver_kwargs = {
        # can set time limit to make the process faster
        # "time_limit": 300 * 60,
    }

    if args.time_limit is not None:
        solver_kwargs['time_limit'] = args.time_limit * 60
    
    optimizer = ElasticOptimizer(**solver_kwargs)

    if args.num_plan_limit is not None:
        optimizer.set_construct_num_plan_limit(1, profile_num_layer, profile_num_head, args.num_plan_limit)

    # normalize_factor = torch.sum(elastic_importance_loss, dim=(-1,-2)).reshape(num_elastic_length, 1, 1)
    normalize_factor = torch.ones((num_elastic_length, 1, 1), dtype=elastic_importance_loss.dtype)
    normed_elastic_importance_loss = elastic_importance_loss / normalize_factor

    opt_cofig_index_list = optimizer.optimize(accuracy_loss_list=[normed_elastic_importance_loss], latency_cost_list=[elastic_latency_cost], latency_bound=torch.Tensor(latency_bounds), latency_lower_bound = torch.Tensor(latency_bounds) * args.latency_lower_bound_ratio if args.latency_lower_bound_ratio is not None else None)

    optimize_config_ids = opt_cofig_index_list[0] # shape [num_plan, keep_dim]. If is is single objective, then shape [1, keep_dim]; if it is multi-objective, then shape [num_objective, keep_dim]
    print(optimize_config_ids)

    return optimize_config_ids

optimize_config_ids = elastic_optimize(elastic_importance_loss, elastic_latency_cost, latency_bounds) # shape [num_plan, keep_dim]

num_plan = optimize_config_ids.shape[0]
print(f"num_plan: {num_plan}")

"""
Output plans
"""
def elastic_output_plan(block_sparse_generator, optimize_config_ids, output_length, num_plan):
    # chosen plans
    opt_alpha = torch.tensor(block_sparse_generator.config['alpha'])[optimize_config_ids].reshape(num_plan, profile_num_layer, -1)
    print('average opt_alpha for each layer')
    print(torch.mean(opt_alpha, dim=-1))
    opt_beta = torch.tensor(block_sparse_generator.config['beta'])[optimize_config_ids].reshape(num_plan, profile_num_layer, -1)
    print('average opt_beta for each layer')
    print(torch.mean(opt_beta, dim=-1))
    print('max opt_beta for each layer')
    print(torch.max(opt_beta, dim=-1).values)

    # save the opt_alpha and beta to csv by plan
    def convert_to_df(alpha, beta) -> pd.DataFrame:
        # Create index arrays for each dimension
        plan_id, layer_id, head_id = np.indices(alpha.shape)
        
        # Flatten the indices and tensor values
        data = {
            'plan_id': plan_id.flatten(),
            'layer_id': layer_id.flatten(),
            'head_id': head_id.flatten(),
            'alpha_value': alpha.flatten(),
            'beta_value': beta.flatten()
        }
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)

        return df

    convert_to_df(opt_alpha.numpy(), opt_beta.numpy()).to_csv(os.path.join(output_dir, 'opt_alpha_beta.csv'))

    opt_importance_loss = elastic_importance_loss.gather(-1, optimize_config_ids.T.view(1, -1, num_plan).expand(num_elastic_length, -1, -1)).permute(2, 0, 1) # shape [num_plan, num_elastic_length, keep_dim]
    print('opt_importance_loss')
    print(torch.sum(opt_importance_loss, dim=(-1)))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save the importance loss of each plan to csv
    pd.DataFrame(torch.sum(opt_importance_loss, dim=(-1)).numpy()).to_csv(os.path.join(output_dir, 'opt_importance_loss.csv'))

    for plan_id in range(num_plan):
        # output the plan at all length
        for token_length in output_length:
            print(f"Token Length: {token_length}")
            num_block = token_length // block_size
            optimize_config_id = optimize_config_ids[plan_id].reshape(-1,1)
            layout = block_sparse_generator.config_id_to_plan(optimize_config_id, shape=(token_length, token_length))
            if same_per_layer:
                layout = layout.reshape(profile_num_layer, 1, num_block, num_block).expand(-1, profile_num_head, -1, -1)
            else:
                layout = layout.reshape(profile_num_layer, profile_num_head, num_block, num_block)
            # print density
            causal_layout = gen_causal_pattern(num_block, num_block, dtype=torch.bool)

            density: float = block_sparse_generator._calculate_density(layout)
            print(f"Average Density {density}")

            # save as file
            # convert to lut and save
            lut = layout_to_lut_single_density(layout)
            if args.save_layout:
                layout_path = os.path.join(output_dir, f'layout_{token_length}_plan_{plan_id}.pt')
                print(f"Save layout to {layout_path}")
                torch.save(layout, layout_path)
            if args.num_key_value_groups > 1:
                lut = [layer_lut.repeat_interleave(args.num_key_value_groups, dim=0) for layer_lut in lut]
            lut_path = os.path.join(output_dir, f'lut_{token_length}_plan_{plan_id}.pt')
            print(f"Save lut to {lut_path}")
            torch.save(lut, lut_path)
    print("done")

output_length = args.output_length
elastic_output_plan(block_sparse_generator, optimize_config_ids, output_length, num_plan)