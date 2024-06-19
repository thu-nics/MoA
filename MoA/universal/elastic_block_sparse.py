import torch
from torch import Tensor
from typing import Tuple, Dict, Optional, Union
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import warnings

from MoA.attention.sparse_mask import matrix_reserve_score
from MoA.universal.generator import ElasticGenerator
from MoA.attention.pattern import gen_bigbird_pattern, gen_causal_pattern
from MoA.universal.hardware import Hardware

class ElasticBlockSparse(ElasticGenerator):
    def __init__(self, hardware: Hardware, block_size: int = 64, aggregating_block_size: int = 1, use_norm_z_score: bool = False, is_causal: bool = True, device: str = 'cpu', accuracy_loss_ratio: float = 1.0) -> None:
        """
        BlockSparse is a universal generator for block sparse attention.
        Input:
            hardware: Hardware, the hardware to run the model
            block_size: int, the block size
            use_norm_z_score: bool, whether to use z-score to normalize the importance score
            is_causal: bool, whether the attention is causal
            device: str, the device to run the model
            accuracy_loss_ratio: float, the ratio of accuracy loss
        """
        self.hardware = hardware
        self.block_size = block_size
        self.aggregating_block_size = aggregating_block_size
        self.use_norm_z_score = use_norm_z_score
        self.is_causal = is_causal
        assert is_causal, "Only causal attention is supported now because of the density calculation."
        self.device = device
        self.accuracy_loss_ratio = accuracy_loss_ratio

        self.summary_table: pd.DataFrame = None

        assert aggregating_block_size == 1 or block_size == aggregating_block_size

    def prepare_config(self, alpha: Tensor = None, beta: Tensor = None, max_profile_length: int = 4096, max_token_length: int = 16384, min_token_length: int = 1024) -> None:
        """
        width = alpha + beta * length
        """
        alpha = alpha.to(self.device)
        beta = beta.to(self.device)

        alpha, beta = torch.meshgrid(alpha, beta)
        alpha = alpha.reshape(-1)
        beta = beta.reshape(-1)

        # eliminate the invalid config
        index = (alpha + beta * max_profile_length <= max_profile_length) & (alpha + beta * max_token_length <= max_token_length) & (alpha + beta * min_token_length > 0)
        # index = (alpha + beta * max_token_length <= max_token_length) & (alpha + beta * min_token_length >= 0)
        alpha = alpha[index]
        beta = beta[index]

        # sort based on alpha + beta * max_token_length
        _, index = torch.sort(alpha + beta * max_token_length, stable=True)
        alpha = alpha[index]
        beta = beta[index]

        config = dict() # each value of the config is a tensor of shape num_plan_config
        config['alpha'] = alpha.tolist()
        config['beta'] = beta.tolist()

        num_plan_config = len(alpha)
        config_id = torch.arange(num_plan_config) # shape: (num_plan_config,)

        # reserve for later computation
        config['density'] = dict() # (num_query, num_key): density

        self.config = config
        self.num_plan_config = num_plan_config
        self.config_id = config_id

    def layout_generate_function(self, shape: Union[Tuple[int, int], Tuple[int, int, int]], config_id: int) -> Tensor:
        """
        Generate the layout tensor of shape [num_block_x, num_block_y]
        """
        alpha = self.config['alpha'][config_id]
        beta = self.config['beta'][config_id]

        num_block_x = shape[-2] // self.block_size
        num_block_y = shape[-1] // self.block_size

        assert num_block_x == num_block_y

        num_band_block = int(alpha // self.block_size + beta * num_block_x)
        num_band_block = max(2*num_band_block - 1, 1) # make sure it is odd
        num_global_block = 1 # warning: this is a constant, noqa
        num_rand_block = 0

        bigbird_layout = gen_bigbird_pattern(
            num_block_x, num_block_y, 
            num_band_block, num_global_block, num_rand_block, 
            num_layer=1, num_head=1, dtype=torch.bool,
        ).squeeze().squeeze().to(self.device) # shape: (1, 1, num_block_x, num_block_y)

        return bigbird_layout


    def importance_to_accuracy_loss(self, importance_tensor: Tensor, shape: Union[Tuple[int, int], Tuple[int, int, int]], **kwargs) -> Tensor:
        """
        Hardware agnostic.
        Input:
            importance_tensor: Tensor of shape [keep_dim, N, M]
            shape: Tuple of shape of the tensor to be processed and the output
        Output:
            accuracy_loss: Tensor of shape [keep_dim, num_plan_config], indicating the accuracy loss of each plan config
        """
        importance_tensor = importance_tensor[:, :shape[-2] // self.aggregating_block_size, :shape[-1] // self.aggregating_block_size].to(self.device)
        keep_dim, inner_dim1, inner_dim2 = importance_tensor.shape

        if self.aggregating_block_size == 1:
            num_block_x = inner_dim1 // self.block_size
            num_block_y = inner_dim2 // self.block_size
        else:
            num_block_x = inner_dim1
            num_block_y = inner_dim2

        is_causal = self.is_causal
        if is_causal:
            assert num_block_x == num_block_y
        
        num_plan_config = self.num_plan_config

        z_loss = torch.empty(num_plan_config, keep_dim)

        # prepare density calculation
        density_list = []
        causal_layout = gen_causal_pattern(num_block_x, num_block_y, dtype=torch.bool).squeeze().to(self.device) # shape: (1, num_block_x, num_block_y)

        if is_causal and self.aggregating_block_size == 1:
            causal_mask = gen_causal_pattern(inner_dim1, inner_dim2, dtype=torch.bool).squeeze().to(self.device) # shape: (1, 1, num_query, num_key)
            importance_tensor = importance_tensor * causal_mask

        for i in range(num_plan_config):
            # generate bigbird attention mask
            layout = self.layout_generate_function(shape[-2:], i).to(self.device) # shape: (num_block_x, num_block_y)
            layout = layout * causal_layout if is_causal else layout

            # calculate density
            density = self._calculate_density(layout)
            density_list.append(density)

            # calculate loss
            if self.aggregating_block_size == 1:
                layout_importance_tensor = F.avg_pool2d(importance_tensor, self.block_size) * self.block_size * self.block_size # shape: (keep_dim, num_block_x, num_block_y), each element is the sum of the importance score in the block
            else:
                layout_importance_tensor = importance_tensor

            # compute z_loss
            z_loss[i] = matrix_reserve_score(layout_importance_tensor, ~layout.unsqueeze(0).repeat(keep_dim, 1, 1), is_causal=is_causal, norm_score=self.use_norm_z_score) # use the reverse of mask to calculate loss instead of score
        
        # warning if inf or nan exists
        if torch.isinf(z_loss).any() or torch.isnan(z_loss).any():
            warnings.warn('Warning: z_loss has inf or nan')

        self.config['density'][(inner_dim1, inner_dim2)] = density_list
        

        # mask = torch.permute(mask, (1, 0, 2, 3)) # shape (keep_dim, num_config, num_query, num_key)
        z_loss = torch.permute(z_loss, (1, 0)) # shape (keep_dim, num_config)

        z_loss = z_loss * self.accuracy_loss_ratio

        sum_accuracy_loss = torch.sum(z_loss, dim=0).to(float).tolist()
    
        # use pandas to print the table of config and z_loss
        relavent_config = {k:v for k,v in self.config.items() if len(v)==num_plan_config}
        relavent_config.update({
            'density': density_list,
            'accuracy_loss' : sum_accuracy_loss,
            'shape' : [shape[-1]] * num_plan_config,
        })

        df = pd.DataFrame(relavent_config)
        self.summary_table = df

        return z_loss
        
    def plan_config_to_latency(self, shape: Tuple[int, int, int], runtime_config: Optional[Dict] = None, **kwargs) -> Tensor:
        """
        Hardware aware.
        Input:
            shape: Tuple of shape of the tensor to be processed and the output
        Output:
            latency_cost: Tensor of shape [keep_dim, num_plan_config], indicating the latency cost of each plan config
        """
        keep_dim, num_query, num_key = shape

        # check if the latency is already calculated
        if (num_query, num_key) in self.config['density'].keys():
            density_list = self.config['density'][(num_query, num_key)]
        else:
            density_list = []
            num_block_x = shape[-2] // self.block_size
            num_block_y = shape[-1] // self.block_size
            causal_layout = gen_causal_pattern(num_block_x, num_block_y, dtype=torch.bool).squeeze().to(self.device) # shape: (1, num_block_x, num_block_y)
            for i in range(self.num_plan_config):
                # generate bigbird attention mask
                layout = self.layout_generate_function(shape[-2:], i).to(self.device) # shape: (num_block_x, num_block_y)
                layout = layout * causal_layout if self.is_causal else layout

                # calculate density
                density = self._calculate_density(layout)
                density_list.append(density)
            
        density_tensor = torch.tensor(density_list).repeat(keep_dim, 1)

        latency = self.hardware.get_attention_time(density_tensor, shape=shape) # should include shape

        # add latency to summary table
        sum_latency = torch.sum(latency, dim=0).to(float).tolist()

        self.summary_table['latency'] = sum_latency
        return latency

    def primary_config(self, config_id: Optional[Tensor] = None, **kwargs) -> float:
        """
        Hardware agnostic.
        Input:
            config_id: Tensor of shape [keep_dim, num_plan_config], indicating the config id of each plan
            config: Dict of config.
        Output:
            primary_config: density of shape [keep_dim, num_plan_config], indicating the density of each plan
        """
        if config_id is None:
            config_id = self.config_id
        return torch.tensor(self.config['density'])[config_id]
    
    def config_id_to_plan(self, config_id: Tensor, shape: Union[Tuple[int, int], Tuple[int, int, int]], **kwargs) -> Tensor:
        """
        Hardware agnostic.
        Input:
            config_id: Tensor of shape [keep_dim, 1], indicating the config id of each plan
        Output:
            plan: Layout indexed by keep_dim
        """

        num_query, num_key = shape[-2:]
        keep_dim = config_id.shape[0]

        num_block_x = num_query // self.block_size
        num_block_y = num_key // self.block_size

        is_causal = self.is_causal and (num_block_x==num_block_y)
        # print("is_causal: {}".format(is_causal))

        layout_output = []
        for id in config_id:
            layout = self.layout_generate_function(shape[-2:], id)
            layout = layout_output.append(layout)

        layout_output = torch.stack(layout_output, dim=0)

        if is_causal:
            causal_layout = gen_causal_pattern(num_block_x, num_block_y, dtype=torch.bool).squeeze().to(self.device) # shape: (1, num_block_x, num_block_y)
            layout_output = layout_output * causal_layout

        return layout_output
    
    @staticmethod
    def _calculate_density(layout: Tensor, mode="kv") -> float:
        """
        Calculate the density of the layout.
        Input:
            layout: Tensor of shape [..., num_block_x, num_block_y]
        """
        # calculate density
        num_block_x = layout.shape[-2]
        num_block_y = layout.shape[-1]

        keep_dim = layout.view(-1, num_block_x, num_block_y).shape[0]

        # full attention
        if mode == "full":
            density = torch.sum(layout).to(float) / (num_block_x * num_block_y) / keep_dim 
        # causal attention
        elif mode == "causal":
            density = torch.sum(layout).to(float) / ( (1+num_block_y-num_block_x+num_block_y)*num_block_x/2 ) / keep_dim
        # kv attention
        elif mode == "kv":
            density = torch.sum(torch.max(torch.sum(layout, dim=-1), dim=-1).values).to(float) / num_block_y / keep_dim
        else:
            raise NotImplementedError

        return density.item()
    
