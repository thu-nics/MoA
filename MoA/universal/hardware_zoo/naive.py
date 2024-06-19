from torch import Tensor
from MoA.universal.hardware import Hardware
from typing import Dict
import pandas as pd
from typing import Tuple

class NaiveHardware(Hardware):
    def __init__(self):
        self.runtime_df = None    
        self.hardware_name = 'NaiveHardware'
    
    def get_latency_bound(self, avg_latency_ratio, shape: Tuple[int, int]) -> float:
        """
        Input:
            shape: Tuple of shape of the tensor to be processed and the output
        """
        num_query, num_key = shape[-2:]
        total_time = num_query * num_key

        return total_time * avg_latency_ratio

    def get_attention_time(self, density_tensor: Tensor, shape: Tuple[int, int]):
        """
        Input:
            density_tensor: Tensor of shape [keep_dim, num_plan_config]
            shape: Tuple of shape of the tensor to be processed and the output
        """
        num_query, num_key = shape[-2:]
        total_time = num_query * num_key

        return total_time * density_tensor
        

class NaiveKVHardware(Hardware):

    def __init__(self):
        self.runtime_df = None    
        self.hardware_name = 'NaiveHardware'

    def get_latency_bound(self, avg_latency_ratio, shape: Tuple[int, int]) -> float:
        """
        Input:
            shape: Tuple of shape of the tensor to be processed and the output
        """
        _, num_key = shape[-2:]
        total_time = num_key

        return total_time * avg_latency_ratio
    
    def get_attention_time(self, density_tensor: Tensor, shape: Tuple[int, int]):
        """
        Input:
            density_tensor: Tensor of shape [keep_dim, num_plan_config]
            shape: Tuple of shape of the tensor to be processed and the output
        """
        _, num_key = shape[-2:]
        total_time = num_key

        return total_time * density_tensor