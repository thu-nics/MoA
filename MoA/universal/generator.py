import torch 
from torch import Tensor
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any

"""
Given the importance tensor of shape [keep_dim, N, M],
Return three tensors: plan_config, accuracy-loss, latency-cost. Accuracy-loss, latency-cost are of shape [keep_dim, num_plan_config]
"""

class UniversalGenerator:
    def __init__(self) -> None:
        pass

    def importance_to_accuracy_loss_and_lantency(self, importance_tensor: Tensor, runtime_config: Optional[Dict] = None, **kwargs) -> Tuple[Tensor, Tensor, Tensor, Dict]:
        """
        Hardware aware.
        Input:
            importance tensor: Tensor of shape [keep_dim, N, M]
            runtime_config: Dict of runtime config.
        Output:
            accuracy_loss: Tensor of shape [keep_dim, num_plan_config], indicating the accuracy loss of each plan config
            latency_cost: Tensor of shape [keep_dim, num_plan_config]
            config_id: Tensor of shape [keep_dim, num_plan_config], indicating the config id of each plan
            config: Dict of config. 
        """
        # function example
        accuracy_loss, config_id, config = self.importance_to_accuracy_loss(importance_tensor, **kwargs)
        latency_cost = self.plan_config_to_latency(config_id, config, runtime_config, **kwargs)
        return accuracy_loss, latency_cost, config_id, config

    def importance_to_accuracy_loss(self, importance_tensor: Tensor, **kwargs) -> Tuple[Tensor, Tensor, Dict]:
        """
        Hardware agnostic.
        Input:
            importance tensor: Tensor of shape [keep_dim, N, M]
        Output:
            accuracy_loss: Tensor of shape [keep_dim, num_plan_config], indicating the accuracy loss of each plan config
            config_id: Tensor of shape [keep_dim, num_plan_config], indicating the config id of each plan
            config: Dict of config.
        """
        raise DeprecationWarning

    def plan_config_to_latency(self, config_id: Tensor, config: Optional[Dict] = None, runtime_config: Optional[Dict] = None, **kwargs) -> Tensor:
        """
        Hardware aware.
        Input:
            config_id: Tensor of shape [keep_dim, num_plan_config], indicating the config id of each plan
            config: Dict of config.
            runtime_config: Dict of runtime config.
        Output:
            
        """
        raise DeprecationWarning

    def config_id_to_plan(self, config_id: Tensor, config: Optional[Dict] = None, importance_tensor: Optional[Tensor] = None, **kwargs) -> Any:
        """
        Hardware agnostic.
        Input:
            config_id: Tensor of shape [keep_dim, 1], indicating the config id of each plan
            config: Dict of config.
            importance_tensor: Tensor of shape [keep_dim, N, M]
        Output:
            plan: Anything that can be indexed by keep_dim
        """
        pass

    def generate(self, importance: Tensor, runtime_config: Dict, **kwargs) -> Dict:
        """
        Input:
            importance tensor: Tensor of shape [keep_dim, N, M]
            runtime_config: Dict of runtime config.
        Output:
            generation_output: dict including the following keys:
                accuracy_loss: Tensor of shape [keep_dim, num_plan_config], indicating the accuracy loss of each plan config
                latency_cost: Tensor of shape [keep_dim, num_plan_config], indicating the latency cost of each plan config
                config_id: Tensor of shape [keep_dim, num_plan_config], indicating the config id of each plan
                config: Dict of config.
        """
        accuracy_loss, latency_cost, config_id, config = self.importance_to_accuracy_loss_and_lantency(importance, runtime_config, **kwargs)
        primary_config = self.config_id_to_primary_config(config_id, config, **kwargs)

        return {'accuracy_loss': accuracy_loss, 'latency_cost': latency_cost, 'config_id': config_id, 'config': config, 'primary_config': primary_config}

    def get_summary(self) -> Any:
        """
        Return the summary of the generator, e.g., DataFrame of config, accuracy_loss, and latency_cost.
        The summary is also printed in this function.
        """
        if hasattr(self, 'summary_table'):
            if type(self.summary_table) == pd.DataFrame:
                print(self.summary_table.round(2))
            return self.summary_table
        else:
            return None

    def config_id_to_primary_config(self, config_id: Tensor, config: Optional[Dict] = None, **kwargs) -> float:
        """
        Hardware agnostic.
        Input:
            config_id: Tensor of shape [keep_dim, num_plan_config], indicating the config id of each plan
            config: Dict of config.
        Output:
            primary_config: a number that can be indexed by keep_dim, indicating the primary value that represent each plan
        """
        return torch.zeros_like(config_id)
    


class ElasticGenerator:
    def __init__(self, hardware) -> None:
        """
        config: Dict of config
        config_id: Tensor of shape [keep_dim, num_plan_config], indicating the config id of each plan
        """
        self.hardware = hardware
        pass

    def prepare_config(self, **kwargs) -> None:
        """
        Prepare the config of the generator
        """
        self.config = dict()
        self.config_id: Optional[Tensor] = None

    def importance_to_accuracy_loss(self, importance_tensor: Tensor, shape: Tuple[int, int, int], **kwargs) -> Tensor:
        """
        Hardware agnostic.
        Input:
            importance_tensor: Tensor of shape [keep_dim, N, M]
            shape: Tuple of shape of the tensor to be processed and the output
        Output:
            accuracy_loss: Tensor of shape [keep_dim, num_plan_config], indicating the accuracy loss of each plan config
        """
        raise DeprecationWarning

    def plan_config_to_latency(self, shape: Tuple[int, int, int], runtime_config: Optional[Dict] = None, **kwargs) -> Tensor:
        """
        Hardware aware.
        Input:
            shape: Tuple of shape of the tensor to be processed and the output
        Output:
            latency_cost: Tensor of shape [keep_dim, num_plan_config], indicating the latency cost of each plan config
        """
        raise DeprecationWarning

    def primary_config(self, config_id: Optional[Tensor] = None, **kwargs) -> float:
        """
        Hardware agnostic.
        Input:
            config_id: Tensor of shape [keep_dim, num_plan_config], indicating the config id of each plan
        Output:
            primary_config: a number that can be indexed by keep_dim, indicating the primary value that represent each plan
        """
        raise NotImplementedError
        return torch.zeros_like(self.config_id)

    def config_id_to_plan(self, config_id: Tensor, shape: Tuple[int, int, int], **kwargs) -> Any:
        """
        Hardware agnostic.
        Input:
            config_id: Tensor of shape [keep_dim, 1], indicating the config id of each plan
        Output:
            plan: Anything that can be indexed by keep_dim
        """
        pass

    def get_summary(self) -> Any:
        """
        Return the summary of the generator, e.g., DataFrame of config, accuracy_loss, and latency_cost.
        The summary is also printed in this function.
        """
        if hasattr(self, 'summary_table'):
            if type(self.summary_table) == pd.DataFrame:
                print(self.summary_table.round(2))
            return self.summary_table
        else:
            return None

    def generate(self, importance: Tensor, runtime_config: Dict, **kwargs) -> Dict:
        """
        Input:
            importance tensor: Tensor of shape [keep_dim, N, M]
            runtime_config: Dict of runtime config.
        Output:
            generation_output: dict including the following keys:
                accuracy_loss: Tensor of shape [keep_dim, num_plan_config], indicating the accuracy loss of each plan config
                latency_cost: Tensor of shape [keep_dim, num_plan_config], indicating the latency cost of each plan config
                config_id: Tensor of shape [keep_dim, num_plan_config], indicating the config id of each plan
                config: Dict of config.
        """
        accuracy_loss = self.importance_to_accuracy_loss(importance_tensor=importance, **kwargs)
        latency_cost = self.plan_config_to_latency(runtime_config=runtime_config, **kwargs)
        
        primary_config = self.config_id_to_primary_config(config_id=self.config_id, **kwargs)

        return {'accuracy_loss': accuracy_loss, 'latency_cost': latency_cost, 'config_id': self.config_id, 'config': self.config, 'primary_config': primary_config}

