from typing import Dict

class Hardware:
    def __init__(self, hardware_name, model_name, config_dict: Dict):
        self.runtime_df = None
        
        self.hardware_name = hardware_name
        self.model_name = model_name
        
        # convert all items in config_dict as member parameters
        for key in config_dict:
            setattr(self, key, config_dict[key])
    
        # further initialization
        self.init_model_members()
    
    def init_model_members(self):
        pass
    
    def update_runtime_df(self):
        """
        return a runtime df with columns:
        """
        pass
    
    def get_attention_time(self):
        return 0
    
    def get_ffn_time(self):
        return 0
    
    def get_QKV_time(self):
        return 0
    
    def get_latency_bound(self, latency_bound_ratio) -> float:
        return 0.0
    