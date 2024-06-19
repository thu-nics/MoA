from transformers import OPTForCausalLM, OPTConfig
from transformers import LlamaConfig, LlamaForCausalLM

def load_config(model_name):
    if "opt" in model_name:
        return OPTConfig.from_pretrained(model_name)
    elif "llama" in model_name:
        return LlamaConfig.from_pretrained(model_name)
    elif "vicuna" in model_name:
        return LlamaConfig.from_pretrained(model_name)
    elif "longchat" in model_name:
        return LlamaConfig.from_pretrained(model_name)
    else:
        raise NotImplementedError("Model {} not implemented".format(model_name))

def load_causal_model(model_name, config=None, ignore_mismatched_sizes=False, **kwargs):
    if config is None:
        config = load_config(model_name)
    if "opt" in model_name:
        return OPTForCausalLM.from_pretrained(model_name, config=config, ignore_mismatched_sizes=ignore_mismatched_sizes, **kwargs)
    elif "llama" in model_name:
        return LlamaForCausalLM.from_pretrained(model_name, config=config, ignore_mismatched_sizes=ignore_mismatched_sizes, **kwargs)
    elif "vicuna" in model_name:
        return LlamaForCausalLM.from_pretrained(model_name, config=config, ignore_mismatched_sizes=ignore_mismatched_sizes, **kwargs)
    elif "longchat" in model_name:
        return LlamaForCausalLM.from_pretrained(model_name, config=config, ignore_mismatched_sizes=ignore_mismatched_sizes, **kwargs)
    else:
        raise NotImplementedError("Model {} not implemented".format(model_name))
