import torch

def update_model_function(model, model_name):
    """
    Update the model function based on the model name
    """
    if (
        ("llama" in model_name)
        or ("Llama" in model_name)
        or ("vicuna" in model_name)
        or ("longchat" in model_name)
    ):
        print("Update functions from `modeling_llama` in `MoA`")

        # lut
        from MoA.models.llama.modeling_llama import (
            LlamaModel_use_block_sparse_attention_lut, LlamaModel_set_mixture_of_attention
        )

        model.model.use_block_sparse_attention_lut = (
            LlamaModel_use_block_sparse_attention_lut.__get__(model.model)
        )
        model.model.set_mixture_of_attention = (
            LlamaModel_set_mixture_of_attention.__get__(model.model)
        )
    else:
        raise NotImplementedError("Model {} not implemented".format(model_name))

    return model