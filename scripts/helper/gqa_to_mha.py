from transformers import LlamaModel, AutoTokenizer, LlamaForCausalLM, AutoConfig
import torch
import torch.nn as nn
import argparse

## for llama-3 accuracy test only ##

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='gradientai/Llama-3-8B-Instruct-262k')
parser.add_argument("--output_path", type=str, default='gradientai--Llama-3-8B-Instruct-262k-expanded')
args = parser.parse_args()

model_path = args.model_path

config = AutoConfig.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model1 = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

# allow mismatch
num_attention_heads = model1.config.num_attention_heads
model2 = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, num_key_value_heads=num_attention_heads, ignore_mismatched_sizes=True)

num_key_value_heads = config.num_key_value_heads
num_heads = config.num_attention_heads
num_key_value_groups =num_heads // num_key_value_heads

def expand_proj(
    layer: nn.Linear,
    num_key_value_groups: int,
    original_num_heads: int,
):
    input_dim = layer.in_features
    output_dim = layer.out_features
    head_dim = output_dim // original_num_heads

    original_weights = layer.weight.data
    weights_reshaped = original_weights.reshape(original_num_heads, head_dim, input_dim)
    weights_expanded = weights_reshaped.repeat_interleave(num_key_value_groups, dim=0).reshape(input_dim, output_dim * num_key_value_groups)
    weights_expanded = weights_expanded.contiguous()

    layer.weight.data = weights_expanded

    if layer.bias is not None:
        original_bias = layer.bias.data
        bias_reshaped = original_bias.reshape(original_num_heads, head_dim)
        bias_expanded = bias_reshaped.repeat_interleave(num_key_value_groups, dim=0).reshape(output_dim * num_key_value_groups)
        bias_expanded = bias_expanded.contiguous()

        layer.bias.data = bias_expanded

    return

for decoder_layer1, decoder_layer2 in zip(model1.model.layers, model2.model.layers):
    self_attn1 = decoder_layer1.self_attn
    k_proj1 = self_attn1.k_proj
    v_proj1 = self_attn1.v_proj

    expand_proj(k_proj1, num_key_value_groups, num_key_value_heads)
    expand_proj(v_proj1, num_key_value_groups, num_key_value_heads)

    decoder_layer2.self_attn.k_proj = k_proj1
    decoder_layer2.self_attn.v_proj = v_proj1

model2.save_pretrained(args.output_path)
tokenizer.save_pretrained(args.output_path)