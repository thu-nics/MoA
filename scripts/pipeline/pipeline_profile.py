import os
import torch
from torch.nn import CrossEntropyLoss
import argparse
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_from_disk
import warnings

from MoA.models.llama.modify_llama_profile import LlamaModel_use_attention_matrix_grad_log
from MoA.dataset.utils import find_subtensor_position
from MoA.universal.utils import get_user_assistant_prefix

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def Grad_Collect(model, tokenizer, args, data=None, max_length=2048, user_prefix=None, assistant_prefix=None):
    model = model.eval()

    for name, param in model.named_parameters():
        if param.dim() < 2:
            param.requires_grad = False
        if 'lm_head' in name or 'embed' in name:
            param.requires_grad = False

    attention_record_keys = ['sum_effect']

    grad_W_dict = {}

    if 'total_length_level' in data.column_names:
        data = data.filter(lambda x: (x['total_length_level'] <= args.total_length_level) and (x['total_length_level'] > args.total_length_level - args.total_length_level_down))

    if args.sort:
        # sort the data by total length level, large at the front
        data = data.sort('total_length_level', reverse=True)

    print(f"Total number of data samples: {len(data)}")

    data_select_range = args.data_range
    if data_select_range is None:
        data_select_range = [0, len(data)]
    elif len(data_select_range) == 1:
        data_select_range = [0, data_select_range[0]]
    elif len(data_select_range) > 2:
        raise ValueError("data_range should be a list of 0, 1 or 2 elements")
    
    if data_select_range[1] > len(data):
        warnings.warn("data_range[1] {} is larger than the length of the dataset, set to the length of the dataset {}".format(data_select_range[1], len(data)))
        data_select_range[1] = len(data)

    num_data = data_select_range[1] - data_select_range[0]
    
    pbar = tqdm(total=num_data, desc="Collecting Grad")
    for i, data_sample in enumerate(data):
        if i >= data_select_range[1] or i < data_select_range[0]:
            break
        tokenized_prompt = tokenizer.tokenize(data_sample['text'])
        if len(tokenized_prompt) > max_length + 1:
            print(f"tokenized prompt len: {len(tokenized_prompt)} is larger than max_length: {max_length}, discrad it")
            pbar.update(1)
            continue

        tokenizer_output = tokenizer(data_sample['text'], return_tensors='pt', padding = 'max_length', max_length=max_length + 1, truncation=True)
        data_sample = tokenizer_output['input_ids']
        data_mask = (tokenizer_output['attention_mask']==1)

        logits = model(data_sample.to(next(model.parameters()).device))[0] # cross entropy loss
        labels = data_sample.to(next(model.parameters()).device)
        data_mask = data_mask.to(next(model.parameters()).device)

        if args.response_mask:
            assert user_prefix is not None and assistant_prefix is not None
            # find and mask all the ids that is between the user and assistant prefix
            user_prefix_pos = find_subtensor_position(user_prefix, data_sample)
            assistant_prefix_pos = find_subtensor_position(assistant_prefix, data_sample)
            # user A, assistant B, user C, assistant D ...; find B, D ..., that is the content after assistant_prefix and before user_prefix, or the end of the sequence
            response_mask = torch.zeros_like(data_mask)
            for batch_id in range(response_mask.size(0)):
                starts = assistant_prefix_pos[batch_id] + len(assistant_prefix)
                ends = torch.cat((user_prefix_pos[batch_id], torch.tensor([data_sample.size(1)])))
                for i in range(len(assistant_prefix_pos[batch_id])):
                    start = starts[i]
                    end = ends[i+1]
                    response_mask[batch_id][start:end] = 1

            data_mask = response_mask & data_mask

        # shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_data_mask = data_mask[..., 1:].contiguous()

        vocab_size = shift_logits.size(-1)

        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_data_mask = shift_data_mask.view(-1)

        gradient_mask = torch.zeros(shift_labels.size(0), dtype=torch.bool, device=shift_labels.device)
        # use args.gradient_range to select the gradient range
        keep_index = [i for i in range(*args.gradient_range) if i < max_length]
        gradient_mask[keep_index] = True

        gradient_mask = shift_data_mask & gradient_mask

        filtered_logits = shift_logits[gradient_mask]
        filtered_labels = shift_labels[gradient_mask]

        filtered_labels = filtered_labels.to(filtered_logits.device)

        loss = loss_fct(filtered_logits, filtered_labels)

        if args.loss_type == 'ppl':
            loss = torch.exp(loss) # use ppl as loss
        elif args.loss_type == 'cross_entropy':
            pass
        else:
            raise NotImplementedError

        loss.backward()

        # store weight grad on cpu
        if args.weight_gradient:
            for name, m in model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    if not m.weight.requires_grad:
                        continue
                    grad_W = m.weight.grad.detach()
                    if args.gradient_abs:
                        if name in grad_W_dict:
                            grad_W_dict[name + '.weight'] += grad_W.abs().cpu()
                        else:
                            grad_W_dict[name + '.weight'] = grad_W.abs().cpu()
                    else:
                        if name in grad_W_dict:
                            grad_W_dict[name + '.weight'] += grad_W.cpu()
                        else:
                            grad_W_dict[name + '.weight'] = grad_W.cpu()

        # store attention grad on cpu
        log = model.model.get_attention_matrix_log(max_length=max_length, take_abs=True, aggregating_block_size=args.aggregating_block_size)
        for key in attention_record_keys:
            if args.gradient_abs:
                if key in model.model.attention_matrix_log:
                    model.model.attention_matrix_log[key][0] = model.model.attention_matrix_log[key][0] + torch.abs(log[key])
                else:
                    # called the first time
                    model.model.attention_matrix_log[key].append(torch.abs(log[key]))
                    print("the shape of the logged {} is {}".format(key, model.model.attention_matrix_log[key][0].shape))
            else:
                if key in model.model.attention_matrix_log:
                    model.model.attention_matrix_log[key][0] = model.model.attention_matrix_log[key][0] + log[key]
                else:
                    # called the first time
                    model.model.attention_matrix_log[key].append(log[key])
                    print("the shape of the logged {} is {}".format(key, model.model.attention_matrix_log[key][0].shape))

        pbar.update(1)
    
    pbar.close()

    key = 'sum_effect'
    # key = 'grad'
    grad_Attn_tensor = torch.sum(model.model.attention_matrix_log[key][0], dim=0) / num_data

    for key, value in grad_W_dict.items():
        grad_W_dict[key] = value / num_data

    return grad_W_dict, grad_Attn_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='lmsys/vicuna-13b-v1.5-16k', help='model name')
parser.add_argument('--tokenizer_name', type=str, default=None)
parser.add_argument('--grad_dir', type=str, help='directory to save grad')
parser.add_argument('--max_length', type=int, default=2048, help='max length of the sequence')
parser.add_argument('--gradient_range', type=int, nargs='+', default=None, help='gradient range. by default cross entropy loss on all token prediction to do back propogation.')
parser.add_argument('--dataset_dir', type=str, help='dataset directory')
parser.add_argument('--data_range', nargs='+', type=int, default=None, help='data range')
parser.add_argument('--gradient_abs', action='store_true', help='whether to use absolute value for each gradient')
parser.add_argument('--response_mask', action='store_true', help='whether to mask the response part')
parser.add_argument('--weight_gradient', action='store_true', help='whether to collect gradient of weight')
parser.add_argument('--loss_type', choices=['cross_entropy', 'ppl'], default='cross_entropy', help='loss type')
parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='bf16')
parser.add_argument('--aggregating_block_size', type=int, default=64, help='block size for aggregating attention matrix')
parser.add_argument('--total_length_level_down', type=int, default=2, help="the token lengths passed in for profiling is max_length - total_length_level_down * 1024")
parser.add_argument('--sort', action='store_true', help='whether to sort the data by length')

args = parser.parse_args()


if __name__ == '__main__':
    if args.dtype == 'fp32':
        dtype = torch.float32
    elif args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError("unsupported data type")

    if args.gradient_range is None:
        args.gradient_range = [0, args.max_length]
    assert len(args.gradient_range) == 2

    assert args.max_length % 1024 == 0

    args.total_length_level = args.max_length // 1024

    config = AutoConfig.from_pretrained(args.model_name)
    config._attn_implementation_internal = "eager"

    if args.tokenizer_name is None:
        args.tokenizer_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.padding_side='right'

    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config, torch_dtype=dtype, device_map='auto')

    if model.config.architectures[0] == "LlamaForCausalLM":
        print("LlamaForCausalLM")
        LlamaModel_use_attention_matrix_grad_log(model.model)
        # model.model.gradient_checkpointing = True
        model.gradient_checkpointing_enable()
    else:
        raise NotImplementedError
    
    if args.response_mask:
        user_prefix, assistant_prefix = get_user_assistant_prefix(args.tokenizer_name)
    else:
        user_prefix = None
        assistant_prefix = None

    dataset = load_from_disk(args.dataset_dir)

    grad_W_dict, grad_Attn_tensor = Grad_Collect(model, tokenizer, args, dataset, args.max_length, user_prefix=user_prefix, assistant_prefix=assistant_prefix)
    if not os.path.exists(args.grad_dir):
        os.makedirs(args.grad_dir)
    print("Saving profile grad to {}".format(args.grad_dir))
    torch.save(grad_W_dict, os.path.join(args.grad_dir, 'grad_w_dict_{}.pt'.format(args.max_length)))
    torch.save(grad_Attn_tensor, os.path.join(args.grad_dir, 'grad_attn_tensor_{}.pt'.format(args.max_length)))