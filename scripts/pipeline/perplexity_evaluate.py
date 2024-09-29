import os
import torch
from torch.nn import CrossEntropyLoss
import argparse
import numpy as np
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset, load_from_disk
import json

from MoA.models.interface import update_model_function
from MoA.dataset.utils import find_subtensor_position
from MoA.dataset.convert import multi_round_qa_to_multi_round_conversation
from MoA.universal.utils import get_user_assistant_prefix
from MoA.models.llama.modeling_llama import LlamaModel_use_streamingllm_attention

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def evaluate(model, tokenizer, args, user_prefix=None, assistant_prefix=None):
    model = model.eval()  # make model is in evaluation model
    max_length = args.max_length

    if args.load_from_disk:
        data = load_from_disk(args.dataset_dir)
    else:
        data = load_dataset(args.dataset_dir, "default", split=args.split)
        prompt_subset = load_dataset(args.dataset_dir, "prompt", split='train')

    data = data.filter(lambda x: x['dataset'] != 'hotpot_qa')

    for _, param in model.named_parameters():
        param.requires_grad = False

    if 'total_length_level' in data.column_names:
        data = data.filter(lambda x: ((x['total_length_level'] <= args.total_length_level) and (x['total_length_level'] > args.total_length_level - args.total_length_level_down)))
    
    print(f"using length level from {args.total_length_level - args.total_length_level_down + 1} to {args.total_length_level}")
    print(f"Total number of data samples: {len(data)}")

    loss_list = []
    if 'dataset' in data.column_names:
        loss_list = {dataset: [] for dataset in data.unique('dataset')}

    data_select_range = [0, len(data)]

    num_data = data_select_range[1] - data_select_range[0]

    prompt_dict = {}

    if not args.load_from_disk:
        for datum in prompt_subset:
            prompt_dict[datum['dataset_names']] = datum

    pbar = tqdm(total=num_data, desc="Evaluating")
    for i, data_sample in enumerate(data):
        if i >= data_select_range[1] or i < data_select_range[0]:
            break
        original_data_sample = data_sample

        if not args.load_from_disk:
            dataset_prompt = prompt_dict[data_sample['dataset']]

            data_sample = multi_round_qa_to_multi_round_conversation(
                data_sample,
                args.model_name,
                tokenizer,
                dataset_prompt['prompt_format'],
                dataset_prompt['question_format'],
                dataset_prompt['answer_format']
            )

        tokenizer_output = tokenizer(data_sample['text'], return_tensors='pt', max_length=max_length, truncation=True)
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

        gradient_mask = torch.ones(shift_labels.size(0), dtype=torch.bool, device=shift_labels.device)
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

        loss_list[original_data_sample['dataset']].append(loss.item())

        pbar.update(1)

    pbar.close()

    return loss_list

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name", type=str, default="lmsys/vicuna-7b-v1.5-16k", help="model name"
)
parser.add_argument("--tokenizer_name", type=str, default=None)
parser.add_argument(
    "--max_length", type=int, default=2048, help="max length of the sequence"
)
parser.add_argument("--dataset_dir", type=str, help="dataset directory")
parser.add_argument("--split", type=str, default=None, help="split of the dataset")
parser.add_argument("--load_from_disk", action="store_true")
parser.add_argument(
    "--response_mask", action="store_true", help="whether to mask the response part"
)
parser.add_argument(
    "--loss_type",
    choices=["cross_entropy", "ppl"],
    default="cross_entropy",
    help="loss type",
)
parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
parser.add_argument(
    "--moa_config",
    type=str,
    default=None,
    help="the path to moa configuration file",
)
parser.add_argument("--streamingllm", action="store_true", help="use streamingllm")
parser.add_argument("--global_size", type=int, default=4, help="global size")
parser.add_argument("--band_size", type=int, default=2044, help="band size")
parser.add_argument("--result_path", type=str, default=None, help="result path")
parser.add_argument("--total_length_level_down", type=int, default=None)

parser.add_argument("--h2o", action="store_true", help="use h2o")
parser.add_argument("--heavy", type=int, default=512)
parser.add_argument("--recent", type=int, default=512)

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

    assert args.max_length % 1024 == 0

    if args.total_length_level_down is None:
        if "llama-3" in args.model_name.lower():
            args.total_length_level_down = 4
        else:
            args.total_length_level_down = 2

    args.total_length_level = args.max_length // 1024

    config = AutoConfig.from_pretrained(args.model_name)
    config._attn_implementation_internal = "sdpa"
    config._attn_implementation = "sdpa"
    if args.tokenizer_name is None:
        args.tokenizer_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.padding_side='right'
        tokenizer.pad_token=tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config, torch_dtype=dtype, device_map='auto', attn_implementation='sdpa')

    if args.moa_config is not None:
        moa_config_path = args.moa_config
        with open(moa_config_path, 'r') as f:
            moa_config = json.load(f)
        # Add mixture of sparse attention capability to the model
        model = update_model_function(model, args.model_name)
        model.model.set_mixture_of_attention(moa_config, permute_head=True, last_hidden_only=False) # it is important to set last_hidden_only to False for PPL tests

    if args.streamingllm:
        LlamaModel_use_streamingllm_attention(model.model, args.global_size, args.band_size)

    if args.response_mask:
        user_prefix, assistant_prefix = get_user_assistant_prefix(args.tokenizer_name)
    else:
        user_prefix = None
        assistant_prefix = None

    loss_list = evaluate(model, tokenizer, args, user_prefix, assistant_prefix)
    num_dataset = len(loss_list)
    
    final_result = {}
    sum_result = 0.0
    for dataset_name, loss in loss_list.items():
        final_result[dataset_name] = np.mean(loss)
        sum_result += np.mean(loss)
        print("Dataset: {}, Average loss: {}".format(dataset_name, np.mean(loss)))

    # save the result in the result path using pandas
    import pandas as pd
    final_result['mean'] = sum_result / len(loss_list)
    df = pd.DataFrame(final_result, index=[0])
    df.to_csv(args.result_path, index=False)
