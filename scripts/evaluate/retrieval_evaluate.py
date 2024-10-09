from math import floor
import os
from turtle import position
from typing import Dict, Tuple, Optional
from cvxpy import length
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
import argparse
from datasets import load_from_disk, load_dataset
import re
from fastchat.model import get_conversation_template
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional
import json

from MoA.models.interface import update_model_function
from MoA.attention.set import set_static_attention_lut
from MoA.models.llama.modeling_llama import LlamaModel_use_streamingllm_attention
from MoA.models.llama.h2o import convert_kvcache_llama_heavy_recent
from MoA.dataset.long_eval.visualize import plot_correct_rate_heatmap_input_length_position, plot_data_count_heatmap_input_length_position

def generate_input(test_case: Dict, tokenizer, model_name_or_path: str) -> Tuple[str, int]:
    """
    Generate the prompt and calculate its length.
    """
    prompt: str = test_case["question"]

    conv = get_conversation_template(model_name_or_path)
    
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_token_ids = conv.stop_token_ids

    return prompt, stop_token_ids

def process_prompt(input, model, tokenizer, test_case: Dict, output_file: Optional[str] = None, idx: int = 0, stop_token_ids: Optional[list] = None) -> Tuple[bool, int, str]:
    expected_number: int = test_case["value"]

    prompt_length = input.input_ids.shape[-1]

    # print(f"Prompt length: {prompt_length}")
    
    use_cache = True

    device = getattr(model, "device", "cpu")
    
    input = input.to(device)

    output = model.generate(
        **input, 
        max_new_tokens=100, 
        use_cache=use_cache,
        eos_token_id=stop_token_ids,
    )[0]
    output = output[prompt_length:]
    output = tokenizer.batch_decode([output], skip_special_tokens=True)[0]

    # Matching the last digit of the model output
    response_number = re.findall("\d+", output)
    if response_number is not None and len(response_number) > 0:
        response_number = int(response_number[-1])
    else:
        print(f"Got unparsable result")
        response_number = -1

    summary = f"Label: {expected_number}, Predict: {output}, Parsed: {response_number}".replace('\n', ' ')
    
    if output_file is not None:
        print(summary)
        if idx ==0:
            with open(output_file, "w") as f:
                try:
                    f.write(summary)
                    f.write("\n")
                except:
                    f.write(f"Label: {expected_number}, Predict: -1, Parsed: {response_number}, prompt length: {prompt_length}".replace('\n', ' '))
                    f.write("\n")
        else:
            with open(output_file, "a+") as f:
                try:
                    f.write(summary)
                    f.write("\n")
                except:
                    f.write(f"Label: {expected_number}, Predict: -1, Parsed: {response_number}, prompt length: {prompt_length}".replace('\n', ' '))
                    f.write("\n")


    # if expected_number == response_number:
    #     print("Correct")
    # else:
    #     print("Incorrect")
    
    return expected_number == response_number, summary

if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, help="Path of the model"
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default=None)
    parser.add_argument(
        "--dtype", type=str, choices=["fp16", "fp32", "bf16"], default="fp16"
    )
    parser.add_argument(
        "--moa_config",
        type=str,
        default=None,
        help="the path to moa configuration file",
    )
    parser.add_argument('--not_permute_head', action='store_true')
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        default=False,
        help="Whether to use flash attention",
    )
    parser.add_argument(
        "--use_streamingLLM",
        action="store_true",
        default=False,
        help="Whether to use streaming LLM",
    )
    parser.add_argument(
        "--band_size", default=2044, type=int
    )
    parser.add_argument(
        "--global_size", type=int, default=4,
    )
    parser.add_argument(
        '--h2o', action='store_true', help='Use H2O attention'
    )

    parser.add_argument('--recent', type=int, default=1024, help='Recent budget ratio')
    parser.add_argument('--heavy', type=int, default=1024, help='Heavy budget ratio')

    parser.add_argument(
        "--output_dir",
        type=str,
        default="local/universal/dataset/longeval_lines/",
        help="Folder to the output file",
    )
    parser.add_argument(
        "--length_level",
        nargs="+",
        type=int,
        default=[8],
        help="Length level to test, the number is multiplied by token interval",
    )
    parser.add_argument(
        "--token_interval",
        type=int,
        default=1024,
        help="Token interval",
    )
    parser.add_argument(
        "--test_num",
        type=int,
        default=10,
        help="Number of test case for each length level and position interval",
    )

    parser.add_argument('--dataset_path', type=str, default=None)
    args = parser.parse_args()

    args.use_flash_attention = True if (not args.use_streamingLLM) and (not args.h2o) and (args.moa_config is None) else args.use_flash_attention # noqa: if lut_path is not None, use flash attention
    print("using flash attention", args.use_flash_attention)

    # load tokenizer
    if args.tokenizer_name is None:
        args.tokenizer_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # define model config
    config = AutoConfig.from_pretrained(args.model_name)
    if args.use_flash_attention:
        config._attn_implementation = "flash_attention_2"
    else:
        config._attn_implementation_internal = "sdpa"

    # load model
    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'fp32':
        dtype = torch.float32
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(f"unsupported type: {args.dtype}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        device_map="auto",
        attn_implementation=(
            "sdpa" if not args.use_flash_attention else "flash_attention_2"
        ),
        torch_dtype=dtype,
    ).eval()
    
    if args.moa_config is not None:
        moa_config_path = args.moa_config
        with open(moa_config_path, 'r') as f:
            moa_config = json.load(f)
        # Add mixture of sparse attention capability to the model
        model = update_model_function(model, args.model_name)
        model.model.set_mixture_of_attention(moa_config, permute_head=True)

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    """
    evaluate
    """
    # model
    model_name = args.model_name

    # load dataset
    remote_dataset_path = "nics-efc/MoA_Long_Retrieval"

    if args.dataset_path is not None:
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(remote_dataset_path, split="test")
    

    # iter through dataset
    result_dict = {
        'id': [],
        'is_correct': [],
        'prompt_length': [],
        'num_lines': [],
        'key_id': [],
        'length_level': [],
        'summary': [],
        'context_length': [],
    }

    token_interval = args.token_interval
    length_level = args.length_level
    length_level_interval = [token_interval*i for i in length_level]

    position_interval = 0.1
    inverse_position_interval = int(1/position_interval) # noqa
    position_level = [i for i in range(1, 11)]
    position_level_interval = [position_interval*i for i in position_level]

    # input length * position level
    test_num_bound = args.test_num

    # noqa: random sample 100 items from the dataset
    # dataset = dataset.shuffle(seed=42).select(range(50))
    if args.use_streamingLLM:
        context_length_range = [int(length * 0.5) for length in length_level_interval]
        print(context_length_range)
    else:
        context_length_range = [0.0]
    global_size = 4

    os.makedirs(args.output_dir, exist_ok=True)

    for context_length in tqdm(context_length_range, position=0):
        # initialize everything
        now = datetime.now()
        datetime_str = now.strftime("%Y%m%d-%H%M")
        print(f"running test at {datetime_str}")
        meshgrid_count = np.zeros((len(length_level), len(position_level)))

        # process bar
        pbar = tqdm(total=len(dataset)-1, position=1)

        # set the attention implementation here
        if args.use_streamingLLM:
            LlamaModel_use_streamingllm_attention(model.model, global_size=args.global_size, band_size=args.band_size, max_length=16384)
            print(f"using streamingllm, global size: {args.global_size}, band size: {args.band_size}")

        if args.h2o:
            model = convert_kvcache_llama_heavy_recent(model, args.heavy, args.recent)
            print(f"using h2o, heavy: {args.heavy}, recent: {args.recent}")

        # start test
        for i, data in enumerate(dataset):
            pbar.update(1)

            prompt, stop_token_ids = generate_input(data, tokenizer, model_name)

            # check whether tokenized_len key is in data
            if 'tokenized_len' in data:
                prompt_length = data['tokenized_len']
                current_length_level = int((prompt_length - 1) // token_interval) + 1
                if (current_length_level not in length_level):
                    continue

            input = tokenizer(prompt, return_tensors="pt")

            # check length and record
            prompt_length = input.input_ids.shape[-1] # the length of tokenized prompt
            current_length_level = int((prompt_length - 1) // token_interval) + 1
            current_position_level = floor((float(data['key_id']) * inverse_position_interval / float(data['num_lines'])))
            if (current_length_level not in length_level):
                continue
            # current input length now become a index 
            current_length_level = length_level.index(current_length_level)
            meshgrid_count[current_length_level, current_position_level] += 1
            if meshgrid_count[current_length_level, current_position_level] > test_num_bound:
                continue

            # retrieval test
            with torch.inference_mode():
                is_correct, summary = process_prompt(input, model, tokenizer, data, stop_token_ids=stop_token_ids)
            
            # record
            pbar.write(f"Prompt_Length: {prompt_length}, Correct: {is_correct}, {summary}")
            result_dict['id'].append(i)
            result_dict['is_correct'].append(is_correct)
            result_dict['prompt_length'].append(prompt_length)
            result_dict['summary'].append(summary)
            result_dict['num_lines'].append(data['num_lines'])
            result_dict['key_id'].append(data['key_id'])
            result_dict['length_level'].append(length_level_interval[current_length_level])
            result_dict['context_length'].append(context_length)

        # save and visualize
        df = pd.DataFrame(result_dict)
        df = df[df['context_length'] == context_length]
        print(df)

        output_dir = args.output_dir
        try:
            df.to_csv(os.path.join(output_dir, f"test_result_{datetime_str}.csv"), index=False)
            # plot everything
            plot_correct_rate_heatmap_input_length_position(df, os.path.join(output_dir, f"correct_rate_heatmap_{datetime_str}.png"))

            plot_data_count_heatmap_input_length_position(df, os.path.join(output_dir, f"data_point_distribution_heatmap.png"))
        except:
            print("error in saving the result")
            pass

    print("Retrieval Evaluation Finished")
    now = datetime.now()
    datetime_str = now.strftime("%Y%m%d-%H%M")
    print(f"Finish at {datetime_str}")


    # visualize and save the results
    df = pd.DataFrame(result_dict)
    print(df)

    correct_rate = df['is_correct'].sum() / len(df)
    print(f"The overall correct rate is {correct_rate:.4f}")

    print(f"Saving the result to {args.output_dir}")
    output_dir = args.output_dir
    
    # plot everything
    df.to_csv(os.path.join(output_dir, f"test_result_{datetime_str}.csv"), index=False)
    plot_correct_rate_heatmap_input_length_position(df, os.path.join(output_dir, f"correct_rate_heatmap_{datetime_str}.png"))
    plot_data_count_heatmap_input_length_position(df, os.path.join(output_dir, f"data_point_distribution_heatmap.png"))

    print("Retrieval Visualization Finished")