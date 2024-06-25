import logging
import socket
from datetime import datetime
import cProfile
import os

import torch
import random

import argparse
from typing import List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import json

from MoA.models.llama.modeling_llama import LlamaModel_use_block_sparse_attention_lut
from MoA.attention.set import set_static_attention_lut

# !!! remember to calculate only the last logit for lm_head when doing efficiency test.
# !!! to do this, refer to the return value of LlamaModel_block_sparse_lut_forward function in MoA/models/llama/modeling_llama.py. The same can be done to LlamaModel.forward in the transformers library.
# !!! also set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

logging.basicConfig(
   format="%(levelname)s:%(asctime)s %(message)s",
   level=logging.INFO,
   datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='lmsys/vicuna-7b-v1.5-16k', help='model name')
parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')
parser.add_argument('--lut_path', type=str, nargs='+', help='a list of lut path',)
parser.add_argument('--num_iters', type=int, default=1)
parser.add_argument('--block_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--prefill_len', type=int, required=True)
parser.add_argument('--decode_len', type=int, default=None, help='decoding max len. if None, just do prefilling')
parser.add_argument('--attention_implementation', type=str, default='eager')
parser.add_argument('--profiler', action='store_true')
parser.add_argument('--cuda_event', action='store_true')
parser.add_argument('--cuda_cache', action="store_true")
parser.add_argument('--record_memory', action='store_true')
parser.add_argument('--memory_file_name', type=str, default=None)
parser.add_argument('--test_mode', type=str, choices=['decode', 'prefill', 'whole'], default='whole')

args = parser.parse_args()

def fake_decode(model, tokenizer, prefill_output, max_len):
    result_idx = []
    past_key_values = prefill_output.past_key_values
    pred_token_idx = prefill_output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    # result_idx.append(pred_token_idx.item())

    for _ in range(max_len - 1):
        outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        # random sample a pred token idx from the logits shape
        pred_token_idx = random.randint(0, outputs.logits.shape[-1] - 1)
        pred_token_idx = torch.tensor([[pred_token_idx] for _ in range(args.batch_size)]).to(device="cuda:0", dtype=torch.long)

        # result_idx.append(pred_token_idx.item())
    
    return result_idx


def start_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )

def stop_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Stopping snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(enabled=None)

def export_memory_snapshot() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not exporting memory snapshot")
        return

    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    pickle_file_name = f"{host_name}_{timestamp}.pickle" if args.memory_file_name is None else args.memory_file_name

    if not pickle_file_name.endswith(".pickle"):
        pickle_file_name += ".pickle"

    torch_path = torch.__file__
    torch_path = os.path.dirname(torch_path)
    mem_viz_script_path = os.path.join(torch_path, "cuda/_memory_viz.py")

    html_file_name = pickle_file_name.replace(".pickle", ".html")

    try:
        logger.info(f"Saving snapshot to local file: {pickle_file_name} and {html_file_name}")
        torch.cuda.memory._dump_snapshot(pickle_file_name)
        # run a python code from bash
        os.system(f"python {mem_viz_script_path} trace_plot {pickle_file_name} -o {html_file_name}")

    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
        return


def run_model(num_iters=5, device="cuda:0"):
   
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=False, use_fast=False)
    config = AutoConfig.from_pretrained(args.model_name)
    config._attn_implementation = args.attention_implementation
    config._attn_implementation_internal = args.attention_implementation

    config.torch_dtype = "float16"
    ####
    model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config, trust_remote_code=False, torch_dtype=torch.float16, device_map='auto')
    ####
    # disable gradient for all parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    if args.lut_path is not None:
        assert args.attention_implementation == "eager" or args.attention_implementation == "sdpa"
        print("Using lut from {}, with block size {}".format(args.lut_path, args.block_size))

        model.model.use_block_sparse_attention_lut = LlamaModel_use_block_sparse_attention_lut.__get__(model.model)
        model.model.use_block_sparse_attention_lut(permute_head=True, sparse_decode=True)
        set_static_attention_lut(args.lut_path, None, model.model.layers, args.block_size, permute_head=True, sparse_decode=True)

    test_file = 'data/70_lines.jsonl'
    with open(test_file, 'r') as f:
        json_list = list(f)

    test_cases = []
    for json_str in json_list:
        test_cases.append(json.loads(json_str))

    prompt = test_cases[0]["prompt"]
    prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: " + prompt
    prompt = prompt + " ASSISTANT:"
    prompt = prompt * 10
    prompt = [prompt for _ in range(args.batch_size)]


    input = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=args.prefill_len, truncation=True)
    input_ids = input.input_ids
    print(f"input_shape: {input.input_ids.shape}")
    device = torch.device("cuda:0")


    # warmup
    with torch.inference_mode():
        for _ in range(2):
            output = model(input_ids.to(device), use_cache=True, output_hidden_states=False)
            output = None

    if args.test_mode == "decode":
        # generate the past key value states
        generation_prefill_output_list = []
        with torch.inference_mode():
            for _ in range(num_iters):
                generation_prefill_output_list.append(model(input.input_ids.to(device), use_cache=True, output_hidden_states=False))

    print("Start testing")
    
    # Start profiling
    if args.profiler:
        profiler = cProfile.Profile()
        profiler.enable()

    if args.cuda_event:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    if args.cuda_cache:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if args.record_memory:
        start_record_memory_history()

    if args.test_mode == "prefill":
        with torch.inference_mode():
            for _ in range(num_iters):
                output = model(input.input_ids.to(device), use_cache=True, output_hidden_states=False)
                print(len(output))
                output = None

    if args.test_mode == 'decode':
        with torch.inference_mode():
            for i in range(num_iters):
                result_idx = fake_decode(model, tokenizer, generation_prefill_output_list[i], args.decode_len)
                result_idx = None

    if args.test_mode == 'whole':
        with torch.inference_mode():
            for i in range(num_iters):
                outputs = model(input.input_ids.to(device), use_cache=True, output_hidden_states=False)
                # past_key_values = outputs.past_key_values
                # pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                # outputs = None

                # for _ in range(args.decode_len - 1):
                #     outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values, use_cache=True)
                #     past_key_values = outputs.past_key_values
                #     pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                fake_decode(model, tokenizer, outputs, args.decode_len)
                outputs = None

                result_idx = None

                print(f"iter {i} done")


    if args.profiler:
        profiler.disable()
        profiler.dump_stats("profile.prof")
    
    if args.cuda_event:
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Elapsed time: {elapsed_time_ms} ms")
        print(f"per iter: {elapsed_time_ms / num_iters} ms")

        if args.test_mode == 'prefill':
            print(f"input shape: {input.input_ids.shape}")
            print(f"per token: {elapsed_time_ms / num_iters / args.batch_size / input.input_ids.shape[1]} ms")

                                
        if args.test_mode == "decode":
            print(f"per token: {elapsed_time_ms / num_iters / args.decode_len/args.batch_size} ms")
            print(f"throughput: {args.batch_size * args.decode_len * num_iters / (elapsed_time_ms / 1000)} tokens/s")

    if args.cuda_cache:
        max_memory = torch.cuda.max_memory_allocated() // 2**20
        print(f"Max memory: {max_memory} MB")

    if args.record_memory:
        # Create the memory snapshot file
        export_memory_snapshot()

        # Stop recording memory snapshot history
        stop_record_memory_history()

if __name__ == "__main__":
    run_model(args.num_iters)