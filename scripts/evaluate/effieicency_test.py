import logging
import socket
from datetime import datetime
import cProfile
import os
import time
import threading
import math

import torch
import random
import torch.cuda.nvtx as nvtx
import argparse
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import json

from MoA.models.interface import update_model_function

try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except (ImportError, pynvml.NVMLError) as err:
    print(f"Warning: PYNVML not available: {err}. Power monitoring will be disabled.")
    PYNVML_AVAILABLE = False


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
parser.add_argument('--prefill_len', type=int, required=True)


parser.add_argument('--model_name', type=str, default='lmsys/vicuna-7b-v1.5-16k', help='model name')
parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')
parser.add_argument('--moa_config', type=str, default=None, help='the path to moa configuration file')
parser.add_argument('--num_iters', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--decode_len', type=int, default=None, help='decoding max len. if None, just do prefilling')
parser.add_argument('--attention_implementation', type=str, choices=['sdpa', 'eager'], default='sdpa',
                    help="Choose the type of attention implementation to use: 'sdpa' or 'eager'.")
parser.add_argument('--profiler', action='store_true')
parser.add_argument('--cuda_event', action='store_true')
parser.add_argument('--cuda_cache', action="store_true")
parser.add_argument('--record_memory', action='store_true')
parser.add_argument('--memory_file_name', type=str, default=None)
parser.add_argument('--test_mode', type=str, choices=['decode', 'prefill', 'whole'], default='whole')
parser.add_argument('--monitor_power', action='store_true', help='Monitor GPU power usage during inference')
parser.add_argument('--power_sample_interval', type=float, default=0.01, help='Power sampling interval in seconds')
parser.add_argument('--measure_actual_flops', action='store_true', help='Measure actual FLOPs using PyTorch profiler')
parser.add_argument('--export_flop_trace', type=str, default=None, help='Export FLOP profiler trace to file (Chrome trace format)')

args = parser.parse_args()

def fake_decode(model, tokenizer, prefill_output, max_len):
    result_idx = []
    past_key_values = prefill_output.past_key_values
    pred_token_idx = prefill_output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    # result_idx.append(pred_token_idx.item())

    for step in tqdm(range(max_len - 1), desc="Decoding"):
        nvtx.range_push(f"decode_step_{step}")
        outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values, use_cache=True)
        nvtx.range_pop()
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

class PowerMonitor:
    def __init__(self, device_id=0, sample_interval=0.01):
        self.device_id = device_id
        self.sample_interval = sample_interval
        self.power_samples = []
        self.timestamps = []
        self.monitoring = False
        self.monitor_thread = None
        
        if PYNVML_AVAILABLE:
            try:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                # Test if we can get power reading
                pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.power_available = True
            except pynvml.NVMLError as err:
                print(f"Warning: Cannot get power readings for GPU {device_id}: {err}")
                self.power_available = False
        else:
            self.power_available = False
    
    def _monitor_power(self):
        """Background thread function to monitor power usage"""
        start_time = time.time()
        while self.monitoring:
            try:
                if self.power_available:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(self.handle)  # Power in milliwatts
                    current_time = time.time()
                    self.power_samples.append(power_usage / 1000.0)  # Convert to watts
                    self.timestamps.append(current_time - start_time)
                time.sleep(self.sample_interval)
            except Exception as e:
                print(f"Error monitoring power: {e}")
                break
    
    def start_monitoring(self):
        """Start power monitoring in a background thread"""
        if not self.power_available:
            return
        
        self.power_samples = []
        self.timestamps = []
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_power)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop power monitoring and return statistics"""
        if not self.power_available:
            return None
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if not self.power_samples:
            return None
        
        total_duration = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0
        avg_power = sum(self.power_samples) / len(self.power_samples)
        max_power = max(self.power_samples)
        min_power = min(self.power_samples)
        
        # Calculate energy consumption (power * time)
        energy_joules = 0
        for i in range(1, len(self.power_samples)):
            time_diff = self.timestamps[i] - self.timestamps[i-1]
            avg_power_interval = (self.power_samples[i] + self.power_samples[i-1]) / 2
            energy_joules += avg_power_interval * time_diff
        
        return {
            'avg_power_watts': avg_power,
            'max_power_watts': max_power,
            'min_power_watts': min_power,
            'total_energy_joules': energy_joules,
            'duration_seconds': total_duration,
            'num_samples': len(self.power_samples),
            'power_samples': self.power_samples,
            'timestamps': self.timestamps
        }
    
    def get_power_info(self):
        """Get current GPU power information"""
        if not self.power_available:
            return None
        
        try:
            power_usage = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert to watts
            power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(self.handle) / 1000.0  # Convert to watts
            
            return {
                'current_power_watts': power_usage,
                'power_limit_watts': power_limit
            }
        except Exception as e:
            print(f"Error getting power info: {e}")
            return None

class FLOPAnalyzer:
    """Analyzer for measuring actual FLOPs during model execution."""
    
    def __init__(self):
        self.total_flops = 0
        self.operation_counts = {}
        self.profiler_results = None
        self.current_profiler = None
    
    def start_profiling(self):
        """Start FLOP profiling context."""
        if self.current_profiler is not None:
            return  # Already profiling
            
        self.current_profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
            with_modules=True,
        )
        self.current_profiler.__enter__()
    
    def stop_profiling(self):
        """Stop FLOP profiling and return measured FLOPs."""
        if self.current_profiler is None:
            return 0
            
        self.current_profiler.__exit__(None, None, None)
        prof = self.current_profiler
        self.current_profiler = None
        self.profiler_results = prof
        
        return self._analyze_flops(prof)
    
    def _analyze_flops(self, prof):
        """Analyze profiler results to extract FLOP counts."""
        total_flops = 0
        operation_counts = {}
        
        for event in prof.events():
            if hasattr(event, 'flops') and event.flops is not None:
                total_flops += event.flops
                op_name = event.name
                if op_name not in operation_counts:
                    operation_counts[op_name] = {'count': 0, 'flops': 0}
                operation_counts[op_name]['count'] += 1
                operation_counts[op_name]['flops'] += event.flops
        
        # Update cumulative counts
        for op_name, data in operation_counts.items():
            if op_name not in self.operation_counts:
                self.operation_counts[op_name] = {'count': 0, 'flops': 0}
            self.operation_counts[op_name]['count'] += data['count']
            self.operation_counts[op_name]['flops'] += data['flops']
        
        self.total_flops += total_flops
        return total_flops
    
    def get_flop_summary(self):
        """Get summary of FLOP measurements."""
        if not self.operation_counts:
            return "No FLOP data available"
        
        summary = f"Total FLOPs measured: {self.total_flops:.2e}\n"
        summary += "Top operations by FLOPs:\n"
        
        # Sort operations by FLOP count
        sorted_ops = sorted(self.operation_counts.items(), 
                          key=lambda x: x[1]['flops'], reverse=True)
        
        for op_name, data in sorted_ops[:10]:  # Top 10 operations
            summary += f"  {op_name}: {data['flops']:.2e} FLOPs ({data['count']} calls)\n"
        
        return summary
    
    def export_chrome_trace(self, path):
        """Export profiler results for visualization."""
        if self.profiler_results:
            self.profiler_results.export_chrome_trace(path)

def run_model(num_iters=5, device="cuda:0"):
   
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=False, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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

    if args.moa_config is not None:
        moa_config_path = args.moa_config
        with open(moa_config_path, 'r') as f:
            moa_config = json.load(f)
        # Add mixture of sparse attention capability to the model
        model = update_model_function(model, args.model_name)
        model.model.set_mixture_of_attention(moa_config, permute_head=True, moa_verbose=False)

    # Initialize power monitor
    power_monitor = None
    if args.monitor_power:
        # Get GPU device ID (assume using cuda:0)
        gpu_id = 0 if device == "cuda:0" else int(device.split(':')[1])
        power_monitor = PowerMonitor(device_id=gpu_id, sample_interval=args.power_sample_interval)
        
        # Print initial power info
        power_info = power_monitor.get_power_info()
        if power_info:
            print(f"GPU Power Info:")
            print(f"  Current Power: {power_info['current_power_watts']:.2f} W")
            print(f"  Power Limit: {power_info['power_limit_watts']:.2f} W")

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
            for _ in tqdm(range(num_iters), desc="Decoding"):
                generation_prefill_output_list.append(model(input.input_ids.to(device), use_cache=True, output_hidden_states=False))

    print("Start testing")
    
    # Initialize FLOP analyzer if requested
    flop_analyzer = None
    if args.measure_actual_flops:
        flop_analyzer = FLOPAnalyzer()
    
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

    # Start power monitoring
    prefill_power_stats = None
    decode_power_stats = None
    
    if args.test_mode == "prefill":
        if power_monitor:
            power_monitor.start_monitoring()
        
        # Start FLOP profiling if requested
        if flop_analyzer:
            flop_analyzer.start_profiling()
        
        with torch.inference_mode():
            for i in range(num_iters):
                nvtx.range_push(f"prefill_iter_{i}")
                output = model(input.input_ids.to(device), use_cache=True, output_hidden_states=False)
                nvtx.range_pop()
                print(len(output))
                output = None
        
        # Stop FLOP profiling
        if flop_analyzer:
            prefill_flops = flop_analyzer.stop_profiling()
        
        if power_monitor:
            prefill_power_stats = power_monitor.stop_monitoring()

    if args.test_mode == 'decode':
        if power_monitor:
            power_monitor.start_monitoring()
        
        # Start FLOP profiling if requested
        if flop_analyzer:
            flop_analyzer.start_profiling()
        
        with torch.inference_mode():
            for i in range(num_iters):
                nvtx.range_push(f"decode_iter_{i}")
                result_idx = fake_decode(model, tokenizer, generation_prefill_output_list[i], args.decode_len)
                nvtx.range_pop()
                result_idx = None
        
        # Stop FLOP profiling
        if flop_analyzer:
            decode_flops = flop_analyzer.stop_profiling()
        
        if power_monitor:
            decode_power_stats = power_monitor.stop_monitoring()

    if args.test_mode == 'whole':
        prefill_flops_total = 0
        decode_flops_total = 0
        
        with torch.inference_mode():
            for i in range(num_iters):
                # Prefill phase with separate power monitoring
                if power_monitor:
                    power_monitor.start_monitoring()
                
                # Start FLOP profiling for prefill
                if flop_analyzer:
                    flop_analyzer.start_profiling()
                
                nvtx.range_push(f"whole_prefill_iter_{i}")
                outputs = model(input.input_ids.to(device), use_cache=True, output_hidden_states=False)
                nvtx.range_pop()
                
                # Stop FLOP profiling for prefill
                if flop_analyzer:
                    iter_prefill_flops = flop_analyzer.stop_profiling()
                    prefill_flops_total += iter_prefill_flops
                
                if power_monitor:
                    current_prefill_stats = power_monitor.stop_monitoring()
                    if prefill_power_stats is None:
                        prefill_power_stats = current_prefill_stats
                    else:
                        # Accumulate stats across iterations
                        if current_prefill_stats:
                            prefill_power_stats['avg_power_watts'] = (prefill_power_stats['avg_power_watts'] * prefill_power_stats['num_samples'] + 
                                                                    current_prefill_stats['avg_power_watts'] * current_prefill_stats['num_samples']) / (prefill_power_stats['num_samples'] + current_prefill_stats['num_samples'])
                            prefill_power_stats['max_power_watts'] = max(prefill_power_stats['max_power_watts'], current_prefill_stats['max_power_watts'])
                            prefill_power_stats['min_power_watts'] = min(prefill_power_stats['min_power_watts'], current_prefill_stats['min_power_watts'])
                            prefill_power_stats['total_energy_joules'] += current_prefill_stats['total_energy_joules']
                            prefill_power_stats['duration_seconds'] += current_prefill_stats['duration_seconds']
                            prefill_power_stats['num_samples'] += current_prefill_stats['num_samples']
                
                # Decode phase with separate power monitoring
                if power_monitor:
                    power_monitor.start_monitoring()
                
                # Start FLOP profiling for decode
                if flop_analyzer:
                    flop_analyzer.start_profiling()
                
                nvtx.range_push(f"whole_decode_iter_{i}")
                fake_decode(model, tokenizer, outputs, args.decode_len)
                nvtx.range_pop()
                
                # Stop FLOP profiling for decode
                if flop_analyzer:
                    iter_decode_flops = flop_analyzer.stop_profiling()
                    decode_flops_total += iter_decode_flops
                
                if power_monitor:
                    current_decode_stats = power_monitor.stop_monitoring()
                    if decode_power_stats is None:
                        decode_power_stats = current_decode_stats
                    else:
                        # Accumulate stats across iterations
                        if current_decode_stats:
                            decode_power_stats['avg_power_watts'] = (decode_power_stats['avg_power_watts'] * decode_power_stats['num_samples'] + 
                                                                   current_decode_stats['avg_power_watts'] * current_decode_stats['num_samples']) / (decode_power_stats['num_samples'] + current_decode_stats['num_samples'])
                            decode_power_stats['max_power_watts'] = max(decode_power_stats['max_power_watts'], current_decode_stats['max_power_watts'])
                            decode_power_stats['min_power_watts'] = min(decode_power_stats['min_power_watts'], current_decode_stats['min_power_watts'])
                            decode_power_stats['total_energy_joules'] += current_decode_stats['total_energy_joules']
                            decode_power_stats['duration_seconds'] += current_decode_stats['duration_seconds']
                            decode_power_stats['num_samples'] += current_decode_stats['num_samples']
                result_idx = None

                print(f"iter {i} done")
        
        # Calculate average FLOPs for whole mode
        if flop_analyzer:
            prefill_flops = prefill_flops_total / num_iters
            decode_flops = decode_flops_total / num_iters

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

    # Measure actual FLOPs if requested
    if args.measure_actual_flops:
        print("\n" + "="*50)
        print("ACTUAL FLOP MEASUREMENT")
        print("="*50)
        
        if args.test_mode == "prefill":
            print(f"Prefill FLOPs per iteration: {prefill_flops / num_iters:.2e}")
            print(f"Prefill FLOPs per token: {prefill_flops / num_iters / args.batch_size / input.input_ids.shape[1]:.2e}")
            
        elif args.test_mode == "decode":
            print(f"Decode FLOPs per iteration: {decode_flops / num_iters:.2e}")
            print(f"Decode FLOPs per token: {decode_flops / num_iters / args.decode_len / args.batch_size:.2e}")
            
        elif args.test_mode == "whole":
            total_flops_per_iter = prefill_flops + decode_flops
            prefill_tokens = args.batch_size * input.input_ids.shape[1]
            decode_tokens = args.batch_size * args.decode_len
            total_tokens = prefill_tokens + decode_tokens
            
            print(f"Prefill FLOPs per iteration: {prefill_flops:.2e}")
            print(f"Decode FLOPs per iteration: {decode_flops:.2e}")
            print(f"Total FLOPs per iteration: {total_flops_per_iter:.2e}")
            print()
            print("Per-token FLOP breakdown:")
            print(f"  Prefill FLOPs per token: {prefill_flops / prefill_tokens:.2e}")
            print(f"  Decode FLOPs per token: {decode_flops / decode_tokens:.2e}")
            print(f"  Combined FLOPs per token: {total_flops_per_iter / total_tokens:.2e}")
            print()
            print(f"Prefill vs Decode FLOP ratio: {prefill_flops / decode_flops:.2f}:1")
            print(f"Prefill vs Decode per-token ratio: {(prefill_flops / prefill_tokens) / (decode_flops / decode_tokens):.2f}:1")
        
        # Print detailed FLOP summary
        print("\nDetailed FLOP breakdown:")
        print(flop_analyzer.get_flop_summary())
        
        # Export Chrome trace if requested
        if args.export_flop_trace:
            flop_analyzer.export_chrome_trace(args.export_flop_trace)
            print(f"FLOP profiler trace exported to: {args.export_flop_trace}")
        
        print("="*50)

    # Print power monitoring results
    if args.monitor_power and power_monitor:
        print("\n" + "="*50)
        print("GPU POWER USAGE ANALYSIS")
        print("="*50)
        
        if args.test_mode == "prefill" and prefill_power_stats:
            print(f"PREFILL PHASE POWER STATS:")
            print(f"  Duration: {prefill_power_stats['duration_seconds']:.3f} s")
            print(f"  Average Power: {prefill_power_stats['avg_power_watts']:.2f} W")
            print(f"  Peak Power: {prefill_power_stats['max_power_watts']:.2f} W")
            print(f"  Min Power: {prefill_power_stats['min_power_watts']:.2f} W")
            print(f"  Total Energy: {prefill_power_stats['total_energy_joules']:.2f} J")
            print(f"  Samples: {prefill_power_stats['num_samples']}")
            
            # Calculate per-iteration and per-token metrics
            energy_per_iter = prefill_power_stats['total_energy_joules'] / num_iters
            tokens_per_iter = args.batch_size * input.input_ids.shape[1]  # Total tokens across all batch items
            energy_per_token = energy_per_iter / tokens_per_iter  # Energy per individual token
            print(f"  Energy per iteration: {energy_per_iter:.3f} J")
            print(f"  Energy per token (prefill): {energy_per_token:.3f} J")
            
        elif args.test_mode == "decode" and decode_power_stats:
            print(f"DECODE PHASE POWER STATS:")
            print(f"  Duration: {decode_power_stats['duration_seconds']:.3f} s")
            print(f"  Average Power: {decode_power_stats['avg_power_watts']:.2f} W")
            print(f"  Peak Power: {decode_power_stats['max_power_watts']:.2f} W")
            print(f"  Min Power: {decode_power_stats['min_power_watts']:.2f} W")
            print(f"  Total Energy: {decode_power_stats['total_energy_joules']:.2f} J")
            print(f"  Samples: {decode_power_stats['num_samples']}")
            
            # Calculate per-iteration and per-token metrics
            energy_per_iter = decode_power_stats['total_energy_joules'] / num_iters
            tokens_per_iter = args.batch_size * args.decode_len  # Total tokens across all batch items
            energy_per_token = energy_per_iter / tokens_per_iter  # Energy per individual token
            print(f"  Energy per iteration: {energy_per_iter:.3f} J")
            print(f"  Energy per token (decode): {energy_per_token:.3f} J")
            
        elif args.test_mode == "whole":
            # Display separate prefill and decode stats
            if prefill_power_stats:
                print(f"PREFILL PHASE POWER STATS:")
                print(f"  Duration: {prefill_power_stats['duration_seconds']:.3f} s")
                print(f"  Average Power: {prefill_power_stats['avg_power_watts']:.2f} W")
                print(f"  Peak Power: {prefill_power_stats['max_power_watts']:.2f} W")
                print(f"  Min Power: {prefill_power_stats['min_power_watts']:.2f} W")
                print(f"  Total Energy: {prefill_power_stats['total_energy_joules']:.2f} J")
                print(f"  Samples: {prefill_power_stats['num_samples']}")
                
                # Calculate per-iteration and per-token metrics for prefill
                prefill_energy_per_iter = prefill_power_stats['total_energy_joules'] / num_iters
                prefill_tokens_per_iter = args.batch_size * input.input_ids.shape[1]  # Total tokens across all batch items
                prefill_energy_per_token = prefill_energy_per_iter / prefill_tokens_per_iter  # Energy per individual token
                print(f"  Energy per iteration: {prefill_energy_per_iter:.3f} J")
                print(f"  Energy per token (prefill): {prefill_energy_per_token:.3f} J")
                print()
            
            if decode_power_stats:
                print(f"DECODE PHASE POWER STATS:")
                print(f"  Duration: {decode_power_stats['duration_seconds']:.3f} s")
                print(f"  Average Power: {decode_power_stats['avg_power_watts']:.2f} W")
                print(f"  Peak Power: {decode_power_stats['max_power_watts']:.2f} W")
                print(f"  Min Power: {decode_power_stats['min_power_watts']:.2f} W")
                print(f"  Total Energy: {decode_power_stats['total_energy_joules']:.2f} J")
                print(f"  Samples: {decode_power_stats['num_samples']}")
                
                # Calculate per-iteration and per-token metrics for decode
                decode_energy_per_iter = decode_power_stats['total_energy_joules'] / num_iters
                decode_tokens_per_iter = args.batch_size * args.decode_len  # Total tokens across all batch items
                decode_energy_per_token = decode_energy_per_iter / decode_tokens_per_iter  # Energy per individual token
                print(f"  Energy per iteration: {decode_energy_per_iter:.3f} J")
                print(f"  Energy per token (decode): {decode_energy_per_token:.3f} J")
                print()
            
            # Display combined statistics
            if prefill_power_stats and decode_power_stats:
                total_energy = prefill_power_stats['total_energy_joules'] + decode_power_stats['total_energy_joules']
                total_duration = prefill_power_stats['duration_seconds'] + decode_power_stats['duration_seconds']
                avg_combined_power = total_energy / total_duration if total_duration > 0 else 0
                
                print(f"COMBINED SUMMARY:")
                print(f"  Total Duration: {total_duration:.3f} s")
                print(f"  Total Energy: {total_energy:.2f} J")
                print(f"  Average Combined Power: {avg_combined_power:.2f} W")
                print(f"  Energy per iteration (total): {total_energy / num_iters:.3f} J")
                
                # Combined energy per token calculation - accounts for both prefill and decode tokens
                total_tokens_per_iter = prefill_tokens_per_iter + decode_tokens_per_iter  # Total tokens across all batch items for both phases
                combined_energy_per_token = (total_energy / num_iters) / total_tokens_per_iter  # Energy per individual token (combined)
                print(f"  Energy per token (combined): {combined_energy_per_token:.3f} J")
                print(f"  Prefill vs Decode Energy Ratio: {prefill_power_stats['total_energy_joules'] / decode_power_stats['total_energy_joules']:.2f}:1")
        
        print("="*50)

    if args.record_memory:
        # Create the memory snapshot file
        export_memory_snapshot()

        # Stop recording memory snapshot history
        stop_record_memory_history()

if __name__ == "__main__":
    run_model(args.num_iters)