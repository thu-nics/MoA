import os
import argparse
from datasets import load_dataset
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from typing import Tuple

from MoA.models.llama.modeling_llama import LlamaModel_use_streamingllm_attention
from MoA.models.llama.h2o import convert_kvcache_llama_heavy_recent

from MoA.evaluation.LongBench.eval import scorer, scorer_e
from MoA.models.interface import update_model_function

# set cuda visible device
# triton implementation ONLY support single GPU inference
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Evaluation")
    parser.add_argument(
        "--model_name", required=True, type=str, help="Path of the model"
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default=None
    )
    parser.add_argument("--output_file", default=None, type=str, help="Output filename")
    parser.add_argument(
        "--block_size", default=64, type=int, help="Block size of the attention map"
    )
    parser.add_argument(
        "--streamingllm", action="store_true", default=False, help="Whether to use streamingllm"
    )
    parser.add_argument(
        "--band_size", default=1024, type=int, help="Band size of the attention map"
    )
    parser.add_argument(
        "--global_size", default=64, type=int, help="Global size of the attention map"
    )
    parser.add_argument(
        "--moa_config",
        type=str,
        default=None,
        help="the path to moa configuration file",
    )
    parser.add_argument(
        '--h2o', action='store_true', help='Whether to use h2o'
    )
    parser.add_argument(
        '--heavy', type=int, default=1024
    )
    parser.add_argument(
        '--recent', type=int, default=1024
    )
    
    # tokenize related
    parser.add_argument(
        "--max_length", default=2048, type=int, help="Maximum token length"
    )
    parser.add_argument(
        "--padding",
        default="longest",
        type=str,
        choices=["do_not_pad", "longest", "max_length"],
        help="Padding strategy, choose from do_not_pad, longest, max_length",
    )

    # flash attention
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        default=False,
        help="Whether to use flash attention",
    )

    # evaluation
    parser.add_argument(
        "--dataset", type=str, default="lambada", help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--subset", type=str, default=None, help="Subset of the dataset to evaluate on"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default=None,
        help="Split of the dataset to evaluate on, e.g., train[:10]",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size for evaluation"
    )
    parser.add_argument(
        "--chunk",
        action="store_true",
        default=False,
        help="Whether to chunk the long sentence in dataset into multiple sentences",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default="longbench_fast",
        choices=[
            "longbench",
            "longbench_fast",
        ],
        help="Which evaluation method to use.",
    )

    parser.add_argument(
        "--shuffle_dataset",
        action="store_true",
        default=False,
        help="Whether to shuffle the dataset",
    )
    parser.add_argument(
        "--shuffle_seed", type=int, default=42, help="Seed for shuffling the dataset"
    )
    parser.add_argument(
        "--select_dataset_range",
        type=int,
        nargs="+",
        default=None,
        help="Select a subset of the dataset for evaluation, enter two numbers as the range, enter one number as the number of examples to select",
    )
    parser.add_argument(
        "--rows_to_one",
        action="store_true",
        default=False,
        help="Concatenate all rows in the dataset into one long string",
    )
    parser.add_argument(
        "--ignore_last_batch",
        action="store_true",
        default=False,
        help="Ignore the last batch if it is smaller than batch size",
    )
    parser.add_argument(
        "--rename_column",
        type=str,
        default=None,
        help="Rename the column of the dataset to 'text'",
    )
    parser.add_argument(
        "--load_from_disk",
        action="store_true",
        default=False,
        help="Whether to load the dataset from disk",
    )

    # longbench
    parser.add_argument(
        "--longbench_e", action="store_true", help="Evaluate on LongBench-E"
    )

    parser.add_argument(
        "--evaluation_dataset", type=str, nargs='+', default=None,
    )

    parser.add_argument(
        "--longbench_result_dir", type=str, default=None, help="Path to save the evaluation result",
    )

    parser.add_argument(
        "--longbench_length_range", type=str, choices=["all", "0-4k", "4-8k", "8k+"], default="all",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    truncation = True
    last_words = False

    args = parse_args()

    print(args)

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
        config._attn_implementation_internal = "eager"

    print(f"using {config._attn_implementation_internal} attention implementation")

    # load model
    if args.moa_config is not None:
        attn_implementation = "sdpa"
    elif args.use_flash_attention:
        attn_implementation = "flash_attention_2"
    elif args.streamingllm:
        attn_implementation = "eager"
    elif args.h2o:
        attn_implementation = "eager"
    else:
        attn_implementation = "sdpa"
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation=attn_implementation,
    ).eval()

    if args.streamingllm:
        LlamaModel_use_streamingllm_attention(model.model, args.global_size, args.band_size)
        print(f"using streamingllm, global size: {args.global_size}, band size: {args.band_size}")

    if args.h2o:
        model = convert_kvcache_llama_heavy_recent(model, args.heavy, args.recent)
        print(f"using h2o, heavy: {args.heavy}, recent: {args.recent}")

    if args.moa_config is not None:
        moa_config_path = args.moa_config
        with open(moa_config_path, 'r') as f:
            moa_config = json.load(f)
        # Add mixture of sparse attention capability to the model
        model = update_model_function(model, args.model_name)
        model.model.set_mixture_of_attention(moa_config, permute_head=True)

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id


    # evaluate with LongBench
    if args.eval == "longbench" or args.eval == "longbench_fast":
        from MoA.evaluation.LongBench.pred import seed_everything, get_pred

        ### copy from LongBench ###
        seed_everything(42)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = args.model_name
        max_length = args.max_length

        print("max_length", max_length)

        if args.longbench_e:
            datasets = [
                "qasper",
                "multifieldqa_en",
                "hotpotqa",
                "2wikimqa",
                "gov_report",
                "multi_news",
                "trec",
                "triviaqa",
                "samsum",
                "passage_count",
                "passage_retrieval_en",
                "lcc",
                "repobench-p",
            ]

            if args.evaluation_dataset is not None:
                datasets = args.evaluation_dataset
        else:
            if args.eval == "longbench_fast":
                datasets = [
                    "multifieldqa_en",
                    "2wikimqa",
                    "lcc",
                    "samsum",
                    "multi_news",
                ]

                if args.evaluation_dataset is not None:
                    datasets = args.evaluation_dataset
            else:
                datasets = [
                    "narrativeqa",
                    "qasper",
                    "multifieldqa_en",
                    "multifieldqa_zh",
                    "hotpotqa",
                    "2wikimqa",
                    "musique",
                    "dureader",
                    "gov_report",
                    "qmsum",
                    "multi_news",
                    "vcsum",
                    "trec",
                    "triviaqa",
                    "samsum",
                    "lsht",
                    "passage_count",
                    "passage_retrieval_en",
                    "passage_retrieval_zh",
                    "lcc",
                    "repobench-p",
                ]

                if args.evaluation_dataset is not None:
                    datasets = args.evaluation_dataset
        
        print(f"evaluating on {datasets}")
        
        # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
        dataset2prompt = json.load(
            open("data/LongBench/config/dataset2prompt.json", "r")
        )
        dataset2maxlen = json.load(
            open("data/LongBench/config/dataset2maxlen.json", "r")
        )
        # predict on each dataset
        result_dir = args.longbench_result_dir
        pred_dir = os.path.join(result_dir, "longbench/pred", args.longbench_length_range)
        pred_e_dir = os.path.join(result_dir, "longbench/pred_e", args.longbench_length_range)

        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(pred_e_dir, exist_ok=True)

        base_model_name = model_name.rstrip("/").split("/")[-1]
        print(f"base_model_name: {base_model_name}")

        for dataset in datasets:
            if args.longbench_e:
                data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
                os.makedirs(os.path.join(pred_e_dir, base_model_name), exist_ok=True)
                out_path = os.path.join(pred_e_dir, base_model_name, f"{dataset}.jsonl")
            else:
                data = load_dataset("THUDM/LongBench", dataset, split="test")
                os.makedirs(os.path.join(pred_dir, base_model_name), exist_ok=True)
                out_path = os.path.join(pred_dir, base_model_name, f"{dataset}.jsonl")

            prompt_format = dataset2prompt[dataset]
            max_gen = dataset2maxlen[dataset]
            preds = get_pred(
                model,
                tokenizer,
                data,
                max_length,
                max_gen,
                prompt_format,
                dataset,
                device,
                model_name,
                length_range=args.longbench_length_range,
            )

            with open(out_path, "w", encoding="utf-8") as f:
                for pred in preds:
                    json.dump(pred, f, ensure_ascii=False)
                    f.write("\n")

        # evaluate the result
        scores = dict()
        if args.longbench_e:
            path = os.path.join(pred_e_dir, base_model_name)
        else:
            path = os.path.join(pred_dir, base_model_name)
        all_files = os.listdir(path)
        print("Evaluating on:", all_files)
        for filename in all_files:
            if not filename.endswith("jsonl"):
                continue
            predictions, answers, lengths = [], [], []
            dataset = filename.split(".")[0]
            with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    predictions.append(data["pred"])
                    answers.append(data["answers"])
                    all_classes = data["all_classes"]
                    if "length" in data:
                        lengths.append(data["length"])
            if args.longbench_e:
                score = scorer_e(dataset, predictions, answers, lengths, all_classes, args.longbench_length_range)
            else:
                score = scorer(dataset, predictions, answers, all_classes)
            scores[dataset] = score
        if args.longbench_e:
            out_path = os.path.join(pred_e_dir, base_model_name, "result.json")
        else:
            out_path = os.path.join(pred_dir, base_model_name, "result.json")
        with open(out_path, "w") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)