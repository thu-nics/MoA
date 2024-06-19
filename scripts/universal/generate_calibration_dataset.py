import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from multiprocess import set_start_method
from MoA.dataset.convert import multi_round_qa_to_multi_round_qa_model_by_batch, multi_round_qa_to_multi_round_conversation, context_reduction, multi_round_qa_to_length
import functools
import os
import pandas as pd
import argparse

"""
model
"""

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="lmsys/vicuna-7b-v1.5-16k")
parser.add_argument("--model_name", type=str, default="vicuna-7b-v1.5-16k")
parser.add_argument("--dataset_name", type=str, default="multi_news")
parser.add_argument("--output_path_base", type=str, default="local/dataset")
args = parser.parse_args()

model_path = args.model_path

model_name = args.model_name

"""
dataset
"""
dataset_name = args.dataset_name

huggingface_dataset_path = "fuvty/MoA_Human"
output_path_base = args.output_path_base

dataset_name_short_dict = {
    "microsoft_LCC_python": "lcc",
    "hotpot_qa": "hotpot_qa",
    "multi_news": "multi_news",
    "trec": "trec",
    "allenai_qasper": "qasper",
}

dataset_name_short = dataset_name_short_dict[dataset_name]

"""
prompt
"""
prompt = load_dataset(huggingface_dataset_path, "prompt")["train"]
df = prompt.to_pandas()
prompt_format = df[df["dataset_names"] == dataset_name_short]["prompt_format"].values[0]
question_format = df[df["dataset_names"] == dataset_name_short]["question_format"].values[0]
answer_format = df[df["dataset_names"] == dataset_name_short]["answer_format"].values[0]

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path,  padding_side='left', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# load model
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.eval()

# setups
max_length = 15500 # max length of input+output
batch_size = 1
max_dataitem = 10000
batched = False
save_json = True

get_model_response = functools.partial(
    multi_round_qa_to_multi_round_qa_model_by_batch,
    model=model,
    model_name=model_path,
    tokenizer=tokenizer,
    prompt_format=prompt_format,
    max_length=max_length,
    batch_size=batch_size,
)

if __name__ == "__main__":
    # the multi-round-qa dataset with human response
    multi_round_qa_model_path = f"{output_path_base}/multi_qa_model/{model_name}/{dataset_name}"
    multi_conversation_model_path = f"{output_path_base}/multi_conversation_model/{model_name}/{dataset_name}"

    # generate answers by model
    if True:
        # multi_round_qa_dataset = load_from_disk(multi_round_qa_human_dataset_path)
        multi_round_qa_dataset = load_dataset(huggingface_dataset_path)["train"].filter(lambda x: x["dataset"] == "multi_news")
        
        # filter to contain only length level <= 8
        multi_round_qa_dataset = multi_round_qa_dataset.filter(lambda x: x["total_length_level"] <= 8)

        if len(multi_round_qa_dataset) > max_dataitem:
            print(f"Warning: Dataset too long, truncate the dataset to {max_dataitem}")
            multi_round_qa_dataset = multi_round_qa_dataset.select(range(max_dataitem))

        gpu_num = torch.cuda.device_count()

        # example
        example = get_model_response(multi_round_qa_dataset[0], idx=0)
        print(example)

        # convert
        print("begin converting")
        print(prompt_format)
        print(question_format)
        print(answer_format)

        set_start_method("spawn")
        multi_round_qa_dataset = multi_round_qa_dataset.map(
            lambda x, idx: get_model_response(entry=x, idx=idx), # Now simply pass entry and idx
            batched=batched,
            batch_size=gpu_num if batched else None,
            with_rank=True,
            num_proc=gpu_num,
        )

        print(multi_round_qa_dataset)

        # save to file
        if not os.path.exists(multi_round_qa_model_path):
            os.makedirs(multi_round_qa_model_path, exist_ok=True)

        multi_round_qa_dataset.save_to_disk(multi_round_qa_model_path)
        # save the json version to review
        if save_json:
            multi_round_qa_dataset.to_json(os.path.join(multi_round_qa_model_path, "text.json"))

    # convert to multi_conversation
    if True:
        multi_round_qa_dataset = load_from_disk(multi_round_qa_model_path)

        # context reduction
        if dataset_name == "trec":
            multi_round_qa_dataset = multi_round_qa_dataset.map(
                lambda x: context_reduction(
                    entry=x,
                    tokenizer=tokenizer,
                    expected_total_length=int(((x["total_length_level"]-1) // 2 + 1)*2048 * 0.81) - 8,
                ),
            )
        else:
            multi_round_qa_dataset = multi_round_qa_dataset.map(
                lambda x: context_reduction(
                    entry=x,
                    tokenizer=tokenizer,
                    expected_total_length=x["total_length_level"]*1024 - x["reserve_length"] - 8,
                ),
            )

        # re-calculate length
        multi_round_qa_dataset = multi_round_qa_dataset.map(
            lambda x: multi_round_qa_to_length(
                entry=x,
                tokenizer=tokenizer,
                prompt_format=prompt_format,
                question_format=question_format,
                answer_format=answer_format,
            ),
        )

        # convert to multi-round conversation
        example = multi_round_qa_to_multi_round_conversation(
            entry=multi_round_qa_dataset[0],
            model_name=model_path,
            tokenizer=tokenizer,
            prompt_format=prompt_format,
            question_format=question_format,
            answer_format=answer_format,
        )
        print(example['text'])

        multi_round_conversation_dataset = multi_round_qa_dataset.map(
            lambda x: multi_round_qa_to_multi_round_conversation(
                entry=x,
                model_name=model_path,
                tokenizer=tokenizer,
                prompt_format=prompt_format,
                question_format=question_format,
                answer_format=answer_format,
            ),
            remove_columns=[c for c in multi_round_qa_dataset.column_names if c not in ["dataset", "total_length_level", "truncate", "total_length"]],
        )
        # output columns: ['text', 'model_name', 'input_length', 'answer_length']

        # save to file
        if not os.path.exists(multi_conversation_model_path):
            os.makedirs(multi_conversation_model_path, exist_ok=True)

        multi_round_conversation_dataset.save_to_disk(multi_conversation_model_path)
        if save_json:
            multi_round_conversation_dataset.to_json(os.path.join(multi_conversation_model_path, "text.json"))

        # convert the input_length and answer_length to a DataFrame
        df = pd.DataFrame(multi_round_conversation_dataset)
        columns = multi_round_conversation_dataset.column_names
        # remove text from columns
        columns.remove("text")
        df = df[columns]
        df.to_csv(os.path.join(multi_conversation_model_path, "lengths.csv"), index=False)