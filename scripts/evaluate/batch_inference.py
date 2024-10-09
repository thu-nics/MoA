import argparse
import torch
import functools
import json
from typing import List, Dict
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_from_disk, load_dataset, Dataset
from fastchat.model import get_conversation_template

from transformers.models.llama import LlamaForCausalLM

from MoA.models.interface import update_model_function
from MoA.attention.set import set_static_attention_lut
from MoA.dataset.convert import multi_round_qa_to_multi_round_qa_model_by_batch

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

# Input args
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='lmsys/vicuna-7b-v1.5-16k', help='model name')
parser.add_argument('--moa_config', type=str, default=None, help='the path to moa configuration file')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--max_new_tokens', type=int, default=512, help='max new tokens')
parser.add_argument('--dataset_name', type=str, default='multi_news', help='dataset name')
args = parser.parse_args()


def batch_inference(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Dataset, batch_size: int, model_name: str, prompt_format: str = None, max_length: int = 16384, max_new_tokens: int = 256, verbose: bool = False):

    def collate_first_items_in_list(batch: List[Dict]):
        for entry in batch:
            for key, value in entry.items():
                if isinstance(value, list):
                    entry[key] = value[0]

        return default_collate(batch)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_first_items_in_list)

    questions = []
    answers = []
    
    progress_bar = tqdm(data_loader, desc="Batch Inference", total=len(data_loader))

    for i, batch in enumerate(data_loader):
        progress_bar.update(1)

        context_batch: list = batch["context"]
        # question_batch: list = batch["questions"][0] # take the first question of each context
        question_batch: list = batch["questions"]
        model_input_batch: list = []

        # if verbose:
        #     for question in question_batch:
        #         print(question)


        for context, question in zip(context_batch, question_batch):

            if prompt_format is not None:
                user_input = prompt_format.format(context=context, question=question)
            else:
                user_input = context + question

            conv = get_conversation_template(model_name)
            conv.append_message(conv.roles[0], user_input)
            conv.append_message(conv.roles[1], None)
            stop_token_ids = conv.stop_token_ids

            model_input = conv.get_prompt()

            model_input_batch.append(model_input)

        # batch inference
        model_input = tokenizer(model_input_batch, return_tensors="pt", padding=True).to(model.device)
        input_lengths = torch.sum(model_input.attention_mask, dim=1)

        if max(input_lengths) > max_length:
            break

        # try:
        if True:
            # Generate the user input and model response
            model_responses = model.generate(
                **model_input,
                max_new_tokens=max_new_tokens,
                eos_token_id=stop_token_ids,
                # do_sample=True,
                # temperature=0.6,
                # top_p=0.9,
                num_return_sequences=1,
            ) # return the first generated sequence of shape [input, length]
            model_responses = model_responses[..., model_input.input_ids.shape[-1]:] # left padding
            model_response_lengths = torch.sum(model_responses!=tokenizer.pad_token_id, dim=-1)
            model_responses = tokenizer.batch_decode(model_responses,  skip_special_tokens=True)
        # except Exception as e:
        #     print(e)
        #     model_responses = [""] * len(model_input)

        if verbose:
            for question, answer in zip(question_batch, model_responses):
                # print(question)
                # print(answer)
                # print()
                progress_bar.write(f"Question: {question}")
                progress_bar.write(f"Answer: {answer}")
                progress_bar.write("\n")

        answers.extend(model_responses)
        questions.extend(question_batch)

    return {'questions': questions, 'answers': answers}
    

if __name__ == "__main__":
    # Load the huggingface model
    model_name = args.model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)

    if args.moa_config is not None:
        moa_config_path = args.moa_config
        with open(moa_config_path, 'r') as f:
            moa_config = json.load(f)
        # Add mixture of sparse attention capability to the model
        model = update_model_function(model, model_name)
        model.model.set_mixture_of_attention(moa_config, permute_head=True)

    # Batch Inference
    huggingface_dataset_path = "nics-efc/MoA_Long_HumanQA"
    dataset_name_short = args.dataset_name
    max_dataitem = 8

    batch_size = args.batch_size

    prompt = load_dataset(huggingface_dataset_path, "prompt")["train"]
    df = prompt.to_pandas()
    prompt_format = df[df["dataset_names"] == dataset_name_short]["prompt_format"].values[0]
    question_format = df[df["dataset_names"] == dataset_name_short]["question_format"].values[0]
    answer_format = df[df["dataset_names"] == dataset_name_short]["answer_format"].values[0]

    multi_round_qa_dataset = load_dataset(huggingface_dataset_path)["train"].filter(lambda x: x["dataset"] == dataset_name_short).filter(lambda x: x["total_length_level"] <= 8).select(range(max_dataitem))

    with torch.no_grad():
        batch_inference(model, tokenizer, multi_round_qa_dataset, batch_size, model_name, prompt_format, max_length=16384, max_new_tokens=args.max_new_tokens, verbose=True)