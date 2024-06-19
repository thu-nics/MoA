from datasets import load_dataset
import torch
from tqdm import tqdm
import numpy as np
import random

from fastchat.model import get_conversation_template

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
        stop_token_ids = tokenizer.eos_token_id
    else:
        conv = get_conversation_template(model_name)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
    return prompt

def build_stop_token(model_name, tokenizer):
    conv = get_conversation_template(model_name)
    if conv.stop_token_ids is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer does not have eos_token_id")
        else:
            return [tokenizer.eos_token_id]
    else:
        return conv.stop_token_ids  

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, length_range='all'):
    preds = []
    if length_range == 'all':
        pass
    elif length_range == '0-4k':
        data = [d for d in data if d["length"] < 4000]
    elif length_range == '4-8k':
        data = [d for d in data if d["length"] >= 4000 and d["length"] < 8000]
    elif length_range == '8k+':
        data = [d for d in data if d["length"] >= 8000]
    else:
        raise ValueError("Invalid length range")

    stop_token_ids = build_stop_token(model_name, tokenizer)

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        print(f"input id length: {input.input_ids.shape[-1]}")
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            stop_token_ids = [tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]] if stop_token_ids is None else stop_token_ids+[tokenizer.encode("\n", add_special_tokens=False)[-1]]
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=stop_token_ids,
            )[0]
        else:
            try:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    eos_token_id=stop_token_ids
                )[0]
            except Exception as e:
                print(e)
                # if generate fails, return empty output result
                output = torch.tensor([tokenizer.eos_token_id] * (context_length + 1)).to(device)
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
    return preds

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
