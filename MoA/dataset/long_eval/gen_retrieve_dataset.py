"""
Most of the code in this file is adapted from the original codebase of the repo:
qllm_eval
"""

import random
import itertools
import uuid
import re
from datasets import Dataset
from collections import defaultdict
import os
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
from MoA.dataset.long_eval.convert import to_question, get_tokenized_len
import numpy as np

def retrieve_expected(lines, random_line_pos):
    correct_line = lines[random_line_pos]
    expected_number = re.search("<\d+>", correct_line)
    if expected_number is not None:
        expected_number = int(expected_number.group()[1:-1])
    else:
        print(f"Got unparsable line: {correct_line}")

    return expected_number, correct_line

def generate_line_index(num_line, idx_opt):
    if idx_opt == "LRT-ABCindex":
        ingredients = ["A", "B", "C", "D", "E", "F"]

        start = 6
        comb = list(itertools.product(ingredients, repeat=start))
        while len(comb) < num_line:
            start += 1
            comb = list(itertools.product(ingredients, repeat=start))
        
        comb = ["".join(i) for i in comb]

        return comb[:num_line]
    elif idx_opt == "LRT-UUID":
        comb = []
        for i in range(num_line):
            comb.append(str(uuid.uuid4()))
        
        return comb
    elif idx_opt == "LRT-NL":
        import wonderwords

        w = wonderwords.RandomWord()
        adjs = w.random_words(num_line, include_categories=["adjective", "verb", "noun"])
        nouns = w.random_words(num_line, include_categories=["noun"])

        comb = []
        for i, (adj, noun) in enumerate(zip(adjs, nouns)):
            comb.append(f"{adj}-{noun}")
        
        return comb

def generate_lines_dataset(cfgs: dict) -> Dataset:
    """
    Generate a dataset for the task of memorizing the content of a line
    Input:
    cfgs: dict
        A dictionary containing the configuration of the dataset, including
        - num_lines: list, the number of lines in the record
        - num_test_samples: int, the number of test samples
        - line_idx_opt: str, the option of generating the line index
    Output:
    dataset: Dataset
        A dataset for the task of memorizing the content of a line
        - random_idx: str, the index str of the retrieved line, for example, "rapid-handgun".
        - random_num: int, the index number of the retrieved line, for example, 51. The range is between 0 and num_lines-1.
        - num_lines: int, the number of lines in the record
        - expected_number: int, the number of the retrieved line, for example, 31898.
        - correct_line: str, the content of the retrieved line, for example, "line rapid-handgun: REGISTER_CONTENT is <31898>\n"
        - text: str, the content of the record
    """
    output = defaultdict(list)

    for n in tqdm(cfgs["num_lines"]):
        for i in range(cfgs["num_test_samples"]):     
            lines = []

            if cfgs["line_idx_opt"] == "LRT":
                line_idxes = list(range(1, n + 1))
                lines.extend([f"line {i}: REGISTER_CONTENT is <{random.randint(1, 50000)}>\n" for i in line_idxes])
                random_idx = random.randint(1, n)
                random_num = random_idx - 1
            else:
                line_idxes = generate_line_index(n, cfgs["line_idx_opt"])
                lines.extend([f"line {i}: REGISTER_CONTENT is <{random.randint(1, 50000)}>\n" for i in line_idxes])
                random_num = random.randint(0, len(line_idxes)-1)
                random_idx = line_idxes[random_num]

            expected_number, correct_line = retrieve_expected(lines, random_num)

            text = ""
            for l in lines:
                text += l

            output["key_str"].append(random_idx) # the index str of the correct line, e.g., "rapid-handgun"
            output["key_id"].append(random_num) # the index of the correct line, e.g., 7
            output["value"].append(expected_number) # the number of the correct line, e.g., 31898
            output["correct_line"].append(correct_line) # the content of the correct line, e.g., "line rapid-handgun: REGISTER_CONTENT is <31898>\n"
            output["content"].append(text) # the content of the record
            output["num_lines"].append(n) # the number of lines in the record, e.g., 256
            
    dataset = Dataset.from_dict(output)
    return dataset

if __name__ == "__main__":
    # line_range = range(16, 705, 16)
    # line_range = range(16, 35, 16)
    # line_range = range(2040, 2150, 32)
    # line_range = range(24, 48, 16)

    parser = argparse.ArgumentParser(description="Generate dataset for long evaluation")
    parser.add_argument(
        "--length_level",
        type=int,
        help="the length of dataset in unit K.",
        default=16,
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="the path to save the generated dataset",
        default="local/universal/dataset/longeval_lines_multiple",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="the model name for the tokenizer",
        default="gradientai/Llama-3-8B-Instruct-262k",
    )

    parser.add_argument(
        "--init_num_lines",
        type=int,
        help="the initial number of lines",
        default=None,
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)

    cfgs = {
        "task": "lines",
        "num_test_samples": 40,
        "num_lines": [],
        "line_idx_opt": "LRT-NL"
    }

    test_cfgs = {
        "task": "lines",
        "num_test_samples": 20,
        "num_lines": [],
        "line_idx_opt": "LRT-NL"
    }

    length_level = args.length_level
    if not args.init_num_lines:
        start = length_level * 1024 // 13
    else:
        start = args.init_num_lines

    output_path = os.path.join(args.output_path, f"longeval-{length_level}k")

    # find the best start
    while True:
        test_line_range = range(start, start + 64, 32)
        test_cfgs["num_lines"] = [i for i in test_line_range]
        test_dataset = generate_lines_dataset(test_cfgs)
        retrieve_dataset = test_dataset.map(lambda data: to_question(data)) # columns: key_id, key_str, value, content, correct_line, num_lines, question
        retrieve_dataset = retrieve_dataset.map(lambda data: get_tokenized_len(data, tokenizer)) # columns: key_id, key_str, value, content, correct_line, num_lines, question, tokenized_len
        test_length = np.mean(retrieve_dataset["tokenized_len"])
        test_length_level = int(test_length - 1) // 1024 + 1
        print(f"test length level: {test_length_level}")
        if test_length_level == length_level:
            break
        elif test_length_level < length_level:
            start += 32
        else:
            start -= 32
        print(f"adjusting start to {start}...")

    cfgs["num_lines"] = [i for i in range(start - 64, start + 96, 32)]

    retrieve_dataset: Dataset = generate_lines_dataset(cfgs)

    retrieve_dataset = retrieve_dataset.map(lambda data: to_question(data)) # columns: key_id, key_str, value, content, correct_line, num_lines, question
    retrieve_dataset = retrieve_dataset.map(lambda data: get_tokenized_len(data, tokenizer)) # columns: key_id, key_str, value, content, correct_line, num_lines, question, tokenized_len
    num = 0
    for data in retrieve_dataset:
        sample_length = (data['tokenized_len'])
        sample_length_level = (sample_length - 1) // 1024 + 1
        if sample_length_level == length_level:
            num += 1

    assert num > 100, f"num of samples: {num}"
    print("save to disk of path: ", output_path)

    # save data
    retrieve_dataset.save_to_disk(output_path)
    key_id_df = retrieve_dataset.to_pandas()[["key_id", "num_lines"]].to_csv(os.path.join(output_path, "key_id.csv"), index=False)
