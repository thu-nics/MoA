"""
Convert all kinds of dataset to multi-round QA format.
"""

# Convert the dataset to multi-round qa format.
# with the following columns:
# - dataset: the name of the dataset
# - questions: a list of questions
# - answers: a list of answers
# - evidences: a list of evidence sections
# - context: the context to answer the question, in markdown format
# - summary: (optional) the summary of the context

from multiprocessing import context
from gurobipy import max_
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, concatenate_datasets
import pandas as pd
from typing import Dict, Optional, Union, List
from fastchat.model import get_conversation_template
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import random
import re
from itertools import islice

from MoA import attention

def qasper_to_multi_round_qa(entry: Dict) -> dict:
    """
    Data name: allenai/qasper
    Data source: https://huggingface.co/datasets/allenai/qasper
    Data structure of the input dataset:
        {
            'id': "Paper ID (string)",
            'title': "Paper Title",
            'abstract': "paper abstract ...",
            'full_text': {
                'paragraphs':[["section1_paragraph1_text","section1_paragraph2_text",...],["section2_paragraph1_text","section2_paragraph2_text",...]],
                'section_name':["section1_title","section2_title"],...},
            'qas': {
            'answers':[{
                'annotation_id': ["q1_answer1_annotation_id","q1_answer2_annotation_id"]
                'answer': [{
                    'unanswerable':False,
                    'extractive_spans':["q1_answer1_extractive_span1","q1_answer1_extractive_span2"],
                    'yes_no':False,
                    'free_form_answer':"q1_answer1",
                    'evidence':["q1_answer1_evidence1","q1_answer1_evidence2",..],
                    'highlighted_evidence':["q1_answer1_highlighted_evidence1","q1_answer1_highlighted_evidence2",..]
                    },
                    {
                    'unanswerable':False,
                    'extractive_spans':["q1_answer2_extractive_span1","q1_answer2_extractive_span2"],
                    'yes_no':False,
                    'free_form_answer':"q1_answer2",
                    'evidence':["q1_answer2_evidence1","q1_answer2_evidence2",..],
                    'highlighted_evidence':["q1_answer2_highlighted_evidence1","q1_answer2_highlighted_evidence2",..]
                    }],
                'worker_id':["q1_answer1_worker_id","q1_answer2_worker_id"]
                },{...["question2's answers"]..},{...["question3's answers"]..}],
            'question':["question1","question2","question3"...],
            'question_id':["question1_id","question2_id","question3_id"...],
            'question_writer':["question1_writer_id","question2_writer_id","question3_writer_id"...],
            'nlp_background':["question1_writer_nlp_background","question2_writer_nlp_background",...],
            'topic_background':["question1_writer_topic_background","question2_writer_topic_background",...],
            'paper_read': ["question1_writer_paper_read_status","question2_writer_paper_read_status",...],
            'search_query':["question1_search_query","question2_search_query","question3_search_query"...],
        }
    """
    title = entry["title"]
    abstract = entry["abstract"]
    full_text = entry["full_text"]
    qas = entry["qas"]

    question_list = []
    answer_lists = []
    evidence_list = []

    for question, answer_data in zip(qas["question"], qas["answers"]):
        for ans in answer_data["answer"]:
            if ans["unanswerable"]:
                answer = "This question is unanswerable based on the provided context."
            elif ans["free_form_answer"]:
                answer = ans["free_form_answer"]
            elif ans["extractive_spans"]:
                answer = " ".join(ans["extractive_spans"])
            else:
                answer = "No answer provided."

            evidence = "\n\n".join(ans["evidence"]) if ans["evidence"] else ""

            question_list.append(question)
            answer_lists.append(answer)
            evidence_list.append(evidence)

    # Construct the context and summary from provided information
    # Assuming abstract, title, or full text are given and relevant
    # In this example, context and summary details are not specified, so placeholders are used
    context_md = (
        "Placeholder for combined context from the abstract and relevant sections."
    )

    title = entry["title"]
    abstract = entry["abstract"]
    full_text = entry["full_text"]

    # Markdown formatted context
    context_md = f"# {title}\n\n## Abstract\n\n{abstract}\n\n"
    for idx, section in enumerate(full_text["section_name"]):
        paragraphs = "\n\n".join(full_text["paragraphs"][idx])
        context_md += f"## {section}\n\n{paragraphs}\n\n"

    # Optionally create a summary from the context, simple example using abstract
    summary = abstract

    # Assuming each paper corresponds to a single entry with multiple questions and answers
    return {
        "dataset": "qasper",
        "questions": question_list,
        "answers": answer_lists,
        "evidences": evidence_list,
        "context": context_md,
        "summary": summary,
    }

def multiNews_to_multi_round_qa(entry: Dict) -> dict:
    """
    Data name: multi_news
    Data source: https://huggingface.co/datasets/multi_news
    Data structure of the input dataset:
        {
            "document": "some line val \n another line",
            "summary": "target val line"
        }
    """
    document = entry["document"]
    summary = entry["summary"].strip("– ") # Remove leading dash and space

    # Return the structured data
    return {
        "dataset": "multi_news",
        "questions": [""],
        "answers": [summary],
        "evidences": [""],
        "context": document,
        "summary": summary,
    }

def hotpotQA_to_multi_round_qa(entry: Dict) -> dict:
    """
    Data name: hotpot_qa
    Data source: https://huggingface.co/datasets/hotpot_qa
    Data structure of the input dataset:
        {
            "answer": "This is the answer",
            "context": {
                "sentences": [["Sent 1"], ["Sent 21", "Sent 22"]],
                "title": ["Title1", "Title 2"]
            },
            "id": "000001",
            "level": "medium",
            "question": "What is the answer?",
            "supporting_facts": {
                "sent_id": [0, 1, 3],
                "title": ["Title of para 1", "Title of para 2", "Title of para 3"]
            },
            "type": "comparison"
        }
    """
    # Extracting elements from the entry
    question = entry["question"]
    answer = entry["answer"]
    context_info = entry["context"]
    supporting_facts = entry["supporting_facts"]

    # Building the context and evidence
    context_md = ""
    evidence_texts = []
    
    # Loop through each context sentence and its title
    for sentences, title in zip(context_info["sentences"], context_info["title"]):
        context_md += f"## {title}\n" + "\n".join(sentences) + "\n\n"
        
        # Collect evidence based on supporting facts
        for fact_title, sent_id in zip(supporting_facts["title"], supporting_facts["sent_id"]):
            if title == fact_title and sent_id < len(sentences):
                evidence_texts.append(sentences[sent_id])

    # Create a markdown section for the evidences
    evidence_md = "\n\n".join(evidence_texts)

    # Summarizing the context (optional and simplistic, can be improved)
    summary = "\n".join(supporting_facts['title'])
    
    # Format the output as specified
    return {
        "dataset": "hotpot_qa",
        "questions": [question],
        "answers": [answer],
        "evidences": [evidence_md],
        "context": context_md,
        "summary": summary,
    }

def lcc_to_multi_round_qa(entry: Dict, num_complete_line = 10) -> dict:
    """
    Data name: lcc
    Data source: https://huggingface.co/datasets/microsoft/LCC_python
    Data structure of the input dataset:
        {
            "context": "The python code"
        }
    Convert the python data to multi-round QA format as code complation problem. First seperate the code to multiple lines, then randomly select some lines as the complete line. The question is the complete line and the answer is the complete line. The context is the code before the first complete line.
    """
    lines = entry["context"].split("\n")
    
    # random split lines to num_complete_line segements
    complete_line_pos_list = random.sample([i for i in range(len(lines))], num_complete_line)
    complete_line_pos_list.sort()
    complete_line_pos_list.insert(0, 0)

    questions = []
    answers = []
    evidences = []

    for start_pos, end_pos in zip(complete_line_pos_list[:-1], complete_line_pos_list[1:]):
        question = "\n".join(lines[start_pos: end_pos])
        answer = lines[end_pos]
        questions.append(question)
        answers.append(answer)
        evidences.append(lines[max(0, end_pos-1)])

    summary = ""

    return {
        "dataset": "lcc",
        "questions": questions,
        "answers": answers,
        "evidences": evidences,
        "context": "",
        "summary": summary,
    }


def longbench_to_multi_round_qa(entry: Dict) -> dict:
    """
    Data name: longbench
    Data source: https://huggingface.co/datasets/THUDM/LongBench
    Data structure of the input dataset:
        {
            "input": "The input/command for the task, usually short, such as questions in QA, queries in Few-shot tasks, etc",
            "context": "The long context required for the task, such as documents, cross-file code, few-shot examples in Few-shot tasks",
            "answers": "A List of all true answers",
            "length": "Total length of the first three items (counted in characters for Chinese and words for English)",
            "dataset": "The name of the dataset to which this piece of data belongs",
            "language": "The language of this piece of data",
            "all_classes": "All categories in classification tasks, null for non-classification tasks",
            "_id": "Random id for each piece of data"
        }
    """
    print("noqa")

    input_command = entry["input"]
    context = entry["context"]
    answers = entry["answers"]  # Assuming this is a list of answers
    dataset_name = entry["dataset"]

    # Since 'evidences' are not provided in 'longbench', we'll leave it as empty placeholders.
    # In a real application, you might generate these based on the context or include relevant sections from the context.
    evidences = [""] * len(answers)  # Create a placeholder list of the same length as answers

    summary = ""
    return {
        "dataset": dataset_name,
        "questions": [input_command] * len(answers),  # Assuming 'input' is akin to a question
        "answers": answers,  # Assuming this is a list of answers; adjust if it's a single answer
        "evidences": evidences,
        "context": context,
        "summary": summary,
    }

def longeval_to_multi_round_qa(entry: Dict, num_questions: int = 5) -> dict:
    """
    Data name: qeval
    Data source: https://huggingface.co/datasets/qeval
    Data structure of the input dataset:
        {
            "key_str": The key string,
            "key_id": The key ID,
            "value": The value,
            "correct_line": The correct line,
            "content": The long_eval content,
            "num_lines": The number of total lines,
        }
    """
    content = entry["content"]
    num_lines = entry["num_lines"]

    # Assuming the content is structured with lines that should be memorized
    content_list = content.split("\n")[:-1]  # Exclude the last empty entry if present
    assert len(content_list) == num_lines, "Number of lines does not match num_lines"

    questions = []
    answers = []
    evidences = []

    for _ in range(num_questions):
        # Randomly select a line to question about
        line_index = random.randint(0, num_lines - 1)
        line = content_list[line_index]

        # Extract the key string and value
        key_str = re.search(r"line (.+?):", line).group(1)
        value = re.search(r"REGISTER_CONTENT is <(.+?)>", line).group(1)

        # question = f"What is the <REGISTER_CONTENT> in line {key_str}?"
        question = key_str
        answer = value
        evidence = line  # Use the line itself as evidence

        questions.append(question)
        answers.append(answer)
        evidences.append(evidence)

    summary = ""

    return {
        "dataset": "longeval",
        "questions": questions,
        "answers": answers,
        "evidences": evidences,
        "context": content,
        "summary": summary,
    }

def trec_to_qa(entry: Dict) -> dict:
    """
    Data name: trec
    Data source: https://huggingface.co/datasets/trec
    Data structure of the input dataset:
        {
            'text': 'How did serfdom develop in and then leave Russia ?',
            'coarse_label': 2,
            'fine_label': 26
        }
    We use the 'text' as the questions and the 'fine_label' as the answer. The context is few shot examples of questions and anwers with format "Question: {question} Type: {answer}."
    """
    label_description_dict = {
        0: "Abbreviation",
        1: "Expression abbreviated",
        2: "Animal",
        3: "Organ of body",
        4: "Color",
        5: "Invention, book and other creative piece",
        6: "Currency name",
        7: "Disease and medicine",
        8: "Event",
        9: "Food",
        10: "Musical instrument",
        11: "Language",
        12: "Letter like a-z",
        13: "Other entity",
        14: "Plant",
        15: "Product",
        16: "Religion",
        17: "Sport",
        18: "Element and substance",
        19: "Symbols and sign",
        20: "Techniques and method",
        21: "Equivalent term",
        22: "Vehicle",
        23: "Word with a special property",
        24: "Definition of something",
        25: "Description of something",
        26: "Manner of an action",
        27: "Reason",
        28: "Group or organization of persons",
        29: "Individual",
        30: "Title of a person",
        31: "Description of a person",
        32: "City",
        33: "Country",
        34: "Mountain",
        35: "Other location",
        36: "State",
        37: "Postcode or other code",
        38: "Number of something",
        39: "Date",
        40: "Distance, linear measure",
        41: "Price",
        42: "Order, rank",
        43: "Other number",
        44: "Lasting time of something",
        45: "Percent, fraction",
        46: "Speed",
        47: "Temperature",
        48: "Size, area and volume",
        49: "Weight"
    }
    question = entry["text"]
    answer = label_description_dict[entry["fine_label"]]  

    return {
        "dataset": 'trec',
        "question": question,  
        "answer": answer,
    }


def trec_qa_dataset_to_few_shot_multi_round_qa_dataset(dataset: Dataset, context_num: int, qa_num: int, num_variance_ratio: Optional[float] = None, context_template: str = "Question: {question}\nType: {answer}\n", separate_str: str = "\n") -> Dataset:
    """
    Data name: trec
    Data source: https://huggingface.co/datasets/trec
    Data structure of the input dataset:
        {
            'dataset': 'trec',
            'question': 'How did serfdom develop in and then leave Russia ?',
            'answer': 'Manner of an action'
        }
    """
    base_context_num = context_num
    base_qa_num = qa_num

    if num_variance_ratio is not None:
        max_context_num = int(base_context_num * (1+num_variance_ratio))
        max_qa_num = int(base_qa_num * (1+num_variance_ratio))
    else:
        max_context_num = base_context_num
        max_qa_num = base_qa_num

    # Create a few-shot QA dataset from the TREC QA dataset
    few_shot_qa_list = []
    for start_p in range(0, len(dataset)-max_context_num-max_qa_num, max_qa_num): # drop the last few examples
        questions = []
        answers = []
        contexts = []
        if num_variance_ratio is not None:
            # ulter the number of context and qa with maximum num_variance_ratio*context_num
            context_num = int(base_context_num * random.uniform(1-num_variance_ratio, 1+num_variance_ratio))
            qa_num = int(base_qa_num * random.uniform(1-num_variance_ratio, 1+num_variance_ratio))


        for i in range(context_num):
            entry = dataset[start_p+i]
            contexts.append(context_template.format(question=entry["question"], answer=entry["answer"]))
        for i in range(context_num, context_num+qa_num):
            entry = dataset[start_p+i]
            questions.append(entry["question"])
            answers.append(entry["answer"])
        few_shot_qa_list.append({
            "dataset": "trec",
            "questions": questions,
            "answers": answers,
            "evidences": answers,
            "context": separate_str.join(contexts),
            "summary": "",
        })
    few_shot_qa_dataset = Dataset.from_pandas(pd.DataFrame(few_shot_qa_list))

    return few_shot_qa_dataset

def trec_qa_dataset_to_few_shot_multi_round_qa_dataset_random(dataset: Dataset, context_num: int, qa_num: int, num_output_items: int, num_variance_ratio: Optional[float] = None, context_template: str = "Question: {question}\nType: {answer}\n", separate_str: str = "\n") -> Dataset:
    """
    Data name: trec
    Data source: https://huggingface.co/datasets/trec
    Data structure of the input dataset:
        {
            'dataset': 'trec',
            'question': 'How did serfdom develop in and then leave Russia ?',
            'answer': 'Manner of an action'
        }
    """
    base_context_num = context_num
    base_qa_num = qa_num

    if num_variance_ratio is not None:
        max_context_num = int(base_context_num * (1+num_variance_ratio))
        max_qa_num = int(base_qa_num * (1+num_variance_ratio))
    else:
        max_context_num = base_context_num
        max_qa_num = base_qa_num

    # Create a few-shot QA dataset from the TREC QA dataset
    few_shot_qa_list = []
    for i in range(num_output_items):
        if num_variance_ratio is not None:
            # ulter the number of context and qa with maximum num_variance_ratio*context_num
            context_num = int(base_context_num * random.uniform(1-num_variance_ratio, 1+num_variance_ratio))
            qa_num = int(base_qa_num * random.uniform(1-num_variance_ratio, 1+num_variance_ratio))
        # random select context_num + qa_num dataset items, ranging from 0 to len(dataset)
        ids = random.sample(range(len(dataset)), context_num + qa_num)

        questions = []
        answers = []
        contexts = []

        for i in range(context_num):
            entry = dataset[ids[i]]
            contexts.append(context_template.format(question=entry["question"], answer=entry["answer"]))
        for i in range(context_num, context_num+qa_num):
            entry = dataset[ids[i]]
            questions.append(entry["question"])
            answers.append(entry["answer"])
        few_shot_qa_list.append({
            "dataset": "trec",
            "questions": questions,
            "answers": answers,
            "evidences": answers,
            "context": separate_str.join(contexts),
            "summary": "",
        })
        
    few_shot_qa_dataset = Dataset.from_pandas(pd.DataFrame(few_shot_qa_list))

    return few_shot_qa_dataset


def merge_multi_round_qa_dataset(
        dataset: Dataset,
        merge_num: int,
        context_merge_str: str = "\n\n",
        summary_merge_str: str = "\n\n",
    ) -> Dataset:
    """
    Merge the rows of multi-round QA dataset
    """
    merged_dataset = []
    for i in tqdm(range(0, len(dataset) - merge_num, merge_num)):
        questions = []
        answers = []
        evidences = []
        contexts = []
        summaries = []
        for j in range(merge_num):
            questions.extend(dataset[i+j]["questions"])
            answers.extend(dataset[i+j]["answers"])
            evidences.extend(dataset[i+j]["evidences"])
            contexts.append(dataset[i+j]["context"])
            summaries.append(dataset[i+j]["summary"])
        contexts = context_merge_str.join(contexts)
        summaries = summary_merge_str.join(summaries)
        merged_dataset.append({
            "dataset": dataset[i]["dataset"],
            "questions": questions,
            "answers": answers,
            "evidences": evidences,
            "context": contexts,
            "summary": summaries,
        })
    return Dataset.from_pandas(pd.DataFrame(merged_dataset))

def multi_round_qa_to_multi_round_conversation(
    entry: Dict,
    model_name: str,
    tokenizer: AutoTokenizer,
    prompt_format: str = None,
    question_format: str = None,
    answer_format: str = None,
) -> dict:
    """
    Convert multi-round QA format to multi-round conversation format.
    """

    context = entry["context"]

    conv = get_conversation_template(model_name)

    input_lengh = 0
    answer_length = 0

    input_lengh += len(tokenizer(conv.dict()["system_message"])["input_ids"])

    for i, (question, answer, evidence) in enumerate(
        zip(entry["questions"], entry["answers"], entry["evidences"])
    ):
        if i == 0:
            # First line of data
            if prompt_format is not None:
                user_input = prompt_format.format(context=context, question=question)
            else:
                user_input = context + question
        else:
            # Subsequent lines
            if question_format is not None:
                user_input = question_format.format(context=context, question=question)
            else:
                user_input = question
            
        input_lengh += len(tokenizer(user_input)["input_ids"])

        # Generate the user input and model response
        if answer_format is not None:
            model_response = answer_format.format(context=context, question=question, answer=answer)
        else:
            model_response = answer

        answer_length += len(tokenizer(model_response)["input_ids"])

        # Add the user input and model response to the conversation
        conv.append_message(conv.roles[0], user_input)
        conv.append_message(conv.roles[1], model_response)

    conversation = conv.get_prompt()

    # Input length and answer length
    total_length = len(tokenizer(conversation)["input_ids"])

    return {"text": conversation, "model_name": model_name, "input_length": input_lengh, "answer_length": answer_length, "total_length": total_length}

def multi_round_qa_to_length(
        entry: Dict, 
        tokenizer: AutoTokenizer, 
        prompt_format: str = None,
        question_format: str = None,
        answer_format: str = None,
        reserve_length: int = 128,
    ) -> dict:
    """
    The input entry has the format
    {
        "dataset": "trec",
        "questions": questions,
        "answers": answers,
        "evidences": answers,
        "context": separate_str.join(contexts),
        "summary": "",
    }
    Reserve some length for system prompts when calculating length level
    Calculate the input length, answer length, prompt length of the entry.
    """
    context = entry["context"]

    context_length = len(tokenizer(context)["input_ids"])
    input_length = 0
    question_length = 0
    answer_length = 0

    for i, (question, answer) in enumerate(zip(entry["questions"], entry["answers"])):
        # raw question length
        question_length += len(tokenizer(question)["input_ids"])

        if i == 0:
            # First line of data
            if prompt_format is not None:
                user_input = prompt_format.format(context=context, question=question)
            else:
                user_input = context + question
        else:
            # Subsequent lines
            if question_format is not None:
                user_input = question_format.format(context=context, question=question)
            else:
                user_input = question

        # input with context length
        input_ids = tokenizer(user_input)["input_ids"]
        input_length += len(input_ids)

        # anwer length
        if answer_format is not None:
            model_response = answer_format.format(context=context, question=question, answer=answer)
        else:
            model_response = answer

        model_response_ids = tokenizer(model_response)["input_ids"]
        answer_length += len(model_response_ids)

    total_length = input_length + answer_length

    # calculate the total length level by 1024, reserve some length
    total_length_level = ((total_length + reserve_length) // 1024) + 1

    return {"context_length": context_length, "question_length": question_length, "answer_length": answer_length, 'input_length': input_length, 'total_length': total_length, 'total_length_level': total_length_level, "reserve_length": reserve_length}
    
def context_reduction(
        entry: Dict,
        tokenizer: AutoTokenizer,
        expected_total_length: int,
    ) -> dict:
    """
    The input entry has the format
    {
        "dataset": "trec",
        "questions": questions,
        "answers": answers,
        "evidences": answers,
        "context": separate_str.join(contexts),
        "summary": "",
    }
    Reduce the original context length to new context length.
    """
    context = entry["context"]
    original_context_length = len(tokenizer(context)["input_ids"])
    tokenized_context = tokenizer(context, truncation=False, return_tensors="pt").input_ids[0]
    reduce_length = entry["total_length"] - expected_total_length
    context_length = original_context_length - reduce_length

    # sometimes, we do not need to truncate
    if "truncate" in entry.keys():
        truncate = entry["truncate"]
    else:
        truncate = False
    if original_context_length < context_length:
        return {"context": context, "truncate": truncate}
    if reduce_length > original_context_length:
        print("The context length is smaller than the expected reduction length.")
        return {"context": context, "truncate": truncate}
    if reduce_length < 0:
        print("The expected total length is larger than the original total length.")
        return {"context": context, "truncate": truncate}
    
    # select the top half and latter half
    half = int(context_length/2)
    new_context = tokenizer.decode(tokenized_context[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_context[-half:], skip_special_tokens=True)

    return {"context": new_context, "truncate": True}

def multi_round_qa_to_multi_round_qa_by_model(
    entry: Dict,
    model: AutoModelForCausalLM,
    model_name: str,
    tokenizer: AutoTokenizer,
    prompt_format: str = None,
    question_format: str = None,
    max_length: int = None,
) -> dict:
    """
    Convert multi-round QA format to multi-round conversation format.
    """
    print("Please use multi_round_qa_to_multi_round_qa_by_batch_model instead.")

    # if max length is not given, set it to a very large number
    if max_length is None:
        max_length = 16384

    context = entry["context"]

    conv = get_conversation_template(model_name)

    input_length = 0
    answer_length = 0

    # unique questions
    questions = list(set(entry["questions"]))
    answers = []

    # input data
    for i, question in enumerate(questions):
        if i == 0:
            # First line of data
            if prompt_format is not None:
                user_input = prompt_format.format(context=context, question=question)
            else:
                user_input = context + question
        else:
            # Subsequent lines
            if question_format is not None:
                user_input = question_format.format(context=context, question=question)
            else:
                user_input = question
        
        conv.append_message(conv.roles[0], user_input)
        conv.append_message(conv.roles[1], None)

        model_input = conv.get_prompt()

        input_ids = tokenizer(model_input, return_tensors="pt", padding=True).input_ids
        input_length = input_ids.shape[-1]

        if input_length > max_length:
            break

        # Generate the user input and model response
        model_response_ids = model.generate(
            input_ids=input_ids.to(model.device),
            max_new_tokens=128,
            num_return_sequences=1,
            # past_key_values=None,
            # use_cache=True,
            # return_dict=True,
        )[0] # return the first generated sequence of shape [length]
        model_response_ids = model_response_ids[..., input_length:]
        
        model_response = tokenizer.decode(model_response_ids, skip_special_tokens=True)

        answers.append(model_response)
        answer_length += model_response_ids.shape[-1]

        # Add the model response to the conversation
        conv.update_last_message(model_response)

    # conversation = conv.get_prompt()

    # Input length and answer length
    questions = questions[:len(answers)]

    return {'questions': questions, 'answers': answers, 'input_length': input_length, 'answer_length': answer_length}

def multi_round_qa_to_multi_round_qa_model_by_batch(
    entry: Dict,
    model: Union[AutoModelForCausalLM, List[AutoModelForCausalLM]],
    model_name: str,
    tokenizer: AutoTokenizer,
    prompt_format: str = None,
    max_length: int = None,
    batch_size: int = 4,
    idx: int = None
) -> dict:
    """
    Convert multi-round QA format to multi-round conversation format.
    """

    # if max length is not given, set it to a very large number
    if max_length is None:
        max_length = 16384
    
    # choose the model based on the idx
    # if isinstance(model, list) and idx is not None:
    #     model = model[idx]
    # elif not isinstance(model, list) and idx is None:
    #     pass
    # else:
    #     raise ValueError("The model should be a single model or a list of models with the same length as the dataset.")
    if idx is not None:
        # print(idx)
        device = f"cuda:{(idx or 0) % torch.cuda.device_count()}"
        # print(device)
        model.to(device)

    context = entry["context"]

    input_length = 0
    answer_length = 0

    # unique questions
    questions = sorted(list(set(entry["questions"])))
    model_inputs = []
    answers = []

    # input data
    for i, question in enumerate(questions):
        if prompt_format is not None:
            user_input = prompt_format.format(context=context, question=question)
        else:
            user_input = context + question
        
        conv = get_conversation_template(model_name)
        conv.append_message(conv.roles[0], user_input)
        conv.append_message(conv.roles[1], None)
        stop_token_ids = conv.stop_token_ids

        model_input = conv.get_prompt()

        model_inputs.append(model_input)
    
    # batch inference
    with torch.no_grad():
        for i in range(0, len(model_inputs), batch_size):
            model_input = model_inputs[i: i+batch_size]
            model_input = tokenizer(model_input, return_tensors="pt", padding=True).to(model.device)

            # input_ids = model_input.input_ids.to(model.device)

            input_lengths = torch.sum(model_input.attention_mask, dim=1)
            input_length += sum(input_lengths)

            if max(input_lengths) > max_length:
                break

            try:
                # terminators = [
                #     tokenizer.eos_token_id,
                #     tokenizer.convert_tokens_to_ids("<|eot_id|>")
                # ]
                
                # Generate the user input and model response
                model_responses = model.generate(
                    **model_input,
                    max_new_tokens=128,
                    eos_token_id=stop_token_ids,
                    # do_sample=True,
                    # temperature=0.6,
                    # top_p=0.9,
                    num_return_sequences=1,
                    # past_key_values=None,
                    # use_cache=True,
                    # return_dict=True,
                ) # return the first generated sequence of shape [input, length]
                # model_response_ids = [model_response_ids[i][input_lengths[i]:] for i in range(len(model_response_ids))]
                model_responses = model_responses[..., model_input.input_ids.shape[-1]:] # left padding
                answer_length += sum([model_response_id.shape[-1] for model_response_id in model_responses])
                
                # model_response = tokenizer.decode(model_response_ids, skip_special_tokens=True)
                model_responses = tokenizer.batch_decode(model_responses,  skip_special_tokens=True)
            except Exception as e:
                print(e)
                model_responses = [""] * len(model_input)

            answers.extend(model_responses)

    # conversation = conv.get_prompt()
    # Input length and answer length
    questions = questions[:len(answers)]

    return {'questions': questions, 'answers': answers, 'input_length': input_length, 'answer_length': answer_length}


if __name__ == "__main__":
    # base_path = "local/universal/final_dataset"
    # base_path = "local/universal/final_validsplit"
    base_path = "local/universal/final_testsplit"

    dataset_names = [
        # "local/universal/dataset/longeval_lines_valid",
        "longeval_lines_test",
        "trec",
        "allenai/qasper",
        "multi_news",
        "microsoft/LCC_python",
        "hotpot_qa"
    ]

    dataset_names_short = [
        "longeval",
        "trec",
        "qasper",
        "multi_news",
        "lcc",
        "hotpot_qa"
    ]

    convert_range = range(0, 5)
    # convert_range = range(1, len(dataset_names))

    acceptable_length_levels = [0, 2, 4, 6, 8, 12, 16]
    # acceptable_length_levels = [0, 2, 4, 6, 8, 12]

    subset_names = [
        None,
        None,
        None,
        None,
        None,
        "fullwiki",
    ]

    local_dataset = [
        True,
        False,
        False,
        False,
        False,
        False,
    ]

    # dataset_splits = [
    #     None,
    #     "train",
    #     "train",
    #     "train",
    #     "train",
    #     "train",
    # ]

    dataset_splits = [
        None,
        "test",
        "validation",
        "test",
        "test",
        "validation",
    ]

    dataset_maps = [
        lambda entry: longeval_to_multi_round_qa(entry, num_questions=10),
        trec_to_qa,
        qasper_to_multi_round_qa,
        multiNews_to_multi_round_qa,
        lcc_to_multi_round_qa,
        hotpotQA_to_multi_round_qa,
    ]

    prompt_format = [
        "Below is a record of lines I want you to remember. Each line begins with 'line <line index>' and contains a '<REGISTER_CONTENT>' at the end of the line as a numerical value. For each line index, memorize its corresponding <REGISTER_CONTENT>. At the end of the record, I will ask you to retrieve the corresponding <REGISTER_CONTENT> of a certain line index. Now the record start:\n\n{context}\nNow the record is over. Tell me what is the <REGISTER_CONTENT> in line {question}? I need the number.",
        "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{question}",
        'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {question}\n\nAnswer:',
        "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:", # empty question,
        "Please complete the code given below. \n{question}\nNext line of code:\n",
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:",
    ]

    question_format = [
        "Tell me what is the <REGISTER_CONTENT> in line {question}? I need the number.",
        "Please determine the type of the question below.\n{question}",
        "Question: {question}\n\nAnswer:",
        "Now, write a one-page summary of all the news.\n\nSummary:", # empty question
        "Please continue completing the code given below. \n{question}\nNext line of code:\n",
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:",
    ]

    answer_format = [
        "The <REGISTER_CONTENT> in line {question} is <{answer}>.",
        "Type: {answer}",
        # "{answer}",
        # "{answer}",
        "{answer}",
        "{answer}",
        "{answer}\n",
        "{answer}",
    ]

    # setup
    df = pd.DataFrame({
        "dataset_names": dataset_names_short, 
        "subset_names": subset_names,
        "local_dataset": local_dataset,
        "dataset_splits": dataset_splits,
        "dataset_maps": dataset_maps,
        "prompt_format": prompt_format,
        "question_format": question_format,
        "answer_format": answer_format,
    })
    print(df)
    # save df to file
    df.to_csv(os.path.join(base_path, "convert_config.csv"), index=False)
    
    """
    Convert to multi_round_qa dataset
    """
    output_qa_dir = os.path.join(base_path, "multi_qa")
    multi_round_qa_dataset_dirs = [os.path.join(output_qa_dir, dataset_name.replace("/", "_")) for dataset_name in dataset_names]
    print(multi_round_qa_dataset_dirs)

    concise_output_qa_dir = output_qa_dir = os.path.join(base_path, "multi_qa_concise")
    concise_multi_round_qa_dataset_dirs = [os.path.join(concise_output_qa_dir, dataset_name.replace("/", "_")) for dataset_name in dataset_names]
    print(concise_multi_round_qa_dataset_dirs)

    if False:
        for i in convert_range:
            # Load the dataset
            if not local_dataset[i]:
                dataset_name = dataset_names[i]
                subset_name = subset_names[i]
                split = dataset_splits[i]
                dataset = load_dataset(dataset_name, subset_name, split=split)
            else:
                dataset_dir = dataset_names[i]
                dataset_name = dataset_dir.split('/')[-1]
                dataset = Dataset.load_from_disk(dataset_dir)

            print("dataset\n", dataset)

            example = dataset_maps[i](
                entry=dataset[0]
            )
            print(example)

            # Convert the dataset to multi-round QA format
            multi_round_qa_dataset = dataset.map(
                dataset_maps[i], remove_columns=dataset.column_names
            )  # output columns: ['dataset', 'questions', 'answers', 'evidences', 'context', 'summary']

            # merge different items as one entry
            if dataset_name == "trec":
                # 55 tokens per question-answer pair 
                num_question_pairs = [18, 37, 56, 74, 93, 112, 130, 149, 186, 204, 223] # 1k, 2k, 3k, 4k, 5k, 6k, 7k, 8k, 10k, 11k, 12k
                num_answer_pairs = [18, 37, 56, 74, 93, 112, 130, 149, 186, 204, 223]
                variance_ratio = 0.1
                if dataset_splits[i] == "train":
                    trec_datasets = []
                    for num_question_pair, num_answer_pair in zip(num_question_pairs, num_answer_pairs):
                        # use few shot setting to convert to multi-round QA
                        # merge all the data
                        multi_round_qa_dataset_this = trec_qa_dataset_to_few_shot_multi_round_qa_dataset(multi_round_qa_dataset, num_question_pair, num_answer_pair, num_variance_ratio=variance_ratio)
                        trec_datasets.append(multi_round_qa_dataset_this)
                    
                elif dataset_splits[i] == "test" or "validation":
                    trec_datasets = []
                    for num_question_pair, num_answer_pair in zip(num_question_pairs, num_answer_pairs):
                        multi_round_qa_dataset_this = trec_qa_dataset_to_few_shot_multi_round_qa_dataset_random(multi_round_qa_dataset, num_question_pair, num_answer_pair, num_output_items=100, num_variance_ratio=variance_ratio)
                        trec_datasets.append(multi_round_qa_dataset_this)
                else:
                    raise ValueError("The dataset split is not supported.")
                multi_round_qa_dataset = concatenate_datasets(trec_datasets)

            # merge different items as one entry
            if dataset_name == "hotpot_qa":
                # merge different items as one entry
                hotpot_qa_datasets = []
                merge_nums = [2, 3, 4]
                for merge_num in merge_nums:
                    hotpot_qa_datasets.append(merge_multi_round_qa_dataset(multi_round_qa_dataset, merge_num=merge_num))

                multi_round_qa_dataset = concatenate_datasets(hotpot_qa_datasets)
                
            # add length data
            tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5-16k")
            multi_round_qa_dataset = multi_round_qa_dataset.map(
                lambda x: multi_round_qa_to_length(
                    entry=x,
                    tokenizer=tokenizer,
                    prompt_format=prompt_format[i],
                    question_format=question_format[i],
                    answer_format=answer_format[i],
                ),
            )

            # sort the dataset based on total length level and answer length
            multi_round_qa_dataset = multi_round_qa_dataset.sort("answer_length", reverse=True)
            multi_round_qa_dataset = multi_round_qa_dataset.sort("total_length_level", reverse=False)
            
            # print the range of total length level
            length_levels = list(set(multi_round_qa_dataset["total_length_level"]))
            print(length_levels)

            # save to file
            multi_round_qa_dataset.save_to_disk(multi_round_qa_dataset_dirs[i])
    
    """
    Convert to the concise version with the complete multi round qa dataset
    """
    if False:
        for i in convert_range:
            multi_round_qa_dataset = Dataset.load_from_disk(multi_round_qa_dataset_dirs[i])
            # select top examples for each total length level to become the new dataset
            topk = 50
            skip_num = 0
            concise_multi_round_qa_dataset = []
            length_level_count = []
            original_length_count = []
            for length_level_index in range(1, len(acceptable_length_levels)):
                multi_round_qa_dataset_this = multi_round_qa_dataset.filter(lambda x: x["total_length_level"] <= acceptable_length_levels[length_level_index] and x["total_length_level"] > acceptable_length_levels[length_level_index-1])
                # truncate = False
                multi_round_qa_dataset_this = multi_round_qa_dataset_this.map(lambda x: {"truncate": False})
                original_length_count.append(len(multi_round_qa_dataset_this))
                if len(multi_round_qa_dataset_this) < topk:
                    print(f"length level {acceptable_length_levels[length_level_index]} has less than {topk} examples")
                    # take from the next level, truncate the data, then concat with the current level
                    if length_level_index == len(acceptable_length_levels) - 1:
                        this_topk = len(multi_round_qa_dataset_this)
                        print(f"length level {acceptable_length_levels[length_level_index]} has less than {topk} examples, set topk to {this_topk}")
                    else:
                        multi_round_qa_dataset_next = multi_round_qa_dataset.filter(lambda x: x["total_length_level"] <= acceptable_length_levels[length_level_index+1] and x["total_length_level"] > acceptable_length_levels[length_level_index])
                        # take data from next level
                        multi_round_qa_dataset_next = multi_round_qa_dataset_next.sort("total_length", reverse=False)
                        next_topk = topk-len(multi_round_qa_dataset_this)
                        if len(multi_round_qa_dataset_next) < next_topk:
                            next_topk = len(multi_round_qa_dataset_next)
                        multi_round_qa_dataset_next = multi_round_qa_dataset_next.select(range(next_topk))
                        # calculatet the expected context length of each data
                        multi_round_qa_dataset_next = multi_round_qa_dataset_next.map(
                            lambda x: context_reduction(
                                entry=x,
                                tokenizer=tokenizer,
                                expected_total_length=acceptable_length_levels[length_level_index]*1024 - x["reserve_length"] - 8,
                            ),
                        )
                        multi_round_qa_dataset_next = multi_round_qa_dataset_next.map(
                            lambda x: multi_round_qa_to_length(
                                entry=x,
                                tokenizer=tokenizer,
                                prompt_format=prompt_format[i],
                                question_format=question_format[i],
                                answer_format=answer_format[i],
                            ),
                        )
                        multi_round_qa_dataset_next = multi_round_qa_dataset_next.map(lambda x: {"truncate": True})
                        multi_round_qa_dataset_this = concatenate_datasets([multi_round_qa_dataset_this, multi_round_qa_dataset_next])
                        this_topk = len(multi_round_qa_dataset_this)
                else:
                    this_topk = topk
                length_level_count.append(this_topk - skip_num)
                # sort by total length level in descending order
                multi_round_qa_dataset_this = multi_round_qa_dataset_this.sort("total_length_level", reverse=True)
                multi_round_qa_dataset_this = multi_round_qa_dataset_this.select(range(skip_num, this_topk))
                concise_multi_round_qa_dataset.append(multi_round_qa_dataset_this)
            concise_multi_round_qa_dataset = concatenate_datasets(concise_multi_round_qa_dataset)
            # save to disk
            concise_multi_round_qa_dataset.save_to_disk(concise_multi_round_qa_dataset_dirs[i])
            length_dict = {f"{length_level_lower}-{length_level_higher}": [count, original_count] for (length_level_lower, length_level_higher), (count, original_count) in zip(zip(acceptable_length_levels[:-1], acceptable_length_levels[1:]), zip(length_level_count, original_length_count))}
            df = pd.DataFrame(length_dict, index=[0, 1]).T
            print(df)
            df.to_csv(os.path.join(concise_multi_round_qa_dataset_dirs[i], "lengths.csv"), index=True)
            concise_multi_round_qa_dataset.to_json(os.path.join(concise_multi_round_qa_dataset_dirs[i], "text.json"))

    """
    Convert to multi_round_conversation dataset with original responses
    """
    if True:
        # model_name = "lmsys/vicuna-7b-v1.5-16k"
        # model_class_name = "vicuna"
        # ALERT: remember to modify the answer format accordingly
        model_name = "gradientai/Llama-3-8B-Instruct-262k"
        model_class_name = "llama3"

        output_conversation_dir = os.path.join(base_path, "multi_conversation", model_class_name)

        multi_round_conversation_dataset_dirs = [os.path.join(output_conversation_dir, dataset_name.replace("/", "_")) for dataset_name in dataset_names]
        print(multi_round_conversation_dataset_dirs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        for i in convert_range:
            # Load the dataset
            dataset_name = concise_multi_round_qa_dataset_dirs[i]
            multi_round_qa_dataset = Dataset.load_from_disk(dataset_name)

            example = multi_round_qa_to_multi_round_conversation(
                entry=multi_round_qa_dataset[0],
                model_name=model_name,
                tokenizer=tokenizer,
                prompt_format=prompt_format[i],
                question_format=question_format[i],
                answer_format=answer_format[i],
            )
            print(example['text'])

            multi_round_conversation_dataset = multi_round_qa_dataset.map(
                lambda x: multi_round_qa_to_multi_round_conversation(
                    entry=x,
                    model_name=model_name,
                    tokenizer=tokenizer,
                    prompt_format=prompt_format[i],
                    question_format=question_format[i],
                    answer_format=answer_format[i],
                ),
                remove_columns=[c for c in multi_round_qa_dataset.column_names if c not in ["dataset", "total_length_level", "truncate"]],
            )
            # output columns: ['text', 'model_name', 'input_length', 'answer_length']

            # save to file
            multi_round_conversation_dataset.save_to_disk(multi_round_conversation_dataset_dirs[i])
            multi_round_conversation_dataset.to_json(os.path.join(multi_round_conversation_dataset_dirs[i], "text.json"))

            # convert the input_length and answer_length to a DataFrame
            df = pd.DataFrame(multi_round_conversation_dataset)
            columns = multi_round_conversation_dataset.column_names
            # remove text from columns
            columns.remove("text")
            df = df[columns]
            df.to_csv(os.path.join(multi_round_conversation_dataset_dirs[i], "lengths.csv"), index=False)



    print("done")
