# MoA: Mixture of Sparse Attention for Automatic Large Language Model Compression

## Environment Setup
```bash
conda create -n moa python=3.10.13
pip install -r requirement.txt
pip install -e .
```

## Automatic Search Pipeline

### Calibration Dataset Generation
To create the calibration dataset, run
```bash
python scripts/universal/generate_calibration_dataset.py --model_path lmsys/vicuna-7b-v1.5-16k --model_name vicuna-7b-v1.5-16k --output_path_base local/dataset
```

### Profile
To do profile on vicuna-7b on 2k, 4k, 8k length, run

```bash
python scripts/universal/universal_pipeline_profile.py --model_name lmsys/vicuna-7b-v1.5-16k --max_length 2048 --response_mask --dataset_dir local/dataset/multi_conversation_model/vicuna-7b-v1.5-16k/multi_news --grad_dir 7b/profile_2k

python scripts/universal/universal_pipeline_profile.py --model_name lmsys/vicuna-7b-v1.5-16k --max_length 4096 --response_mask --dataset_dir local/dataset/multi_conversation_model/vicuna-7b-v1.5-16k/multi_news --grad_dir 7b/profile_4k

python scripts/universal/universal_pipeline_profile.py --model_name lmsys/vicuna-7b-v1.5-16k --max_length 8192 --response_mask --dataset_dir local/dataset/multi_conversation_model/vicuna-7b-v1.5-16k/multi_news --grad_dir 7b/profile_8k
```

### Optimize
You can run

```bash
python scripts/universal/elastic_generate.py --output_dir 7b/lut_result --elastic_length 2048 4096 8192 --extend_length 16384 --density_bounds 0.5 0.5 0.5 0.5 --importance_tensor_dir 7b/ --output_length 4096 8192 12288 16384 --num_plan_limit 2 --aggregating_block_size 64 --latency_lower_bound_ratio 0.9
```

You can set `--time_limit num` to bound the time. Also you might need to apply for gurobi free licenses for academics to use gurobi library.

### Validate
Run the following code to evaluate a rule on a certain length level:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/universal/universal_lut_evaluate.py --model_name lmsys/vicuna-7b-v1.5-16k --max_length 12288 --dataset_dir validation_dataset/vicuna/ --response_mask --lut_path 7b/lut_result/lut_12288_plan_{i}.pt  --result_path validation_test.csv --total_length_level_down 2
```
Replace `i` with actual rule id.

Or you can run
```
scripts/universal/validate.sh <lut_dir> <plan_num> <result_dir> <model_name>
```

to evaluate all the plans under the directory.

## Evaluation

### Apply MoA to LLM

Given the elastic rules found by MoA (given by `elastic_rule_path`), you can simply apply the rules to the model with few lines. 

```python
from transformers import AutoModelForCausalLM
from MoA.models.interface import update_model_function
from MoA.attention.set import set_static_attention_lut

# Load the huggingface model
model_name = "lmsys/vicuna-7b-v1.5-16k"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add sparse attention capability to the model by modifying the forward function
model = update_model_function(model, model_name)
model.model.use_block_sparse_attention_lut(permute_head=True, sparse_decode=True)

# Load the sparse attention rules to the model
set_static_attention_lut(elastic_rule_path, model_layers=model.model.layers, permute_head=True, sparse_decode=True)

# Now you can use the `model` for efficient inference like any regular huggingface model
# For example, you can use it in pipeline to chat with the model
pipe = pipeline(task="text-generation", tokenizer=tokenizer, model=model, trust_remote_code=True)
prompt = "Hi."
output = pipe(prompt)
```

### Retrieval
To test the retrieval performance of a rule on a certain input length, run

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/universal/retrieval_evaluate.py --model_name lmsys/vicuna-7b-v1.5-16k --lut_path 7b/lut_result/lut_8192_plan_{i}.pt --output_dir 7b/retrieval_8k --length_level 8
```

Replace `i` with actual rule id.

### LongBench
To test the performance of a elastic rule on LongBench 8k+ split, run

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/universal/longbench_evaluate.py --model_name lmsys/vicuna-7b-v1.5-16k --max_length 7500 --eval longbench_fast --longbench_e --longbench_result_dir 7b/longbench_result --longbench_length_range 4-8k --use_lut --lut_path 7b/lut_result/lut_8192_plan_{i}.pt

CUDA_VISIBLE_DEVICES=0 python scripts/universal/longbench_evaluate.py --model_name lmsys/vicuna-7b-v1.5-16k --max_length 15500 --eval longbench_fast --longbench_e --longbench_result_dir 7b/longbench_result --longbench_length_range 8k+ --use_lut --lut_path 7b/lut_result/lut_16384_plan_{i}.pt
```

Replace `i` with actual rule id.
