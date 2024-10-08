#!/bin/bash

# Check if the correct number of arguments are passed
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <moa_config_dir> <moa_config_num> <result_dir> <model_name>"
    exit 1
fi

# Assign command line arguments to variables
MOA_CONFIG_DIR=$1
MOA_CONFIG_NUM=$2
RESULT_DIR=$3
MODEL_NAME=$4

mkdir -p $RESULT_DIR

dataset_dir="nics-efc/MoA_Long_HumanQA"
split="valid"
length=12288

# Loop from 0 to the specified moa_config_num (minus 1 to accommodate zero indexing)
for i in $(seq 0 $((MOA_CONFIG_NUM - 1)))
do
  # Set the MOA_CONFIG path and result path based on the current iteration
  MOA_CONFIG_PATH="${MOA_CONFIG_DIR}/moa_config_plan_${i}.json"
  RESULT_PATH="${RESULT_DIR}/validation_${i}.csv"

  # Run the Python command with the dynamically set paths
  command="CUDA_VISIBLE_DEVICES=0 python scripts/pipeline/perplexity_evaluate.py \
    --model_name $MODEL_NAME \
    --max_length ${length} \
    --dataset_dir $dataset_dir \
    --split $split \
    --response_mask \
    --moa_config $MOA_CONFIG_PATH \
    --result_path $RESULT_PATH"

  # Check if RESULT_PATH exists
  if [ -f $RESULT_PATH ]; then
    echo "File $RESULT_PATH exists. Skipping..."
    continue
  fi

  echo $command
  eval $command
done

python scripts/helper/select_validation_result.py --input_dir $RESULT_DIR