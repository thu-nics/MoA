#!/bin/bash

# Check if the correct number of arguments are passed
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <lut_dir> <lut_num> <result_dir> <model_name>"
    exit 1
fi

# Assign command line arguments to variables
LUT_DIR=$1
LUT_NUM=$2
RESULT_DIR=$3
MODEL_NAME=$4

mkdir $RESULT_DIR

dataset_dir="fuvty/MoA_Human"
split="valid"
length=12288
length_level_down=2


# Loop from 0 to the specified lut_num (minus 1 to accommodate zero indexing)
for i in $(seq 0 $((LUT_NUM - 1)))
do
  # Set the LUT path and result path based on the current iteration
  LUT_PATH="${LUT_DIR}/lut_${length}_plan_${i}.pt"
  RESULT_PATH="${RESULT_DIR}/validation_${i}.csv"

  # Run the Python command with the dynamically set paths
  command="CUDA_VISIBLE_DEVICES=0 python scripts/universal/universal_lut_evaluate.py \
    --model_name $MODEL_NAME \
    --max_length ${length} \
    --dataset_dir $dataset_dir \
    --split $split \
    --response_mask \
    --lut_path $LUT_PATH \
    --result_path $RESULT_PATH \
    --total_length_level_down $length_level_down"

  # check if RESULT_PATH exists
  if [ -f $RESULT_PATH ]; then
    echo "File $RESULT_PATH exists. Skipping..."
    continue
  fi

  echo $command
  eval $command
done

python scripts/helper/select_validation_result.py --input_dir $RESULT_DIR
