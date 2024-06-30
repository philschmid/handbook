#!/bin/bash

# Parse the arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        # --key) key="$2"; shift ;;
        --config) config="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check if key and config are not empty
if [ -z "$config" ]; then
    echo "Config file is missing"
    exit 1
fi


# Extract the num_iteration value from the config file
num_iteration=$(awk -F: '/num_iteration/ {gsub(/ /, "", $2); print $2}' "$config")
output_dir=$(awk -F: '/output_dir/ {gsub(/ /, "", $2); print $2}' "$config")
initial_learning_rate=$(awk -F: '/learning_rate/ {gsub(/ /, "", $2); print $2}' "$config")
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Running $num_iteration iterations"
echo "Output directory: $output_dir"
echo "Initial Learning Rate: $initial_learning_rate"
echo "Number of GPUs: $num_gpus"

###########################
# Start Iterative DPO Loop
###########################
# 0. Prepare the dataset
# 1. Generate Candidates
# 2. Rank Candidates
# 3. Generate comparison dataset
# 4. Train model with DPO
# 5. save DPO model and start next iteration

# Split the data for iteration 
python scripts/iterative_dpo/run_prepare_dataset.py --config $config
if [ $? -ne 0 ]; then
  echo "Failed to prepare the dataset"
  exit 1
fi

# Loop over the num_iterations and add them
for ((i=1; i<=num_iteration; i++)); do
    echo "Running iteration $i"
    ########################
    # 1. Generate Candidates
    ########################
    # check if iteration > 1 and overwrite the config file generation_model_name_or_path    
    if [ $i -eq 1 ]; then
        generation_model_name_or_path=$(grep -E '^model_name_or_path:.*$' $config  | sed -E 's/^model_name_or_path:[[:space:]]*//' | tr -d '"') 
    else
        generation_model_name_or_path=$output_dir/iteration_$(($i-1))
    fi
    python scripts/iterative_dpo/run_generate_candidates.py --config recipes/iterative_dpo/dev.yaml --generation_model_name_or_path $generation_model_name_or_path --dataset_path $output_dir/iteration_$i/prompts.json
    if [ $? -ne 0 ]; then
        echo "Failed to generate candidates"
        exit 1
    fi

    #########################
    # 2. Rank Candidates
    #########################
    CUDA_VISIBLE_DEVICES=0 python scripts/iterative_dpo/run_rank_candidates.py --config $config --dataset_path $output_dir/iteration_$i/candidates.json
    if [ $? -ne 0 ]; then
        echo "Failed to rank candidates"
        exit 1
    fi

    ########################
    # 3. Generate DPO dataset
    ########################
    python scripts/iterative_dpo/run_prepare_pairwise_dataset.py --dataset_path $output_dir/iteration_$i/ranked_candidates.json --current_iteration $((i-1))
    if [ $? -ne 0 ]; then
        echo "Failed to prepare the pairwise dataset"
        exit 1
    fi

    ########################
    # 4. Train model with DPO
    ########################
    # check if iteration > 1 and overwrite the config file model_name_or_path    
    if [ $i -eq 1 ]; then
        model_name_or_path=$(grep -E '^model_name_or_path:.*$' $config  | sed -E 's/^model_name_or_path:[[:space:]]*//' | tr -d '"')
        ref_model_name_or_path=$(grep -E '^model_name_or_path:.*$' $config  | sed -E 's/^model_name_or_path:[[:space:]]*//' | tr -d '"')
    else
        model_name_or_path=$output_dir/iteration_$(($i-1))
        ref_model_name_or_path=$(grep -E '^model_name_or_path:.*$' $config  | sed -E 's/^model_name_or_path:[[:space:]]*//' | tr -d '"') 
    fi
    learning_rate=$(echo "$initial_learning_rate * 0.5^($i-1)" | bc | sed -r 's/^(-?)\./\10./' )

    # DS3
    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes $num_gpus scripts/iterative_dpo/run_dpo.py --config $config --model_name_or_path $model_name_or_path --output_dir $output_dir/iteration_$i  --dataset_id_or_path $output_dir/iteration_$i/pairwise.json --num_train_epochs 1 --learning_rate $learning_rate --ref_model_name_or_path $ref_model_name_or_path
    # ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml scripts/iterative_dpo/run_dpo.py --config $config --model_name_or_path $model_name_or_path --output_dir $output_dir/iteration_$i  --dataset_id_or_path $output_dir/iteration_$i/pairwise.json --num_train_epochs 1
    # python scripts/iterative_dpo/run_dpo.py --config $config --model_name_or_path $model_name_or_path --output_dir $output_dir/iteration_$i  --dataset_id_or_path $output_dir/iteration_$i/pairwise.json --num_train_epochs 1
    if [ $? -ne 0 ]; then
        echo "Failed to train the model with DPO"
        exit 1
    fi
done

echo "Finished running $num_iteration iterations"