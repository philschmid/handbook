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
echo "Running $num_iteration iterations"
echo "Output directory: $output_dir"

###########################
# Start Iterative DPO Loop
###########################
# 0. Prepare the dataset
# 1. Generate Candidates
# 2. Rank Candidates
# 3. Generate comparison dataset
# 4. Train model with DPO
# 5. save DPO model and start next iteration

# # Split the data for iteration 
# python scripts/iterative_dpo/run_prepare_dataset.py

# Loop over the num_iterations and add them
for ((i=1; i<=num_iteration; i++)); do
# Generate Candidates
echo "Running iteration $i"
python scripts/iterative_dpo/run_generate_candidates.py --config $config --dataset_path $output_dir/iteration_$i/prompts.json
# Rank Candidates
CUDA_VISIBLE_DEVICES=0 python scripts/iterative_dpo/run_rank_candidates.py --config $config --dataset_path $output_dir/iteration_$i/candidates.json
# Generate comparison dataset
python scripts/iterative_dpo/run_prepare_pairwise_dataset.py --dataset_path $output_dir/iteration_$i/candidates.json --current_iteration $i
# 
accelerate launch scripts/iterative_dpo/run_train_dpo.py --config $config --dataset_path $output_dir/iteration_$i/pairwise.json
done

