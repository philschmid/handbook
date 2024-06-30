
# Instructions to run Iterative DPO Zephyr-7b

Run online Iterative DPO using `alignment-handbook/zephyr-7b-sft-full` and
`HuggingFaceH4/ultrafeedback_binarized` to replicate the results of the Zephyr-7b,
but in an online approach using `RLHFlow/ArmoRM-Llama3-8B-v0.1` as reward model. 

## Slurm 

```bash
sbatch --job-name=interative_dpo_constant_lr --nodes=1 recipes/iterative_dpo/launch.slurm recipes/iterative_dpo/dev.yaml
```

Grep margins: 

```bash
grep -oP "'rewards/margins': \K[0-9.]*" /fsx/philipp/logs/offline_dpo_test_5-7206177.out
``````


## Full training examples

You will require 8 GPUs (80GB of VRAM) to train the full model.
```shell
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/zephyr-7b-beta/sft/config_full.yaml

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/zephyr-7b-beta/dpo/config_full.yaml
```

## QLoRA training examples

Train faster with flash-attention 2 (GPU supporting FA2: A100, H100, etc)
```shell
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info ./scripts/iterative_dpo/run_iterative_dpo.sh --config recipes/iterative_dpo/dev-qlora.yaml
```



accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/iterative_dpo/run_dpo.py --config recipes/iterative_dpo/dev.yaml --model_name_or_path alignment-handbook/zephyr-7b-sft-full --output_dir test/iterative_dpo/iteration_1  --dataset_id_or_path test/iterative_dpo/iteration_1/pairwise.json --num_train_epochs 1


# Results full exp 1

| Iteration/SFT                                                                                                     | Single Turn | Multi Turn | Mean |
| ----------------------------------------------------------------------------------------------------------------- | ----------- | ---------- | ---- |
| SFT [alignment-handbook/zephyr-7b-sft-full](https://huggingface.co/alignment-handbook/zephyr-7b-sft-full)         | 6.7000      | 5.8375     | 6.27 |
| Offline DPO [alignment-handbook/zephyr-7b-dpo-full](https://huggingface.co/alignment-handbook/zephyr-7b-dpo-full) | 7.5438      | 7.0625     | 7.30 |
| Offline DPO trained (Global BS 64)                                                                                | 7.1375      | 7.0125     | 7.08 |
| Offline DPO trained (Global BS 128)                                                                               | 7.225       | 6.775      | 7.00 |
| Offline DPO trained (Global BS 128)                                                                               | 7.6125      | 7.20       | 7.41 |
| Online DPO (Iteration 1)                                                                                          | 7.0875      | 6.4750     | 6.78 |
| Online DPO (Iteration 2)                                                                                          | 7.212       | 5.7875     | 6.5  |
| Online DPO (Iteration 3)                                                                                          | 6.8438      | 4.9625     | 5.85 |


# Results full exp 2

| Iteration/SFT                                                                                                     | Single Turn | Multi Turn | Mean | MixEval Hard |
| ----------------------------------------------------------------------------------------------------------------- | ----------- | ---------- | ---- | ------------ |
| SFT [alignment-handbook/zephyr-7b-sft-full](https://huggingface.co/alignment-handbook/zephyr-7b-sft-full)         | 6.7000      | 5.8375     | 6.27 | 28.25        |
| Offline DPO [alignment-handbook/zephyr-7b-dpo-full](https://huggingface.co/alignment-handbook/zephyr-7b-dpo-full) | 7.5438      | 7.0625     | 7.30 | 34.85        |
| Offline DPO trained (Global BS 128)                                                                               | 7.6125      | 7.20       | 7.41 | 32.65        |
| Online DPO (Iteration 1)                                                                                          | 7.6813      | 6.4750     | 7.07 | 29.05        |
| Online DPO (Iteration 2)                                                                                          | 6.6500      | 6.2875     | 6.46 | 27.45        |
| Online DPO (Iteration 3)                                                                                          | 6.1063      | 5.2375     | 5.67 | 18.05        |



# Results full exp 3 

changes: constant lr which decreases by 0.75 per epoch: 5e-7,  

| Iteration/SFT                                                                                                     | Single Turn | Multi Turn | Mean | MixEval Hard |
| ----------------------------------------------------------------------------------------------------------------- | ----------- | ---------- | ---- | ------------ |
| SFT [alignment-handbook/zephyr-7b-sft-full](https://huggingface.co/alignment-handbook/zephyr-7b-sft-full)         | 6.7000      | 5.8375     | 6.27 | 28.25        |
| Offline DPO [alignment-handbook/zephyr-7b-dpo-full](https://huggingface.co/alignment-handbook/zephyr-7b-dpo-full) | 7.5438      | 7.0625     | 7.30 | 34.85        |
| Offline DPO trained (Global BS 128)                                                                               | 7.6125      | 7.20       | 7.41 | 32.65        |
| Online DPO (Iteration 1)                                                                                          | 7.6813      | 6.4750     | 7.07 | 29.05        |
| Online DPO (Iteration 2)                                                                                          | 6.6500      | 6.2875     | 6.46 | 27.45        |
| Online DPO (Iteration 3)                                                                                          | 6.1063      | 5.2375     | 5.67 | 18.05        |

