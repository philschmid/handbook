
# Instructions to run Iterative DPO Zephyr-7b

TODO

```bash
./scripts/iterative_dpo/run_iterative_dpo.sh --config recipes/iterative_dpo/dev.yaml
```

## Slurm

```bash
sbatch --job-name=interative_dpo_test_3 --nodes=1 recipes/iterative_dpo/launch.slurm recipes/iterative_dpo/dev.yaml
```


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


# Results 15k data

| Iteration/SFT                                                                                             | Single Turn | Multi Turn | Mean   |
| --------------------------------------------------------------------------------------------------------- | ----------- | ---------- | ------ |
| SFT [alignment-handbook/zephyr-7b-sft-full](https://huggingface.co/alignment-handbook/zephyr-7b-sft-full) | 6.7000      | 5.8375     | 6.2688 |
| Online DPO (Iteration 1)                                                                                  | 6.8875      | 6.5250     | 6.7063 |
| Online DPO (Iteration 2)                                                                                  | 6.0563      | 5.1875     | 5.6219 |
| Online DPO (Iteration 3)                                                                                  | 7.0187      | 5.8250     | 6.4219 |

# Results full

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
