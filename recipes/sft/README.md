
# Instructions to run supervised fine-tuning (SFT)

See below for commands to train these models using either DeepSpeed ZeRO-3 or FSDP Q-LoRA.

## Single-GPU

## Multi-GPU

### Full training examples

You will require 8 GPUs (80GB of VRAM) to train the full model.
```shell
accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes=4 scripts/run_sft.py --config recipes/sft/config_full.yaml
```

### Slurm 

```bash
sbatch --job-name=sft --nodes=1 recipes/sft/launch.slurm recipes/sft/config_full.yaml recipes/accelerate_configs/deepspeed_zero3.yaml
```

### QLoRA training examples (WIP)

```shell
accelerate launch --config_file recipes/accelerate_configs/fsdp_qlora.yaml --num_processes=4 scripts/run_sft.py --config recipes/sft/config_full.yaml
```