
# Instructions to run Offline DPO Zephyr-7b

## bash

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/iterative_dpo/run_dpo.py --config recipes/offline_dpo/dev.yaml
```


## Slurm

```bash
# > pwd: handbook
sbatch --job-name=offline_dpo_test_2 --nodes=1 recipes/offline_dpo/launch.slurm recipes/offline_dpo/dev.yaml
```


