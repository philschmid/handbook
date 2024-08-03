import argparse
import os
import subprocess
import yaml
import math
import torch
import sys
import json
from dataclasses import dataclass, asdict
from typing import Optional
from enum import Enum, auto


STATE_FILE = "state.json"


class Step(Enum):
    PREPARE_DATASET = auto()
    GENERATE_CANDIDATES = auto()
    RANK_CANDIDATES = auto()
    GENERATE_DPO_DATASET = auto()
    TRAIN_MODEL = auto()


@dataclass
class State:
    current_iteration: int = 1
    output_dir: str = "state.json"
    last_completed_step: Optional[Step] = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run iterative DPO training")
    parser.add_argument("--config", required=True, help="Path to the config file")
    return parser.parse_args()


def read_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_num_gpus():
    return torch.cuda.device_count()


def run_command(command: str, state: State, step: Step):
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        sys.stdout.flush()

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        print(f"Error: Command '{command}' failed with return code {return_code}")
        exit(1)

    # Update state after successful command execution
    state.last_completed_step = step
    save_state(state)


def save_state(state: State):
    state_dict = asdict(state)
    state_dict["last_completed_step"] = (
        state.last_completed_step.name if state.last_completed_step else None
    )
    with open(state.output_dir, "w") as f:
        json.dump(state_dict, f)


def load_state(output_dir: str) -> State:
    state_path = os.path.join(output_dir, STATE_FILE)
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            data = json.load(f)
            if data["last_completed_step"]:
                data["last_completed_step"] = Step[data["last_completed_step"]]
            return State(**data)
    return State(output_dir=state_path)


def main():
    args = parse_arguments()
    config = read_config(args.config)
    num_iterations = config["num_iterations"]
    output_dir = config["output_dir"]
    initial_learning_rate = config["learning_rate"]
    num_gpus = get_num_gpus()

    print(f"Running {num_iterations} iterations")
    print(f"Output directory: {output_dir}")
    print(f"Initial Learning Rate: {initial_learning_rate}")
    print(f"Number of GPUs: {num_gpus}")

    state = load_state(output_dir=output_dir)
    if state.last_completed_step or state.current_iteration > 1:
        print(
            f"Resuming from iteration {state.current_iteration}, step {state.last_completed_step}"
        )

    # Prepare the dataset if not done already
    if state.last_completed_step is None and state.current_iteration == 1:
        print("Preparing dataset")
        run_command(
            f"python scripts/iterative_dpo/run_prepare_dataset.py --config {args.config}",
            state,
            Step.PREPARE_DATASET,
        )

    for i in range(state.current_iteration, num_iterations + 1):
        print(f"Running iteration {i}")

        # 1. Generate Candidates
        if state.last_completed_step in (None, Step.PREPARE_DATASET):
            if i == 1:
                generation_model_name_or_path = config["model_name_or_path"]
            else:
                generation_model_name_or_path = f"{output_dir}/iteration_{i-1}"

            run_command(
                f"python scripts/iterative_dpo/run_generate_async_candidates.py --config recipes/iterative_dpo/dev.yaml --generation_model_name_or_path {generation_model_name_or_path} --dataset_path {output_dir}/iteration_{i}/prompts.json --data_parallel_size {num_gpus}",
                state,
                Step.GENERATE_CANDIDATES,
            )

        # 2. Rank Candidates
        if state.last_completed_step in (
            None,
            Step.PREPARE_DATASET,
            Step.GENERATE_CANDIDATES,
        ):
            run_command(
                f"python scripts/iterative_dpo/run_rank_candidates.py --config {args.config} --dataset_path {output_dir}/iteration_{i}/candidates.json",
                state,
                Step.RANK_CANDIDATES,
            )

        # 3. Generate DPO dataset
        if state.last_completed_step in (
            None,
            Step.PREPARE_DATASET,
            Step.GENERATE_CANDIDATES,
            Step.RANK_CANDIDATES,
        ):
            run_command(
                f"python scripts/iterative_dpo/run_prepare_pairwise_dataset.py --dataset_path {output_dir}/iteration_{i}/ranked_candidates.json --current_iteration {i-1}",
                state,
                Step.GENERATE_DPO_DATASET,
            )

        # 4. Train model with DPO
        if state.last_completed_step in (
            None,
            Step.PREPARE_DATASET,
            Step.GENERATE_CANDIDATES,
            Step.RANK_CANDIDATES,
            Step.GENERATE_DPO_DATASET,
        ):
            ref_model_name_or_path = config["model_name_or_path"]
            if i == 1:
                model_name_or_path = config["model_name_or_path"]
            else:
                model_name_or_path = f"{output_dir}/iteration_{i-1}"

            learning_rate = initial_learning_rate * math.pow(0.5, i - 1)

            run_command(
                f"ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes {num_gpus} scripts/iterative_dpo/run_dpo.py --config {args.config} --model_name_or_path {model_name_or_path} --output_dir {output_dir}/iteration_{i} --dataset_id_or_path {output_dir}/iteration_{i}/pairwise.json --num_train_epochs 1 --learning_rate {learning_rate} --ref_model_name_or_path {ref_model_name_or_path}",
                state,
                Step.TRAIN_MODEL,
            )

        # Update state for next iteration
        state.current_iteration = i + 1
        state.last_completed_step = None
        save_state(state)

    print(f"Finished running {num_iterations} iterations")


if __name__ == "__main__":
    main()
