import torch
from dataclasses import dataclass, field
import os
from datasets import load_dataset, Dataset, concatenate_datasets
from trl import TrlParser
from alignment.data import create_pairwise_dpo_dataset

# cli args
# python scripts/iterative_dpo/run_prepare_pairwise_dataset.py \
# --dataset_path "test/iterative_dpo/iteration_0/ranked_candidates.json" \
# --current_iteration 0


@dataclass
class PrepareDatasetArguments:
    dataset_path: str = field(
        metadata={"help": "Path to the ranked dataset"}, default=None
    )
    current_iteration: int = field(
        metadata={"help": "The current iteration number"}, default=0
    )
    pair_strategy: str = field(
        metadata={
            "help": 'The strategy to create the pairwise dataset, one of "random", "max_random", "max_min", "max_max"'
        },
        default="max_min",
    )


def create_previous_dataset_path(current_path: str, iteration: int) -> str:
    """Create the path to the previous dataset."""

    parts = current_path.split(os.sep)
    # Find the part that matches the iteration pattern
    for i, part in enumerate(parts):
        if part.startswith("iteration_"):
            parts[i] = f"iteration_{iteration}"
            break

    # Join the parts back into a single path
    return os.path.join(*parts)


def main():
    parser = TrlParser((PrepareDatasetArguments), ignore_extra_args=True)
    script_args = parser.parse_args_and_config()[0]

    # Load Dataset
    dataset = load_dataset("json", data_files=script_args.dataset_path, split="train")
    print(f"Loaded dataset with {len(dataset)} samples")

    if script_args.current_iteration > 0:
        print(
            f"Loading previous datasets for iteration {script_args.current_iteration}"
        )
        # loop over all iteration reverse until 0 and load the dataset
        previous_datasets = []
        percentage = 0.3
        # trying to iteraively load the previous datasets
        for i in range(script_args.current_iteration - 1, 0, -1):
            print(
                f"Loading dataset from iteration {i}, but only {percentage} percentage"
            )
            previous_dataset_path = create_previous_dataset_path(
                script_args.dataset_path, i
            )
            prev_dataset = load_dataset(
                "json",
                data_files=previous_dataset_path,
                split="train",
            )
            # shuffle the dataset and select a percentage of the dataset
            prev_dataset = dataset.shuffle().select(
                range(int(len(prev_dataset) * percentage))
            )
            previous_datasets.append(prev_dataset)
            # reduce the percentage of the dataset to be used in the next iteration
            percentage = percentage / 2

        # concat the previous datasets
        previous_datasets.append(dataset)
        dataset = concatenate_datasets(previous_datasets)
        print(
            f"Updated and included previous datasets to a new length of {len(dataset)} samples"
        )

    # Create pairwise dataset with the best two scores
    pairwise_ds = create_pairwise_dpo_dataset(
        dataset, choose_type=script_args.pair_strategy
    )
    # save the pairwise dataset
    save_dir = os.path.dirname(script_args.dataset_path)
    pairwise_ds.to_json(os.path.join(save_dir, "pairwise.json"))


if __name__ == "__main__":
    main()
