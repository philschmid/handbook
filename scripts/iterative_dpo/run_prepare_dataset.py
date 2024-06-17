import argparse
from dataclasses import dataclass, field
import os
from datasets import load_dataset
from trl import TrlParser

# cli args
# python scripts/iterative_dpo/run_prepare_dataset.py \
# --dataset_id_or_path "HuggingFaceH4/ultrafeedback_binarized" \
# --dataset_split "train_prefs" \
# --num_iterations 1 \
# --output_dir test/iterative_dpo
# config file example
# python scripts/iterative_dpo/run_prepare_dataset.py --config recipes/iterative_dpo/dev.yaml


@dataclass
class PrepareDatasetArguments:
    dataset_id_or_path: str = field(
        metadata={"help": "The dataset id or path to the dataset"},
        default="HuggingFaceH4/ultrafeedback_binarized",
    )
    dataset_split: str = field(
        metadata={"help": "The dataset split to use"}, default="train_prefs"
    )
    num_iterations: int = field(
        metadata={"help": "The number of iterations to split the dataset into"},
        default=1,
    )
    output_dir: str = field(
        metadata={"help": "The output directory to save the iteration datasets"},
        default="iterative_dpo",
    )


def main():
    parser = TrlParser((PrepareDatasetArguments), ignore_extra_args=True)
    script_args = parser.parse_args_and_config()[0]

    if script_args.dataset_id_or_path.endswith(".json"):
        dataset = load_dataset(
            "json", data_files=script_args.dataset_id_or_path, split="train"
        )
    else:
        dataset = load_dataset(
            script_args.dataset_id_or_path, split=script_args.dataset_split
        )
    # TODO: remove after testing
    dataset = dataset.select(range(300))
    # shuffle and split into even sizes for iterations
    dataset = dataset.shuffle(seed=42)
    iteration_length = len(dataset) // script_args.num_iterations
    print(f"Total Dataset Size: {len(dataset)}")
    print(f"Iteration Length: {iteration_length}")
    iteration_datasets = []
    for it in range(script_args.num_iterations):
        it_ds = dataset.select(
            range(it * iteration_length, (it + 1) * iteration_length)
        )
        it_ds = it_ds.select_columns(["chosen"])  # only keep the chosen column

        save_path = os.path.join(
            script_args.output_dir, f"iteration_{it}", "prompts.json"
        )
        it_ds.to_json(save_path)


if __name__ == "__main__":
    main()
