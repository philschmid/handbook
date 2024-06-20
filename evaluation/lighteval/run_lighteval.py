# COPY From https://github.com/huggingface/lighteval/blob/main/run_evals_accelerate.py to avoid copying the whole repository
""" Example run command:
accelerate config
accelerate launch run_evals_accelerate.py --tasks="leaderboard|hellaswag|5|1" --output_dir "/scratch/evals" --model_args "pretrained=gpt2"
"""
import argparse
import os
from lighteval.main_accelerate import CACHE_DIR, main

# get directory from the file

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_parser():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    task_type_group = parser.add_mutually_exclusive_group(required=True)

    # Model type: either use a config file or simply the model name
    task_type_group.add_argument("--model_config_path")
    task_type_group.add_argument("--model_args")

    # Debug
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--override_batch_size", type=int, default=-1)
    parser.add_argument(
        "--job_id", type=str, help="Optional Job ID for future reference", default=""
    )
    # Saving
    parser.add_argument("--output_dir", required=False)
    parser.add_argument("--push_results_to_hub", default=False, action="store_true")
    parser.add_argument("--save_details", action="store_true")
    parser.add_argument("--push_details_to_hub", default=False, action="store_true")
    parser.add_argument(
        "--public_run",
        default=False,
        action="store_true",
        help="Push results and details to a public repo",
    )
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument(
        "--results_org",
        type=str,
        help="Hub organisation where you want to store the results. Your current token must have write access to it",
    )
    # Common parameters
    parser.add_argument("--use_chat_template", default=False, action="store_true")
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--dataset_loading_processes", type=int, default=1)
    parser.add_argument(
        "--custom_tasks",
        type=str,
        default=None,
        help="Path to a file with custom tasks (a TASK list of dict and potentially prompt formating functions)",
    )
    group.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Id of a task, e.g. 'original|mmlu:abstract_algebra|5' or path to a texte file with a list of tasks",
    )
    parser.add_argument(
        "--num_fewshot_seeds",
        type=int,
        default=1,
        help="Number of trials the few shots",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args, unknowns = parser.parse_known_args()

    if args.output_dir is None:
        model_id = args.model_args.split("=")[1].replace("/", "_")
        print(model_id)
        args.output_dir = os.path.join(dir_path, "results", model_id)

    # main(args)
