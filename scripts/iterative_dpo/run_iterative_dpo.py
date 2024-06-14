import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, cast
from alignment.utils import execute_cli_script
from alignment.configs import (
    CandidateArguments,
    DPOConfig,
    DataArguments,
    H4ArgumentParser,
    IterativeDpoArguments,
    ModelArguments,
    RankingArguments,
)
from datasets import load_dataset

# get current abolute path of the file but without the file
absoulte_path = os.path.dirname(os.path.realpath(__file__))

# python /fsx/philipp/alignment-handbook/scripts/iterative_dpo/main.py --model_name_or_path alignment-handbook/zephyr-7b-sft-full --dataset_path philschmid/dolly-15k-oai-style --output_dir /fsx/philipp/alignment-handbook --batch_size 8

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


def main():
    parser = H4ArgumentParser(
        (
            IterativeDpoArguments,
            CandidateArguments,
            RankingArguments,
            ModelArguments,
            DPOConfig,
        )
    )
    dpo_args, candidate_args, ranking_args, model_args, training_args = parser.parse()
    dpo_args = cast(IterativeDpoArguments, dpo_args)
    candidate_args = cast(CandidateArguments, candidate_args)
    ranking_args = cast(RankingArguments, ranking_args)
    training_args = cast(DPOConfig, training_args)

    # ########################################
    # # Load Dataset & create iteration splits
    # ########################################
    # train_total_dataset = load_dataset(
    #     dpo_args.dataset_path, split=[dpo_args.train_split]
    # )[0]

    # # TODO: remove after testing
    # train_total_dataset = train_total_dataset.select(range(300))
    # # shuffle and split into even sizes for iterations
    # train_total_dataset = train_total_dataset.shuffle(seed=42)
    # iteration_length = len(train_total_dataset) // dpo_args.num_iterations
    # print(f"Total Dataset Size: {len(train_total_dataset)}")
    # print(f"Iteration Length: {iteration_length}")
    # iteration_datasets = []
    # for it in range(dpo_args.num_iterations):
    #     it_ds = train_total_dataset.select(
    #         range(it * iteration_length, (it + 1) * iteration_length)
    #     )
    #     it_ds = it_ds.select_columns(["chosen"])  # only keep the chosen column
    #     iteration_datasets.append(it_ds)

    # ###########################
    # # Start Iterative DPO Loop
    # ###########################
    # # 1. Generate Candidates
    # # 2. Rank Candidates
    # # 3. Generate comparison dataset
    # # 4. Train model with DPO
    # # 5. save DPO model and start next iteration

    for iteration in range(dpo_args.num_iterations):
        print(f"Starting Iteration {iteration+1}/{dpo_args.num_iterations}")
        # create iteration checkpoint directory
        iteration_dir = os.path.join(training_args.output_dir, f"iteration_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)

        # ########################
        # # 1. Generate Candidates
        # ########################
        # print("Generating Candidates")
        # # determine generation model path
        # generation_model_name_or_path = (
        #     dpo_args.model_name_or_path
        #     if iteration == 0
        #     else os.path.join(training_args.output_dir, f"iteration_{iteration-1}")
        # )
        # # save dataset and prepare arguments
        # prompt_dataset_path = os.path.join(iteration_dir, "prompts.json")
        # iteration_datasets[iteration].to_json(prompt_dataset_path)

        # per_iteration_candiate_args = CandidateArguments(
        #     generation_model_name_or_path=generation_model_name_or_path,
        #     input_dataset_path=prompt_dataset_path,
        #     output_dataset_path=iteration_dir,
        #     batch_size=candidate_args.batch_size,
        #     num_samples=candidate_args.num_samples,
        #     max_new_tokens=candidate_args.max_new_tokens,
        #     temperature=candidate_args.temperature,
        #     top_p=candidate_args.top_p,
        #     top_k=candidate_args.top_k,
        # )
        # # generate candidates
        # execute_cli_script(
        #     os.path.join(absoulte_path, "run_generate_candidates.py"),
        #     asdict(per_iteration_candiate_args),
        # )

        # ######################################################
        # # 2. Rank Candidates & 3. Generate comparison dataset
        # ######################################################
        # print("Ranking Candidates")
        # candidates_dataset_path = os.path.join(iteration_dir, "candidates.json")
        # per_iteration_ranking_args = RankingArguments(
        #     rank_model_name_or_path=ranking_args.rank_model_name_or_path,
        #     input_rank_dataset_path=candidates_dataset_path,
        #     output_rank_dataset_path=iteration_dir,
        # )
        # # rank candidates
        # execute_cli_script(
        #     os.path.join(absoulte_path, "run_rank_candidates.py"),
        #     asdict(per_iteration_ranking_args),
        #     entrypoint="CUDA_VISIBLE_DEVICES=0 python",  # use GPU 0
        # )

        #########################
        # 4. Train model with DPO
        #########################
        print("Training DPO Model")

        training_args = {
            **asdict(training_args),
            **asdict(model_args),
            **{
                "eval_strategy": training_args.eval_strategy.value,
                "lr_scheduler_type": training_args.lr_scheduler_type.value,
                "logging_strategy": training_args.logging_strategy.value,
                "save_strategy": training_args.save_strategy.value,
                "optim": training_args.optim.value,
                "hub_strategy": training_args.hub_strategy.value,
            },
            **{
                "output_dir": iteration_dir,
                "num_train_epochs": 1,
            },
        }
        # remove all keys which values are 'None' or false
        training_args = {k: v for k, v in training_args.items() if v is not None}
        del training_args["debug"]
        del training_args["fsdp"]
        del training_args["accelerator_config"]
        del training_args["fsdp_config"]
        del training_args["_n_gpu"]
        del training_args["hub_model_revision"]
        del training_args["report_to"]
        del training_args["lr_scheduler_kwargs"]

        # rank candidates
        execute_cli_script(
            os.path.join(absoulte_path, "run_dpo.py"),
            training_args,
            # entrypoint="accelerate", # for DS
            entrypoint="python",
        )


if __name__ == "__main__":
    main()
