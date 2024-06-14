import logging
import os
import time
from pathlib import Path
from typing import Dict, List, cast

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from trl import TrlParser
from datasets import Dataset
from alignment.configs import RankingArguments
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# CUDA_VISIBLE_DEVICES=0 python scripts/iterative_dpo/run_rank_candidates.py \
# --rank_model_name_or_path RLHFlow/ArmoRM-Llama3-8B-v0.1 \
# --rank_trust_remote_code True \
# --dataset_path test/iterative_dpo/iteration_0/candidates.json \
# config file example
# CUDA_VISIBLE_DEVICES=0 python scripts/iterative_dpo/run_rank_candidates.py --config recipes/iterative_dpo/dev.yaml


class ArmoRMPipeline:
    def __init__(
        self,
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        truncation=True,
        trust_remote_code=False,
        max_length=4096,
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return {"score": score}


class RMPipeline:
    def __init__(
        self,
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        truncation=True,
        trust_remote_code=False,
        max_length=4096,
    ):
        self.rm_pipe = pipeline(
            "sentiment-analysis",
            model_id,
            device_map=device_map,
            model_kwargs={"torch_dtype": torch_dtype},
            trust_remote_code=trust_remote_code,
        )
        self.truncation = truncation
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        prompt = self.rm_pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        )
        output = self.rm_pipe(prompt)
        return {"score": output[0]["score"]}


def rank_candidates_with_seq_model(
    dataset: Dataset,
    model_name_or_path: str,
    trust_remote_code: bool = False,
    **kwargs,
):
    # Load the model
    rm_pipe = ArmoRMPipeline(model_name_or_path, trust_remote_code=trust_remote_code)
    # Iterate over the dataset with batch size
    ranked_completions = []
    for s in tqdm(dataset, desc="Generating scores", total=len(dataset)):
        # score the original message
        original = {"messages": s["original"], "score": rm_pipe(s["original"])["score"]}
        candidates = []
        # iterate over the candidates and score them
        for c in s["candidates"]:
            res = rm_pipe(c)
            candidates.append({"messages": c, "score": res["score"]})

        ranked_completions.append({"original": original, "candidates": candidates})

    return Dataset.from_list(ranked_completions)


def main():
    parser = TrlParser((RankingArguments))
    script_args = parser.parse_args_and_config()[0]
    script_args = cast(RankingArguments, script_args)

    # load dataset and tokenizer
    dataset = load_dataset("json", data_files=script_args.dataset_path, split="train")

    # validate dataset format and that the last message is the assistant message
    start_time = time.time()
    ranking_ds = rank_candidates_with_seq_model(
        dataset,
        model_name_or_path=script_args.rank_model_name_or_path,
        trust_remote_code=script_args.rank_trust_remote_code,
    )
    logging.info(
        f"Ranking {len(ranking_ds)} took {time.time() - start_time:.2f} seconds."
    )

    save_dir = os.path.dirname(script_args.dataset_path)
    ranking_ds.to_json(os.path.join(save_dir, "ranked_candidates.json"))


if __name__ == "__main__":
    main()
