import logging
import os
import time
from typing import Dict, List, cast

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from trl import TrlParser
from datasets import Dataset
from alignment.configs import RankingArguments
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed

# python scripts/iterative_dpo/run_rank_candidates.py \
# --rank_model_name_or_path RLHFlow/ArmoRM-Llama3-8B-v0.1 \
# --rank_trust_remote_code True \
# --dataset_path test/iterative_dpo/iteration_0/candidates.json
# config file example

logging.basicConfig(level=logging.INFO)


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


def rank_example(example, rm_pipe):
    original = {
        "messages": example["original"],
        "score": rm_pipe(example["original"])["score"],
    }
    candidates = []
    for c in example["candidates"]:
        res = rm_pipe(c)
        candidates.append({"messages": c, "score": res["score"]})
    return {"original": original, "candidates": candidates}


def main():
    parser = TrlParser((RankingArguments), ignore_extra_args=True)
    script_args = parser.parse_args_and_config()[0]
    print(script_args)
    script_args = cast(RankingArguments, script_args)

    # load dataset and tokenizer
    dataset = load_dataset("json", data_files=script_args.dataset_path, split="train")

    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    logging.info(f"Number of available GPUs: {num_gpus}")

    # Pre-load models
    rm_pipes = []
    for i in range(num_gpus):
        rm_pipe = ArmoRMPipeline(
            script_args.rank_model_name_or_path,
            device_map={"": i},
            trust_remote_code=script_args.rank_trust_remote_code,
        )
        rm_pipes.append(rm_pipe)

    start_time = time.time()

    ranked_completions = []
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []

        for i, example in enumerate(tqdm(dataset, desc="Scoring Examples")):
            rm_pipe = rm_pipes[i % num_gpus]
            future = executor.submit(rank_example, example, rm_pipe)
            futures.append(future)

            # collect after 50 results or at the end of the dataset
            if len(futures) == 50 or i == len(dataset) - 1:
                for completed_future in as_completed(futures):
                    ranked_completions.append(completed_future.result())
                futures = []

    ranking_ds = Dataset.from_list(ranked_completions)

    logging.info(
        f"Ranking {len(ranking_ds)} took {time.time() - start_time:.2f} seconds."
    )
    save_dir = os.path.dirname(script_args.dataset_path)
    ranking_ds.to_json(os.path.join(save_dir, "ranked_candidates.json"))


if __name__ == "__main__":
    main()
