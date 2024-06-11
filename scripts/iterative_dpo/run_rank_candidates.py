import logging
import time
from pathlib import Path
from typing import cast

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from trl import TrlParser
from datasets import Dataset
from alignment.configs import RankingArguments
import torch 
from transformers import pipeline

# python scripts/iterative_dpo/run_rank_candidates.py \
# --rank_model_name_or_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
# --input_rank_dataset_path test/iterative_dpo/iteration_0/candidates.json \
# --output_rank_dataset_path test/iterative_dpo/iteration_0

def create_pairwise_dataset(dataset: Dataset) -> Dataset:
    def create_pair(s):
        arr = [s["original"]] + s["candidates"]
        tensor = torch.tensor([s["score"] for s in arr])
        _, top2_indices = torch.topk(tensor, 2)
        top2_indices.tolist()

        return  {
            "chosen": arr[top2_indices[0]]["messages"],
            "rejected": arr[top2_indices[1]]["messages"]
        }
    return dataset.map(create_pair, remove_columns=dataset.features)


def rank_candidates_with_seq_model(
    dataset: Dataset,
    model_name_or_path: str,
    **kwargs,
): 
    # Load the model
    rm_pipe = pipeline(
        "sentiment-analysis",
        model_name_or_path,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.bfloat16},
        truncation=True,
    )

    pipe_kwargs = {
        "top_k": None,
        "function_to_apply": "none",
        "batch_size": len(dataset[0]["candidates"])+1
    }

    def format(s):
        return rm_pipe.tokenizer.apply_chat_template(s, tokenize=False, add_generation_prompt=False).replace(rm_pipe.tokenizer.bos_token, "")

    # Iterate over the dataset with batch size
    ranked_completions=[]
    for s in tqdm(dataset, desc="Generating scores",total=len(dataset)):
        batch = []
        for c in [s["original"]] + s["candidates"]:
            batch.append(format(c))
        res = rm_pipe(batch, **pipe_kwargs)
        
        scores = [output[0]["score"] for output in res]
        # convert to list with scores 
        result = {
            "original": {
                "messages": s["original"],
                "score": scores[0]
                },
            "candidates": [
            {
                "messages": c,
                "score": scores[i+1]
            } for i, c in enumerate(s["candidates"])
            ]
        }
        ranked_completions.append(result)
        
    return Dataset.from_list(ranked_completions)    

def main():
    parser = TrlParser((RankingArguments))
    script_args = parser.parse_args_and_config()[0]
    script_args = cast(RankingArguments, script_args)

    # Create output dir
    Path(script_args.output_rank_dataset_path).mkdir(parents=True, exist_ok=True)

    # load dataset and tokenizer
    if script_args.input_rank_dataset_path.endswith(".json"):
        dataset = load_dataset(
            "json", data_files=script_args.input_rank_dataset_path, split="train"
        )
    else:
        dataset = load_dataset(script_args.input_rank_dataset_path, split="train")
        
    # validate dataset format and that the last message is the assistant message
    start_time = time.time()
    ranking_ds = rank_candidates_with_seq_model(
        dataset,
        model_name_or_path=script_args.rank_model_name_or_path,
    )
    logging.info(
           f"Ranking {len(ranking_ds)} took {time.time() - start_time:.2f} seconds."
    )
    ranking_ds.to_json(f"{script_args.output_rank_dataset_path}/scores.json")
    # Save Pairwise data and raw data
    
    pairwise_ds = create_pairwise_dataset(ranking_ds)
    pairwise_ds.to_json(f"{script_args.output_rank_dataset_path}/pairwise.json")


if __name__ == "__main__":
    main()
