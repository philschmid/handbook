import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from trl import TrlParser
from vllm import LLM, SamplingParams
from datasets import Dataset


# python /fsx/philipp/alignment-handbook/scripts/run_generate_candidates.py --model_name_or_path alignment-handbook/zephyr-7b-sft-full --dataset_path philschmid/dolly-15k-oai-style --output_dir /fsx/philipp/alignment-handbook --batch_size 8


@dataclass
class ScriptArguments:
    model_name_or_path: str
    dataset_path: str
    output_dir: str
    num_samples: int = 5
    batch_size: int = 1
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_k: int = -1
    top_p: float = 1.0


def validate_dataset(dataset):
    """Validates the input dataset to be in the OAI messages format and that the last response is the assistant turn"""

    def check_last_message(s):
        if s["messages"][-1]["role"] != "assistant":
            raise ValueError("Last message should be assistant message")

    dataset = dataset.map(check_last_message)


def vllm_create_candidates(
    dataset: Dataset,
    model_name_or_path: str,
    num_samples: int,
    max_new_tokens: int,
    batch_size: int = 1,
    **kwargs,
) -> Dataset:
    llm = LLM(
        model=model_name_or_path,
        tokenizer=model_name_or_path,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    # prompt dataset
    tokenizer = llm.get_tokenizer()
    dataset = dataset.map(
        lambda s: {
            "prompt": tokenizer.apply_chat_template(
                s["messages"][:-1], tokenize=False, add_generation_prompt=True
            )
        }
    )

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        n=num_samples,
        temperature=kwargs.get("temperature", 0.7),
        top_k=kwargs.get("top_k", -1),
        top_p=kwargs.get("top_p", 1),
    )

    completions = []
    # Iterate over the dataset with batch size
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating completions"):
        batch = dataset[i : i + batch_size]
        # Generate `num_samples` candidates per batch
        result = llm.generate(batch["prompt"], sampling_params, use_tqdm=False)
        for j in range(0, len(batch["prompt"])):
            original = batch["messages"][j]
            candidate = {"original": original, "candidates": []}
            # iterate each candidate and assemble conversation
            for cand in result[j].outputs:
                conversation = original[:-1] + [
                    {"role": "assistant", "content": cand.text}
                ]
                candidate["candidates"].append(conversation)
            completions.append(candidate)
    return Dataset.from_list(completions)


def main():
    parser = TrlParser((ScriptArguments))
    script_args = parser.parse_args_and_config()[0]
    script_args = cast(ScriptArguments, script_args)

    # Create output dir
    Path(script_args.output_dir).mkdir(parents=True, exist_ok=True)

    # load dataset and tokenizer
    if script_args.dataset_path.endswith(".json"):
        dataset = load_dataset(
            "json", data_files=script_args.dataset_path, split="train"
        )
    else:
        dataset = load_dataset(script_args.dataset_path, split="train")
    validate_dataset(dataset)
    dataset = dataset.shuffle().select(range(2000))

    print(
        f"Generating {script_args.num_samples} candidates for {len(dataset)} prompts..."
    )
    start_time = time.time()
    candidates_ds = vllm_create_candidates(
        dataset,
        model_name_or_path=script_args.model_name_or_path,
        num_samples=script_args.num_samples,
        max_new_tokens=script_args.max_new_tokens,
        batch_size=script_args.batch_size,
        temperature=script_args.temperature,
        top_p=script_args.top_p,
        top_k=script_args.top_k,
    )
    print(
        f"Generated {len(dataset) * script_args.num_samples} completions in {time.time() - start_time:.2f} seconds."
    )
    candidates_ds.to_json(f"{script_args.output_dir}/candidates.json")


if __name__ == "__main__":
    main()
