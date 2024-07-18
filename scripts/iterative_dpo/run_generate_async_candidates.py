import os
import asyncio
import logging
import subprocess
import time
from typing import List, Dict, cast
import torch
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from alignment.configs import CandidateArguments
from peft import LoraConfig, AutoPeftModelForCausalLM
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
from trl import TrlParser

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# python scripts/iterative_dpo/run_generate_async_server.py \
# --generation_model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
# --dataset_path test/iterative_dpo/iteration_0/prompts.json
# config file example
# python scripts/iterative_dpo/run_generate_async_server.py --config recipes/iterative_dpo/dev.yaml


debug = os.environ.get("DEBUG", False)
if debug:
    devnull = None
else:
    devnull = open(os.devnull, "wb")


def validate_dataset(dataset):
    """Validates the input dataset to be in the OAI messages format and that the last response is the assistant turn"""

    def check_last_message(s):
        if s["messages"][-1]["role"] != "assistant":
            raise ValueError("Last message should be assistant message")

    dataset = dataset.map(check_last_message)


def is_peft_model(path):
    if os.path.exists(path + "/adapter_config.json"):
        config = LoraConfig.from_pretrained(path)
        return config


def merge_peft_model(path):
    model = AutoPeftModelForCausalLM.from_pretrained(
        path,
        low_cpu_mem_usage=True,
    )
    logger.info("Merging adapter and base model...")
    merged_model = model.merge_and_unload()  # merge adapter and base model
    merged_model.save_pretrained(path, max_shard_size="3GB")


class VllmAsync:
    def __init__(
        self,
        model_id: str,
        data_parallel_size: int,
        tensor_parallel_size: int,
        max_num_seqs: int = 200,
        max_concurrent_requests: int = 32,
    ):
        self.model_id = model_id
        self.data_parallel_size = data_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.max_num_seqs = max_num_seqs
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.base_port = 8000
        self.processes = []
        self.clients = []

        # Check if the number of available GPUs is enough
        available_gpus = torch.cuda.device_count()
        required_gpus = self.data_parallel_size * self.tensor_parallel_size
        if available_gpus < required_gpus:
            raise ValueError(
                f"Not enough GPUs available. Required: {required_gpus}, Available: {available_gpus}"
            )

    async def start_servers(self):
        for i in range(self.data_parallel_size):
            gpu_ids = list(
                range(
                    i * self.tensor_parallel_size, (i + 1) * self.tensor_parallel_size
                )
            )
            gpu_ids_str = ",".join(map(str, gpu_ids))
            port = self.base_port + i

            cmd = [
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                self.model_id,
                "--gpu-memory-utilization",
                "0.9",
                "--max-num-seqs",
                str(self.max_num_seqs),
                "--host",
                "127.0.0.1",
                "--tensor-parallel-size",
                str(self.tensor_parallel_size),
                "--port",
                str(port),
            ]

            env = dict(subprocess.os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

            logging.info(f"Starting server on port {port} with GPUs {gpu_ids_str}")
            logging.info(f"Command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=devnull,
                stderr=devnull,
            )
            self.processes.append(process)

            client = AsyncOpenAI(
                base_url=f"http://127.0.0.1:{port}/v1",
                api_key="dummy_key",  # vLLM doesn't require a real API key
            )
            self.clients.append(client)

        # query all clients asynchronously until /models returns 200
        while True:
            i = 0
            if i == 60:  # 10 minutes
                raise RuntimeError("Servers did not start in time")
            print("Waiting for servers to start...")
            tasks = [client.models.list() for client in self.clients]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            if all(isinstance(result, Exception) for result in results):
                time.sleep(10)
            else:
                break

    async def stop_servers(self, wait=3):
        logging.info(
            f"Stopping servers, waiting {wait} seconds for pending requests to finish..."
        )
        time.sleep(wait)
        for process in self.processes:
            process.terminate()

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples: int = 2,
    ):
        if not self.clients:
            raise RuntimeError("Servers not started. Call start_servers() first.")

        async def _generate(messages: str):
            async with self.semaphore:
                client = self.clients[hash(messages[0]["content"]) % len(self.clients)]
                try:
                    response = await client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_samples,
                    )
                    return [
                        {"role": "assistant", "content": choice.message.content}
                        for choice in response.choices
                    ]
                except Exception as e:
                    logging.error(f"Error generating response: {e}")
                    return []

        tasks = [asyncio.create_task(_generate(m)) for m in messages]
        results = []

        for i, future in enumerate(
            tqdm.as_completed(tasks, total=len(tasks), desc="Generating responses")
        ):
            result = await future
            results.append((messages[i], result))

        # Sort results to match the original prompt order
        results.sort(key=lambda x: messages.index(x[0]))
        return [result for _, result in results]


async def main():
    parser = TrlParser((CandidateArguments), ignore_extra_args=True)
    script_args = parser.parse_args_and_config()[0]
    script_args = cast(CandidateArguments, script_args)

    # load dataset and tokenizer
    dataset = load_dataset("json", data_files=script_args.dataset_path, split="train")
    # rename the message column to "messages"
    if script_args.messages_column != "messages":
        dataset = dataset.rename_column(script_args.messages_column, "messages")
    # validate dataset format and that the last message is the assistant message
    validate_dataset(dataset)
    print(
        f"Generating {script_args.num_samples} candidates for {len(dataset)} prompts..."
    )
    # create prompt messages
    dataset = dataset.map(lambda s: {"prompts": s["messages"][:-1]})
    print(dataset.features.keys())
    print("First prompt:", dataset["prompts"][0])

    start_time = time.time()
    client = VllmAsync(
        model_id=script_args.generation_model_name_or_path,
        data_parallel_size=script_args.data_parallel_size,
        tensor_parallel_size=script_args.tensor_parallel_size,
    )
    await client.start_servers()

    try:
        results = await client.generate(
            dataset["prompts"], num_samples=script_args.num_samples
        )
        completions = []
        for original, result in zip(dataset, results):

            candidate = {"original": original["messages"], "candidates": []}
            if len(result) == 0:
                continue
            for cand in result:
                _candidate = original["messages"][:-1]
                _candidate.append(cand)
                candidate["candidates"].append(_candidate)
            completions.append(candidate)
        candidates_ds = Dataset.from_list(completions)

        print(
            f"Generated {len(dataset) * script_args.num_samples} completions in {time.time() - start_time:.2f} seconds."
        )
        save_dir = os.path.dirname(script_args.dataset_path)
        candidates_ds.to_json(os.path.join(save_dir, "candidates.json"))

    finally:
        await client.stop_servers()


if __name__ == "__main__":
    asyncio.run(main())
