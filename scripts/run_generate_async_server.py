import os
import asyncio
import subprocess
import time
from typing import List, Dict
import torch
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import logging

logging.basicConfig(level=logging.INFO)

debug = os.environ.get("DEBUG", False)
if debug:
    devnull = None
else:
    devnull = open(os.devnull, "wb")


class VllmAsync:
    def __init__(
        self,
        model_id: str,
        data_parallel_size: int,
        tensor_parallel_size: int,
        max_num_seqs: int = 200,
        max_concurrent_requests: int = 16,
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
            logging.info("Waiting for servers to start...")
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
    ):
        if not self.clients:
            raise RuntimeError("Servers not started. Call start_servers() first.")

        async def _generate(messages: str):
            async with self.semaphore:
                client = self.clients[hash(messages[0]["content"]) % len(self.clients)]
                response = await client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                return {
                    "role": "assistant",
                    "content": response.choices[0].message.content,
                }

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
    client = VllmAsync(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        data_parallel_size=1,
        tensor_parallel_size=1,
    )
    await client.start_servers()

    prompts = [[{"role": "user", "content": "What is the capital of France?"}]] * 128

    try:
        results = await client.generate(prompts)
        for prompt, result in zip(prompts, results):
            print(f"Prompt: {prompt[0]['content']}")
            print(f"Response: {result['content']}")
            print("-" * 80)
    finally:
        await client.stop_servers()


if __name__ == "__main__":
    asyncio.run(main())
