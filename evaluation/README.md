# Model Evaluation

This directory includes instructions and script to run post-training evaluation of trained models. Evaluation is done using [lighteval](https://github.com/huggingface/lighteval/tree/main). Supported tasks are:

* [MT-Bench](#mt-bench): MT-Bench is a benchmark designed to evaluate large language models' performance in multi-turn dialogues. 
* [IFEval](#ifeval): IFEval is a benchmark to evaluate large language models on their ability to follow natural language instructions through around 500 prompts with verifiable instructions​. 
* [MixEval](https://github.com/Psycoy/MixEval/tree/main): _coming soon_
* BHH: _coming soon_
* AGIEval: _coming soon_

## Usage

Before running evaluation, make sure to install the required dependencies by running:

_Note: LightEval is not yet available on PyPI, so you need to install it from the source and then copy some scripts._

```bash
# pip install git+https://github.com/huggingface/lighteval.git langdetect openai
pip install git+https://github.com/huggingface/lighteval@add-gpt-4-judge langdetect openai --upgrade
```

### MT-Bench

MT-Bench is a benchmark specifically designed to evaluate the performance of large language models (LLMs) in multi-turn dialogues. It focuses on assessing various aspects of conversational abilities such as coherence, informativeness, and the ability to follow instructions over multiple interactions. MT-Bench incorporates challenging multi-turn questions and uses strong LLMs as judges to automate and scale the evaluation process. 

You can access the leaderboard [here](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) (MT-Bench paper results, not revised version)

_NOTE 🚨: We are using a revised version of [MT-Bench from Inflection AI](https://github.com/InflectionAI/Inflection-Benchmarks), which found that "a large fraction—nearly 25%—of examples in the reasoning, math, and coding categories had incorrect reference solutions or questions with flawed premises."_  

LightEval does not use varying temperature based on the sample we are evaluating. All samples are generated using do_sample=False and temperature set to 0.0.

To run evaluation on MT-Bench, you can use the following command:

```bash
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
accelerate launch lighteval/run_lighteval.py \
    --model_args "pretrained=/fsx/philipp/alignment-handbook/test/offline_dpo" \
    --tasks "custom|mt_bench|0|0" \
    --custom_tasks "lighteval/custom_tasks/mt_bench.py"  \
    --use_chat_template \
    --override_batch_size 8
```

**slurm**

```bash
# > pwd: handbook
sbatch --job-name=mt_bench --nodes=1 evaluation/slurm/mt_bench.slurm /fsx/philipp/alignment-handbook/test/offline_dpo-2
```


Should lead to 
| Task              | Version | Metric      |  Value |     | Stderr |
| ----------------- | ------: | ----------- | -----: | --- | -----: |
| custom:mt_bench:0 |       0 | single_turn | 4.4313 | ±   | 0.3695 |
|                   |         | multi_turn  | 3.2500 | ±   | 0.3022 |


### IFEval

IFEval, or Instruction-Following Evaluation, is a benchmark designed to assess the ability of large language models (LLMs) to follow natural language instructions accurately. It consists of a set of "verifiable instructions," such as writing a text of a specific length or including specific keywords. The benchmark includes around 500 prompts with 25 different types of instructions, and it evaluates models on their strict and loose adherence to these instructions.

You can find more information and access the IFEval leaderboard [here](https://huggingface.co/spaces/Krisseck/IFEval-Leaderboard).

To run evaluation on IFEval, you can use the following command:

```bash
accelerate launch lighteval/run_lighteval.py \
    --model_args "pretrained=TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --use_chat_template \
    --tasks "extended|ifeval|0|0" \
    --override_batch_size 8
```

local path
```bash
accelerate launch lighteval/run_lighteval.py \
    --model_args "pretrained=/home/ubuntu/alignment-handbook/test/iterative_dpo/iteration_3" \
    --use_chat_template \
    --tasks "extended|ifeval|0|0" \
    --override_batch_size 8
```


Should lead to 

| Task              | Version | Metric                  |  Value |     | Stderr |
| ----------------- | ------: | ----------------------- | -----: | --- | -----: |
| extended:ifeval:0 |       0 | prompt_level_strict_acc | 0.1312 | ±   | 0.0145 |
|                   |         | inst_level_strict_acc   | 0.2302 | ±   | 0.0004 |
|                   |         | prompt_level_loose_acc  | 0.1516 | ±   | 0.0154 |
|                   |         | inst_level_loose_acc    | 0.2554 | ±   | 0.0004 |



### MixEval 

[MixEval](https://github.com/Psycoy/MixEval/) is a A dynamic benchmark evaluating LLMs using real-world user queries and benchmarks, achieving a 0.96 model ranking correlation with Chatbot Arena and costs around $0.6 to run using GPT-3.5 as a Judge.

You can find more information and access the MixEval leaderboard [here](https://mixeval.github.io/#leaderboard).

```bash
# Fork with more losely dependencies
pip install git+https://github.com/philschmid/MixEval --upgrade
```

_Note: If you want to evaluate models that are not included Take a look [here](https://github.com/philschmid/MixEval?tab=readme-ov-file#registering-new-models). Zephyr example [here](https://github.com/philschmid/MixEval/blob/main/mix_eval/models/zephyr_7b_beta.py)._


**Remote Hugging Face model with existing config:**

```bash
# MODEL_PARSER_API=<your openai api key
MODEL_PARSER_API=$(echo $OPENAI_API_KEY) python -m mix_eval.evaluate \
    --data_path hf://zeitgeist-ai/mixeval \
    --model_name zephyr_7b_beta \
    --benchmark mixeval_hard \
    --version 2024-06-01 \
    --batch_size 20 \
    --output_dir results \
    --api_parallel_num 20
```

**Using vLLM/TGI with hosted or local API:**

1. start you environment
```bash
python -m vllm.entrypoints.openai.api_server --model alignment-handbook/zephyr-7b-dpo-full
```

2. run the following command

```bash
MODEL_PARSER_API=$(echo $OPENAI_API_KEY) API_URL=http://localhost:8000/v1 python -m mix_eval.evaluate \
    --data_path hf://zeitgeist-ai/mixeval \
    --model_name local_api \
    --model_path alignment-handbook/zephyr-7b-dpo-full \
    --benchmark mixeval_hard \
    --version 2024-06-01 \
    --batch_size 20 \
    --output_dir results \
    --api_parallel_num 20
```

Takes around 5 minutes to evaluate.

**Local Hugging Face model from path:**

```bash
# MODEL_PARSER_API=<your openai api key>
MODEL_PARSER_API=$(echo $OPENAI_API_KEY) python -m mix_eval.evaluate \
    --data_path hf://zeitgeist-ai/mixeval \
    --model_path my/local/path \
    --output_dir results/agi-5 \
    --model_name local_chat \
    --benchmark mixeval_hard \
    --version 2024-06-01 \
    --batch_size 20 \
    --api_parallel_num 20
```

**Remote Hugging Face model without config and defaults**

_Note: We use the model name `local_chat` to avoid the need for a config file and load it from the Hugging Face model hub._

```bash
# MODEL_PARSER_API=<your openai api key>
MODEL_PARSER_API=$(echo $OPENAI_API_KEY) python -m mix_eval.evaluate \
    --data_path hf://zeitgeist-ai/mixeval \
    --model_path alignment-handbook/zephyr-7b-sft-full \
    --output_dir results/handbook-zephyr \
    --model_name local_chat \
    --benchmark mixeval_hard \
    --version 2024-06-01 \
    --batch_size 20 \
    --api_parallel_num 20
```