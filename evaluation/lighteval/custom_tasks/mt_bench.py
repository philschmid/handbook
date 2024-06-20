import numpy as np
from lighteval.metrics.utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetricGrouping,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
import os
from aenum import extend_enum
from lighteval.metrics import Metrics
from lighteval.metrics.metrics_sample import JudgeLLM

if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

gpt4_judge = SampleLevelMetricGrouping(
    metric=["single_turn", "multi_turn"],
    higher_is_better=True,
    category=MetricCategory.LLM_AS_JUDGE_MULTI_TURN,
    use_case=MetricUseCase.SUMMARIZATION,
    sample_level_fn=JudgeLLM(
        judge_model_name="gpt-4",
        template_path=os.path.join(
            os.getcwd(), "lighteval", "custom_tasks", "mt_bench_judge_prompts.jsonl"
        ),  # evaluation/lighteval/custom_tasks/mt_bench_judge_prompts.jsonl
        multi_turn=True,
    ).compute,
    corpus_level_fn={
        "single_turn": np.mean,
        "multi_turn": np.mean,
    },
)
extend_enum(Metrics, "gpt4_judge", gpt4_judge)


task = LightevalTaskConfig(
    name="mt_bench",
    prompt_function="mt_bench_prompt",  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
    suite=["custom"],
    hf_repo="lighteval/mt-bench",  # alternatively, you can use HuggingFaceH4/mt_bench_prompts (original MT BENCH)
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="",
    few_shots_select="random",
    metric=["gpt4_judge"],
    generation_size=2000,
    stop_sequence=[],
)


_TASKS = [task]

TASKS_TABLE = [task.as_dict() for task in _TASKS]


if __name__ == "__main__":
    print(t["name"] for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
