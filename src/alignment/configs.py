# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, NewType, Optional, Tuple

from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, TrainingArguments
from trl import DPOConfig, ModelConfig

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


DataClassType = NewType("DataClassType", Any)


@dataclass
class ModelArguments(ModelConfig):
    """
    Arguments related to the model loading. For all parameters, see: https://github.com/huggingface/trl/blob/c9d56366ede5990d690f3b7a3f249c434f3633d6/trl/trainer/model_config.py#L8
    Also used for the continued pretraining task.
    """

    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The name or path of the tokenizer to use."},
    )

    merge_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to merge the adapter with the base model."},
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    chat_template: Optional[str] = field(
        default=None, metadata={"help": "The chat template to use."}
    )
    dataset_id_or_path: Optional[str] = field(
        default=None, metadata={"help": "The path to the dataset to use."}
    )

    dataset_mixer: Optional[Dict[str, float]] = field(
        default=None,
        metadata={
            "help": ("Datasets and their proportions to be used for training ift/rl.")
        },
    )
    text_column: Optional[str] = field(
        default="text",
        metadata={
            "help": "The column name to use for the text in the dataset (only used for continued pretraining)."
        },
    )
    dataset_splits: Optional[List[str]] = field(
        default_factory=lambda: ["train", "test"],
        metadata={"help": ("List of train test splits to use in the dataset")},
    )
    dataset_configs: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of dataset config names. If given must be the same length as 'dataset_mixer' keys."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    truncation_side: Optional[str] = field(
        default=None, metadata={"help": "Truncation side to use for the tokenizer."}
    )
    auto_insert_empty_system_msg: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to automatically insert an empty system message as the first message if `system` is mentioned in the chat template."
            )
        },
    )


@dataclass
class SftArguments(TrainingArguments):
    """
    Arguments related to the training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    Also used for the continued pretraining task.
    """

    dataset_kwargs: Optional[Dict[str, Any]] = field(
        default=None, metadata={"help": "Dataset kwargs for the SFTTrainer"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Used by TRL for reward model training, which tries to read this parameter in init."
            )
        },
    )
    logging_first_step: bool = field(
        default=True,
        metadata={
            "help": ("Whether to log and evaluate the first global_step or not.")
        },
    )
    optim: Optional[str] = field(default="adamw_torch")


@dataclass
class DpoArguments(DPOConfig):
    ref_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Huggingface model name or path to model directory, for the reference model."
        },
    )


@dataclass
class ORPOConfig(TrainingArguments):
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length of the sequences in the batch."},
    )
    max_prompt_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length of the prompt."},
    )
    max_completion_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length of the completions."},
    )

    beta: float = field(
        default=0.1,
        metadata={
            "help": "The beta factor in ORPO loss (lambda/alpha in paper/code) that is the weight of the relative loss ratio in the SFT loss."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether or not to disable dropouts in `model`."},
    )

    label_pad_token_id: int = field(
        default=-100,
        metadata={"help": "The label pad token id."},
    )
    padding_value: Optional[int] = field(
        default=None,
        metadata={
            "help": "The padding value if it is different to the tokenizer's pad_token_id."
        },
    )
    truncation_mode: str = field(
        default="keep_end",
        metadata={
            "help": "The truncation mode to use, either `keep_end` or `keep_start`."
        },
    )

    generate_during_eval: bool = field(
        default=False,
        metadata={
            "help": "Whether to sample and log generations during evaluation step."
        },
    )
    is_encoder_decoder: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "If no model is provided, we need to know if the model_init returns an encoder-decoder."
            )
        },
    )

    model_init_kwargs: Optional[Dict] = field(
        default=None,
        metadata={
            "help": (
                "Dict of Optional kwargs to pass when instantiating the model from a string"
            )
        },
    )

    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": ("The number of workers to use to tokenize the data.")},
    )


@dataclass
class CandidateArguments:
    generation_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Huggingface model name or path to model directory, for the model that will be used for generation, defaults to SFT model or previous iteration model."
        },
    )
    dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to the input dataset, that will be used to generate candidates, defaults to previous iteration output dataset."
        },
    )
    messages_column: str = field(
        default="chosen",
        metadata={
            "help": "Column name in the input dataset that contains the messages."
        },
    )
    num_samples: int = field(
        default=5,
        metadata={"help": "Number of samples to generate for each input."},
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for generation."},
    )
    data_parallel_size: int = field(
        default=1,
        metadata={"help": "Data parallel size for generation."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size for generation."},
    )
    max_new_tokens: int = field(
        default=2048,
        metadata={"help": "Maximum number of new tokens to generate."},
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Temperature for generation."},
    )
    top_k: int = field(
        default=-1,
        metadata={"help": "Top-k for generation."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p for generation."},
    )


@dataclass
class RankingArguments:
    rank_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Huggingface model name or path to model directory, for the model that will be used for generation, defaults to SFT model or previous iteration model."
        },
    )
    rank_trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code when loading a model."},
    )

    dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to the input dataset, that will be used to generate candidates, defaults to previous iteration output dataset."
        },
    )
