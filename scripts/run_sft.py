#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""
Supervised fine-tuning script for decoder language models.
"""

import logging
import random
import os
import torch
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    SftArguments,
    ModelArguments,
    apply_chat_template,
    get_checkpoint,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from trl import SFTTrainer, TrlParser, ModelConfig
from alignment.utils import setup_logging
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM


_logger = logging.getLogger(__name__)


def merge_peft_model(adapter_dir, save_dir):
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir,
        low_cpu_mem_usage=True,
    )
    print("Merging adapter and base model...")
    merged_model = model.merge_and_unload()  # merge adapter and base model
    merged_model.save_pretrained(save_dir, max_shard_size="3GB")


def sft_main(
    model_args: ModelArguments, data_args: DataArguments, training_args: SftArguments
):

    ###############
    # Setup logging
    ###############
    logger = setup_logging(training_args.get_process_log_level(), _logger)
    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ###############
    # Load datasets
    ###############
    if data_args.dataset_id_or_path.endswith(".json"):
        train_dataset = load_dataset(
            "json", data_files=data_args.dataset_id_or_path, split="train"
        )
    else:
        train_dataset = load_dataset(
            data_args.dataset_id_or_path, split=data_args.dataset_splits
        )

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################
    train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "sft",
            "auto_insert_empty_system_msg": True,
        },
        desc="Formatting comparisons with prompt template",
    )
    # remove all columns except chosen, rejected
    logger.info(f"Columns: {train_dataset.features.keys()}")
    train_dataset = train_dataset.select_columns(["text"])

    # print random sample on rank 0
    if training_args.distributed_state.is_main_process:
        for index in random.sample(range(len(train_dataset)), 2):
            logger.info(
                f"Sample {index} of the processed training set:\n\n{train_dataset[index]['text']}"
            )
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to print

    #######################
    # Load pretrained model
    #######################
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=None if quantization_config is None else get_kbit_device_map(),
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    peft_config = get_peft_config(model_args)

    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        dataset_kwargs=training_args.dataset_kwargs,
    )
    if trainer.accelerator.is_main_process and peft_config:
        trainer.model.print_trainable_parameters()

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    if trainer.is_fsdp_enabled and peft_config:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True
    if model_args.merge_adapter and peft_config:
        adapter_dir = os.path.join(training_args.output_dir, "adapter")
        trainer.model.save_pretrained(adapter_dir)
        logger.info(f"Adapters saved to {adapter_dir}")
        logger.info("Merging adapter and base model...")
        if trainer.accelerator.is_main_process:
            merge_peft_model(adapter_dir, training_args.output_dir)
        # merge adapter and base model
    else:
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # merge adapters

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelArguments, DataArguments, SftArguments))
    model_args, data_args, training_args = parser.parse_args_and_config()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Run the main training loop
    sft_main(model_args, data_args, training_args)


if __name__ == "__main__":
    main()
