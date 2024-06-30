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
import logging
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    set_seed,
)

from alignment import (
    DataArguments,
    apply_chat_template,
    decontaminate_humaneval,
    get_checkpoint,
    get_tokenizer,
    DpoArguments,
)
from trl import (
    DPOTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    ModelConfig,
)

from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from alignment.utils import setup_logging

_logger = logging.getLogger(__name__)


def merge_peft_model(adapter_dir, save_dir):
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir,
        low_cpu_mem_usage=True,
    )
    print("Merging adapter and base model...")
    merged_model = model.merge_and_unload()  # merge adapter and base model
    merged_model.save_pretrained(save_dir, max_shard_size="3GB")


def dpo_main(
    model_args: ModelConfig, data_args: DataArguments, training_args: DpoArguments
):
    logger = setup_logging(training_args.get_process_log_level(), _logger)

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
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

    #####################################
    # Load tokenizer and process datasets
    #####################################
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################
    train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "dpo",
            "auto_insert_empty_system_msg": True,
        },
        desc="Formatting comparisons with prompt template",
    )
    # remove all columns except chosen, rejected
    print(f"Columns: {train_dataset.features.keys()}")
    train_dataset = train_dataset.select_columns(["prompt", "chosen", "rejected"])

    # Check for last checkpoint for continuing training
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    #######################################
    # Load the model and/or reference model
    #######################################

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
    # Checks wether we use adapters for reference model or not
    if peft_config is None:
        model_name_or_path = (
            training_args.ref_model_name_or_path or model_args.model_name_or_path
        )
        model_ref = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **model_kwargs
        )
    else:
        model_ref = None

    #########################
    # Instantiate DPO trainer
    #########################
    trainer = DPOTrainer(
        model,
        ref_model=model_ref,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=peft_config,
        loss_type=training_args.loss_type,
    )

    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    # Train the model
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True
    if model_args.use_peft:
        adapter_dir = os.path.join(training_args.output_dir, "adapter")
        trainer.model.save_pretrained(adapter_dir)
        logger.info(f"Adapters saved to {adapter_dir}")
        logger.info(f"Merging adapter and base model...")
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
    parser = TrlParser(
        (ModelConfig, DataArguments, DpoArguments), ignore_extra_args=True
    )
    model_args, data_args, training_args, _ = parser.parse_args_and_config()
    print(f"Model Args: {model_args}")
    print(f"Data Args: {data_args}")
    print(f"Training Args: {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Run the main training loop
    dpo_main(model_args, data_args, training_args)


if __name__ == "__main__":
    main()
