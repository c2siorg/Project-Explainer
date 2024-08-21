"""
This script facilitates the fine-tuning of a language model using QLoRA (Quantized Low-Rank Adapter)
adapter tuning.

The main functionalities include:
- Preparing data from a specified repository with specific file extensions.
- Tokenizing the data for model training.
- Loading and configuring a pre-trained language model.
- Applying PEFT (Parameter-Efficient Fine-Tuning) using QLoRA.
- Defining training arguments and creating a Trainer instance.
- Executing the training process with the Trainer.

Requirements:
- A YAML configuration file that specifies model, training, and data parameters.
"""

import argparse
import yaml
import os
import glob
import torch
from datasets import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)


def prepare_data(repo_path: str, extensions: list, output_file: str):
    """
    Collects files with specified extensions from a repository, concatenates their content, and writes it to an output file.

    Args:
        repo_path: Path to the repository to collect files from.
        extensions: List of file extensions to include in the data preparation.
        output_file: Path to the output file where the concatenated content will be saved.

    Returns:
        A string containing the entire content written to the output file.
    """

    files = []
    for ext in extensions:
        files.extend(
            glob.glob(os.path.join(repo_path, "**", f"*.{ext}"), recursive=True)
        )

    with open(output_file, "w", encoding="utf-8") as outfile:
        for path in files:
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
                outfile.write(f"### {path} ###\n")
                outfile.write(content)
                outfile.write("\n\n")

    with open(output_file, "r") as f:
        return f.read()


def data_for_training(content: str, config: dict):
    """
    Tokenizes the content and prepares it for language model training, including creating a data collator.

    Args:
        content: The concatenated text content to be tokenized.
        config: Dictionary containing the model and training configuration.

    Returns:
        A tuple containing the tokenized dataset,tokenizer,data collator for language model training.
    """
    tokenizer = AutoTokenizer.from_pretrained(config["Model"]["model"])
    context_length = config["Model"]["context_length"]
    outputs = tokenizer(
        content,
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    print(f"Input IDs length: {len(outputs['input_ids'])}")
    print(f"Input chunk lengths: {outputs['length']}")
    print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")
    ds = Dataset.from_dict(outputs)
    ds_removed = ds.remove_columns(
        ["attention_mask", "length", "overflow_to_sample_mapping"]
    )
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=config["Training"]["masked_language_modelling"]
    )
    return ds_removed, data_collator, tokenizer


def load_base_model(config: dict):
    """
    Loads the base language model with specified configurations, including quantization settings.

    Args:
        config: The configuration dictionary containing model and BNB (BitsAndBytes) parameters.

    Returns:
        PreTrainedModel: The loaded pre-trained language model ready for training.
    """

    compute_dtype = getattr(torch, config["BNB_CONFIG"]["BNB_4BIT_COMPUTE_DTYPE"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config["BNB_CONFIG"]["USE_NESTED_QUANT"],
    )
    device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        config["MODEL"]["MODEL"],
        load_in_8bit=config["MODEL"]["LOAD_IN_8BIT"],
        quantization_config=bnb_config,
        device_map=device_map,
        use_cache=False,
        trust_remote_code=True,
    )
    return model


def load_peft_model(model: object, config: dict):
    """
    Applies PEFT (Parameter-Efficient Fine-Tuning) using QLoRA to the given model.

    Args:
        model: The pre-trained language model to be fine-tuned.
        config: The configuration dictionary containing LORA (Low-Rank Adapter) parameters.

    Returns:
        PreTrainedModel: The PEFT-configured model ready for training.
    """

    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        lora_alpha=config["LORA"]["LORA_ALPHA"],
        lora_dropout=config["LORA"]["LORA_DROPOUT"],
        r=config["LORA"]["LORA_R"],
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules=,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def create_training_arguments(config: dict):
    """
    Creates and returns the training arguments for the Trainer.

    Args:
        config: The configuration dictionary containing training arguments.

    Returns:
        TrainingArguments: The configured training arguments.
    """

    training_args = TrainingArguments(
        output_dir=f"results/{config['TRAINING_ARGUMENTS']['OUTPUT_DIR']}",
        num_train_epochs=3,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=config["TRAINING_ARGUMENTS"]["EVAL_FREQ"],
        save_steps=config["TRAINING_ARGUMENTS"]["SAVE_FREQ"],
        logging_steps=config["TRAINING_ARGUMENTS"]["LOG_FREQ"],
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        learning_rate=config["TRAINING_ARGUMENTS"]["LR"],
        lr_scheduler_type=config["TRAINING_ARGUMENTS"]["LR_SCHEDULER_TYPE"],
        warmup_steps=config["TRAINING_ARGUMENTS"]["NUM_WARMUP_STEPS"],
        gradient_accumulation_steps=config["TRAINING_ARGUMENTS"]["GR_ACC_STEPS"],
        gradient_checkpointing=True,
        fp16=config["TRAINING_ARGUMENTS"]["FP16"],
        bf16=config["TRAINING_ARGUMENTS"]["BF16"],
        weight_decay=config["TRAINING_ARGUMENTS"]["WEIGHT_DECAY"],
        # push_to_hub=True,
        include_tokens_per_second=True,
    )
    return training_args


def create_trainer(
    tokenizer: object, train_data: object, data_collator: object, model: object
):
    """
    Creates a Trainer instance with the provided tokenizer, training data, data collator, and model.

    Args:
        tokenizer: The tokenizer to be used during training.
        train_data : The tokenized training dataset.
        data_collator: The data collator for language modeling.
        model : The pre-trained and fine-tuned model.

    Returns:
        Trainer: The Trainer instance for model training.
    """
    training_args = create_training_arguments()
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=train_data,
    )
    return trainer


def main():
    """
    The main function that orchestrates the data preparation, model loading,
    and training processes using the provided YAML configuration.
    """

    parser = argparse.ArgumentParser(
        description="Training script for QLoRA adapter tuning"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    content = prepare_data(
        config["Data"]["repo_path"],
        config["Data"]["extensions"],
        config["Data"]["output_file"],
    )

    train_data, data_collator, tokenizer = data_for_training(content, config)
    model = load_base_model(config)
    model = load_peft_model(model, config)
    trainer = create_trainer(config, tokenizer, train_data, data_collator, model)

    trainer.train()


if __name__ == "__main__":
    main()
