"""
This script prepares data from a repository for training a P-tuning model using the PEFT library.
It reads source files, processes them into tokenized chunks, and trains a language model using the specified configuration.

Functions:
- prepare_data: Collects files from a repository, concatenates their content, and saves it to an output file.
- data_for_training: Tokenizes the concatenated content and prepares it for language model training.
- get_peft_model: Initializes and configures a P-tuning model using the specified configuration.
- create_training_arguments: Generates training arguments for the Trainer using the configuration settings.
- create_trainer: Creates a Trainer object with the model, data, and training arguments.
- main: Parses the YAML configuration file and runs the training process.

Requirements:
- A YAML configuration file that specifies model, training, and data parameters.
"""

import argparse
import yaml
import os
import glob
from datasets import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
from transformers import TrainingArguments


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
        A tuple containing the tokenized dataset and the data collator for language model training.
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
    return ds_removed, data_collator


def get_peft_model(config: dict):
    """
    Initializes and configures a P-tuning model using the specified foundational model and prompt tuning configuration.

    Args:
        config: Dictionary containing the model and training configuration.

    Returns:
        A P-tuned model ready for training.
    """

    foundational_model = AutoModelForCausalLM.from_pretrained(
        config["Model"]["model"], trust_remote_code=True
    )
    generation_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=config["Training"]["num_virtual_tokens"],
        tokenizer_name_or_path=config["Model"]["model"],
    )
    peft_model_prompt = get_peft_model(foundational_model, generation_config)
    peft_model_prompt.print_trainable_parameters()
    return peft_model_prompt


def create_training_arguments(config: dict):
    """
    Creates and configures the training arguments for the Trainer object.

    Args:
        config: Dictionary containing the training configuration.

    Returns:
        A TrainingArguments object with the specified settings.
    """

    training_args = TrainingArguments(
        output_dir=config["Training"]["output_dir"],
        save_strategy="steps",
        per_device_train_batch_size=config["Training"]["batch_size"],
        auto_find_batch_size=config["Training"]["auto_batch_size"],
        learning_rate=config["Training"]["learning_rate"],
        num_train_epochs=config["Training"]["num_epochs"],
        push_to_hub=config["Training"]["push_to_hub"],
    )
    return training_args


def create_trainer(
    config: dict, train_data: object, data_collator: object, model: object
):
    """
    Creates a Trainer object for training the model with the provided data and configuration.

    Args:
        config: Dictionary containing the training configuration.
        train_data: The tokenized dataset to be used for training hf Dataset object.
        data_collator: The data collator for handling the tokenized data during training.
        model: The P-tuned model to be trained.

    Returns:
        A Trainer object configured for training the model.
    """

    training_args = create_training_arguments(config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=data_collator,
    )
    return trainer


def main():
    """
    Main function to execute the training pipeline. It parses the YAML configuration file, prepares the data, initializes
    the model, and starts the training process.
    """
    parser = argparse.ArgumentParser(description="Training script for P-tuning model")
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

    train_data, data_collator = data_for_training(content, config)
    model = get_peft_model(config)
    trainer = create_trainer(config, train_data, data_collator, model)

    trainer.train()


if __name__ == "__main__":
    main()
