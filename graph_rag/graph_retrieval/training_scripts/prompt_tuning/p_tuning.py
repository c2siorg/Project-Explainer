import yaml
import os
import glob
from datasets import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
from transformers import TrainingArguments


def find_files_in_repo(repo_path, extensions):
    """
    Finds all files in a local GitHub repository with specified extensions.

    :param repo_path: The path to the local GitHub repository.
    :param extensions: A list of file extensions to include (e.g., ['py', 'md']).
    :return: A list of file paths.
    """
    files = []
    for ext in extensions:
        files.extend(
            glob.glob(os.path.join(repo_path, "**", f"*.{ext}"), recursive=True)
        )
    return files


def read_files(file_paths):
    """
    Reads the contents of the specified files.

    :param file_paths: A list of file paths.
    :return: A list of tuples containing file paths and their contents.
    """
    files_content = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
            files_content.append((path, content))
    return files_content


def merge_files(files, output_file):
    """
    Merges a list of files into a single file.

    :param files: List of tuples containing file paths and their contents.
    :param output_file: The path of the output file.
    """
    with open(output_file, "w", encoding="utf-8") as outfile:
        for path, content in files:
            outfile.write(f"### {path} ###\n")
            outfile.write(content)
            outfile.write("\n\n")


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

file_paths = find_files_in_repo(
    config["Data"]["repo_path"], config["Data"]["extensions"]
)
files_content = read_files(file_paths)
merge_files(files_content, config["Data"]["output_file"])

with open(config["Data"]["output_file"], "r") as f:
    content = f.read()


def data_for_trainig():
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
    print(f"Input chunk lengths: {(outputs['length'])}")
    print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")
    ds = Dataset.from_dict(outputs)
    ds_i = ds.remove_columns(["attention_mask", "length", "overflow_to_sample_mapping"])
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=config["Training"]["masked_language_modelling"]
    )
    return ds_i, data_collator


def get_peft_model():
    foundational_model = AutoModelForCausalLM.from_pretrained(
        config["Model"]["model"], trust_remote_code=True
    )
    generation_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=config["Training"]["num_virtual_tokens"],
        tokenizer_name_or_path=config["Model"]["model"],  # The pre-trained model.
    )
    peft_model_prompt = get_peft_model(foundational_model, generation_config)
    print(peft_model_prompt.print_trainable_parameters())


def create_training_arguments():
    training_args = TrainingArguments(
        output_dir=config["Training"]["output_dir"],
        # max_steps=4,
        save_strategy="steps",
        per_device_train_batch_size=config["Training"][
            "batch_size"
        ],  # Where the model predictions and checkpoints will be written
        # use_cpu=True,  # This is necessary for CPU clusters.
        auto_find_batch_size=config["Training"][
            "auto_batch_size"
        ],  # Find a suitable batch size that will fit into memory automatically
        learning_rate=config["Training"][
            "learning_rate"
        ],  # Higher learning rate than full Fine-Tuning
        num_train_epochs=config["Training"]["num_epochs"],
        push_to_hub=config["Training"]["push_to_hub"],
    )
    return training_args


def create_trainer():
    train_data, data_collator = data_for_trainig()
    model = get_peft_model()
    training_args = create_training_arguments()
    trainer = Trainer(
        model=model,  # We pass in the PEFT version of the foundation model, bloomz-560M
        args=training_args,  # The args for the training.
        train_dataset=train_data,  # The dataset used to tyrain the model.
        data_collator=data_collator,  # mlm=False indicates not to use masked language modeling
    )
    return trainer
