import yaml
import os
import glob
from datasets import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
from transformers import TrainingArguments


def prepare_data(repo_path, extensions, output_file):

    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(repo_path, "**", f"*.{ext}"), recursive=True))

    with open(output_file, "w", encoding="utf-8") as outfile:
        for path in files:
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
                outfile.write(f"### {path} ###\n")
                outfile.write(content)
                outfile.write("\n\n")

    with open(output_file, "r") as f:
        return f.read()


def data_for_training(content, config):
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
    ds_i = ds.remove_columns(["attention_mask", "length", "overflow_to_sample_mapping"])
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=config["Training"]["masked_language_modelling"]
    )
    return ds_i, data_collator


def get_peft_model(config):
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


def create_training_arguments(config):
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


def create_trainer(config, train_data, data_collator, model):
    training_args = create_training_arguments(config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=data_collator,
    )
    return trainer


def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    content = prepare_data(config["Data"]["repo_path"], config["Data"]["extensions"], config["Data"]["output_file"])

    train_data, data_collator = data_for_training(content, config)
    model = get_peft_model(config)
    trainer = create_trainer(config, train_data, data_collator, model)

    trainer.train()


if __name__ == "__main__":
    main()
