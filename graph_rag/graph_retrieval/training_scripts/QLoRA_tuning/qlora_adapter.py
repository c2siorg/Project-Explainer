import yaml
import os
import glob
import torch
from datasets import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
    set_seed,
    BitsAndBytesConfig,
)


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
    return ds_i, data_collator,tokenizer

def load_base_model(config):
    compute_dtype = getattr(torch,config['BNB_CONFIG']['BNB_4BIT_COMPUTE_DTYPE'])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config['BNB_CONFIG']['USE_NESTED_QUANT'],
    )
    device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        config['MODEL']['MODEL'],
        load_in_8bit=config['MODEL']['LOAD_IN_8BIT'],
        quantization_config=bnb_config,
        device_map=device_map,
        use_cache=False,
        trust_remote_code=True,

    )
    return model

def load_peft_model(model,config):
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        lora_alpha=config['LORA']['LORA_ALPHA'],
        lora_dropout=config['LORA']['LORA_DROPOUT'],
        r=config['LORA']['LORA_R'],
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules=,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def create_training_arguments(config):
    training_args = TrainingArguments(
        output_dir=f"results/{config['TRAINING_ARGUMENTS']['OUTPUT_DIR']}",
        num_train_epochs=3,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=config['TRAINING_ARGUMENTS']['EVAL_FREQ'],
        save_steps=config['TRAINING_ARGUMENTS']['SAVE_FREQ'],
        logging_steps=config['TRAINING_ARGUMENTS']['LOG_FREQ'],
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        learning_rate=config['TRAINING_ARGUMENTS']['LR'],
        lr_scheduler_type=config['TRAINING_ARGUMENTS']['LR_SCHEDULER_TYPE'],
        warmup_steps=config['TRAINING_ARGUMENTS']['NUM_WARMUP_STEPS'],
        gradient_accumulation_steps=config['TRAINING_ARGUMENTS']['GR_ACC_STEPS'],
        gradient_checkpointing=True,
        fp16=config['TRAINING_ARGUMENTS']['FP16'],
        bf16=config['TRAINING_ARGUMENTS']['BF16'],
        weight_decay=config['TRAINING_ARGUMENTS']['WEIGHT_DECAY'],
        # push_to_hub=True,
        include_tokens_per_second=True,
    )
    return training_args

def create_trainer(tokenizer, train_data, data_collator, model):
    training_args=create_training_arguments()
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
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    content = prepare_data(
        config["Data"]["repo_path"],
        config["Data"]["extensions"],
        config["Data"]["output_file"]
    )

    train_data, data_collator, tokenizer = data_for_training(content, config)
    model = load_base_model(config)
    model = load_peft_model(model, config)
    trainer = create_trainer(config, tokenizer, train_data, data_collator, model)

    trainer.train()


if __name__ == "__main__":
    main()




