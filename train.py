import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from datasets import load_dataset

# Paths
base_model_path = "./base_model"
ft_model_path = "./ft_model"
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


# Download base model once
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
model.save_pretrained(base_model_path)
tokenizer.save_pretrained(base_model_path)

# Load from disk for PEFT
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load data
dataset = load_dataset("json", data_files="synthetic_data.jsonl", split="train")

# Tokenize function
def tokenize_function(examples):
    inputs = tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = tokenizer(examples["completion"], padding="max_length", truncation=True, max_length=128)["input_ids"]
    return inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training args
training_args = TrainingArguments(
    output_dir=ft_model_path,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10,
    save_total_limit=2
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Train
trainer.train()

# Save FT model
model.save_pretrained(ft_model_path)
tokenizer.save_pretrained(ft_model_path)

# Reuse FT model later: AutoModelForCausalLM.from_pretrained(ft_model_path)