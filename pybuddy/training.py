# training.py
# Later - make more customization from CMD
import os
from typing import Optional

import torch
import structlog
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
)
from datasets import Dataset

from pybuddy import settings
from pybuddy.utils import (
    load_tokenizer_from_disk,
    load_model_from_disk,
    load_4bit_quantized_model,
    load_model_with_peft_config,
)

from data.preprocess import create_dataset_from_json


logger = structlog.get_logger(__name__)


def get_data_collator(
    tokenizer: AutoTokenizer, mlm: bool = False
) -> DataCollatorForLanguageModeling:
    """
    Create a DataCollator for causal language modeling (LM).
    This ensures dynamic padding and shifting of labels.
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=mlm  # for causal LM (not masked LM)
    )


def get_training_args(
    output_dir: str = settings.FTMODEL_DIR,
    overwrite_output_dir: str = False,
    batch_size: int = 2,
    grad_accum: int = 8,
    lr: float = 2e-4,
    epochs: int = 3,
    logging_steps: int = 10,
    save_total_limit: int = 3,
) -> TrainingArguments:
    """
    Define Hugging Face TrainingArguments.

    Args:
        output_dir (str): Path to save model checkpoints and logs.
        batch_size (int): Batch size per device (GPU/CPU).
        grad_accum (int): Gradient accumulation steps (effective batch size = batch_size * grad_accum).
        lr (float): Learning rate.
        epochs (int): Number of training epochs.
        logging_steps (int): Frequency of logging.
        save_total_limit (int): Max number of checkpoints to keep.

        Show additional args:
        print(TrainingArguments.__init__.__code__.co_varnames)

    Returns:
        TrainingArguments: Config for HuggingFace Trainer.
    """
    logger.info("Preparing TrainingArguments")
    USE_BF16 = torch.cuda.is_bf16_supported()
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=not USE_BF16,
        bf16=USE_BF16,
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # or your chosen metric
        greater_is_better=False,  # True if higher is better (e.g., accuracy)
    )


def train_model(
    model,
    tokenizer,
    dataset: Dataset,
    output_dir: str = settings.FTMODEL_DIR,
    batch_size: int = 2,
    grad_accum: int = 8,
    lr: float = 2e-4,
    epochs: int = 3,
    max_length: int = 1024 * 2,
    eval_dataset: Optional[Dataset] = None,
    compute_metrics=None,
):
    """
    Train a LoRA-adapted quantized model on dataset.

    Args:
        model: PEFT-wrapped HF model (from optimization.py).
        tokenizer: Corresponding tokenizer.
        dataset (Dataset): Tokenized HF Dataset with "text".
        output_dir (str): Save dir for outputs.
        batch_size (int): Per-device batch size.
        grad_accum (int): Gradient accumulation steps.
        lr (float): Learning rate.
        epochs (int): Epochs for training.
        max_length (int): Max sequence length.
        eval_dataset (Dataset, optional): Validation split.
        compute_metrics (function, optional): Function for evaluation metrics.

    Returns:
        Trainer: Trained HuggingFace Trainer instance.
    """
    logger.info("Tokenizing dataset", max_length=max_length)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized_ds = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            tokenize_fn, batched=True, remove_columns=["text"]
        )

    data_collator = get_data_collator(tokenizer)
    training_args = get_training_args(
        output_dir=output_dir,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        epochs=epochs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        eval_dataset=eval_dataset if eval_dataset else tokenized_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training", trainer=trainer, output_dir=output_dir)
    trainer.train()

    # Save LoRA adapters
    adapter_dir = os.path.join(output_dir, "lora-adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Saved LoRA adapters", path=adapter_dir)

    return trainer


if __name__ == "__main__":
    DEVICE = settings.DEVICE
    dataset_path = "data/data.json"
    model_path = "model/Qwen/Qwen2.5-Coder-1.5B-Instruct"
    output_dir = settings.FTMODEL_DIR

    # Create dataset, load pretrained model and tokenizer
    dataset = create_dataset_from_json(filepath=dataset_path, tokenizer_path=model_path)
    tokenizer = load_tokenizer_from_disk(model_path=model_path)
    pretrained_model, tokenizer = load_model_from_disk(
        model_path=model_path,
        device=DEVICE,
    )

    # Load 4-bit quantized model
    quantized_model = load_4bit_quantized_model(
        model_path=model_path,
        device=DEVICE,
    )

    # Load PEFT model
    peft_model = load_model_with_peft_config(quantized_model)

    # Train the model
    train_model(
        model=peft_model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=output_dir,
    )
