import torch
import os
import structlog
import argparse

from pybuddy import settings
from pybuddy.utils import (
    load_tokenizer_from_disk,
    load_4bit_quantized_model,
    load_model_with_peft_config,
)
from pybuddy.training import train_model
from data.preprocess import create_dataset_from_json

logger = structlog.get_logger(__name__)

# Set environment variable to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def cmd():
    parser = argparse.ArgumentParser(description="Fine-tune a language model with PEFT")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/data.json",
        help="Path to the training dataset (JSON format)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=settings.FTMODEL_DIR,
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-device batch size for training",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum sequence length for tokenization",
    )
    args = parser.parse_args()
    return args


def main():
    args = cmd()
    DEVICE = settings.DEVICE
    dataset_path = args.dataset_path
    model_path = args.model_path
    output_dir = args.output_dir
    batch_size = args.batch_size
    grad_accum = args.grad_accum
    lr = args.lr
    epochs = args.epochs
    max_length = args.max_length

    if DEVICE != "cpu":
        # Clear GPU memory
        torch.cuda.empty_cache()

    dataset = create_dataset_from_json(filepath=dataset_path, tokenizer_path=model_path)
    tokenizer = load_tokenizer_from_disk(model_path=model_path)
    quantized_model = load_4bit_quantized_model(
        model_path=model_path,
        device=DEVICE,
    )
    peft_model = load_model_with_peft_config(
        model=quantized_model,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    train_model(
        model=peft_model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=output_dir,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        epochs=epochs,
        max_length=max_length,
        eval_dataset=None,
        compute_metrics=None,
    )

if __name__ == "__main__":
    main()