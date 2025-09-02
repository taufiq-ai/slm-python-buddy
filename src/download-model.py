import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import structlog

logger = structlog.get_logger(__name__)


def download_model(model_name: str, model_dir: str, device="auto"):
    """
    Download model from HuggingFace and Save into model_dir.
    """
    model_path=f"{model_dir}/{model_name}"
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return 
    os.makedirs(model_path)
    logger.info(f"Downloading model on '{model_path}'", model_name=model_name, device=device)
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
        ).to("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model {model_name} downloaded and saved to {model_path}")


def main():
    parser = argparse.ArgumentParser(description="HuggingFace Model management script")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Model name to download")
    parser.add_argument("--model-dir", default="models", help="Path to save model")
    parser.add_argument("--device", default="auto", help="Accelator device: cpu or auto ")
    args = parser.parse_args()

    download_model(args.model_name, args.model_dir, args.device)

if __name__ == "__main__":
    main()
