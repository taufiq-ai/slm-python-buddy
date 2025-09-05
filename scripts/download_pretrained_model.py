import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import structlog
from pybuddy import settings

logger = structlog.get_logger(__name__)


def download_model(model_name: str, model_dir: str = settings.MODEL_DIR, device="auto"):
    """
    Download model from HuggingFace and save into model_dir.
    """
    model_path = f"{model_dir}/{model_name}"
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        print(f"Model already exists at {model_path}")
        return
    
    os.makedirs(model_path, exist_ok=True)
    logger.info(f"Downloading model '{model_name}' on '{model_path}'", extra={"device": device})

    device_map = "auto" if device != "cpu" else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device_map
    )
    if device == "cpu":
        model.to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model {model_name} downloaded and saved to {model_path}")


def main():
    # Usage: python scripts/download_pretrained_model.py --model-name <model_name> --model-dir <dir_to_save_model> --device <device>
    parser = argparse.ArgumentParser(description="HuggingFace Model management script")
    parser.add_argument("--model-name", type=str, default=settings.BASEMODEL, help="Model name to download")
    parser.add_argument("--model-dir", type=str, default=settings.MODEL_DIR, help="Path to save model")
    parser.add_argument("--device", type=str, default="auto", help="Accelator device: cpu or auto ")
    args = parser.parse_args()

    download_model(args.model_name, args.model_dir, args.device)

if __name__ == "__main__":
    main()
