from huggingface_hub import HfApi

import os
import structlog
import argparse
from dotenv import load_dotenv

load_dotenv(interpolate=True)
logger = structlog.get_logger("__name__")

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)


def upload_to_hf(
    local_model_dir: str, 
    hf_repo_id: str,
    repo_type: str = "model",
) -> None:
    # Upload model to Hugging Face
    logger.info("Uploading model to Hugging Face")
    api.upload_folder(
        folder_path=local_model_dir,
        repo_id=hf_repo_id,
        repo_type=repo_type,
    )
    logger.info(f"Model uploaded to {hf_repo_id} successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Upload model to Hugging Face Hub"
    )
    parser.add_argument(
        "--local_model_dir",
        type=str,
        required=True,
        help="Path to the local model directory",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., username/repo_name)",
    )
    parser.add_argument(
        "--repo_type",
        type=str,
        default="model",
        help="Type of repository: model, dataset, or space",
    )

    args = parser.parse_args()
    upload_to_hf(
        local_model_dir=args.local_model_dir,
        hf_repo_id=args.hf_repo_id,
        repo_type=args.repo_type,
    )

if __name__ == "__main__":
    main()