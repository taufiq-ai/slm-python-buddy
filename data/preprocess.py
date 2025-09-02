import json
import structlog
from datasets import Dataset

import pybuddy.settings as settings
from pybuddy.utils import load_tokenizer_from_disk

logger = structlog.get_logger(__name__)

def load_data(filepath: str = "data/data.json"):
    """Load dataset from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def create_dataset_from_json(
    filepath: str = "data/data.json",
    tokenizer_path: str = f"{settings.MODEL_DIR}/{settings.BASEMODEL}",
) -> list:
    logger.info("Loading dataset from JSON file", filepath=filepath)
    tokenizer = load_tokenizer_from_disk(tokenizer_path)
    raw_dataset = load_data(filepath)

    processed_data = []
    for messages in raw_dataset:
        text = tokenizer.apply_chat_template(
            messages,  # list of message: {"role": "...", "content": "..."}
            tokenize=False,
            add_generation_prompt=True,
        )
        processed_data.append({"text": text})
    
    # Convert to HF Dataset
    dataset = Dataset.from_list(processed_data)
    logger.info(
        "Loaded HF Dataset", dataset=dataset, 
        raw_dataset=raw_dataset, processed_data=processed_data
    )
    return dataset
