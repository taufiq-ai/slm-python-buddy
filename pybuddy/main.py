import structlog

from pybuddy import settings
from pybuddy.utils import (
    load_tokenizer_from_disk,
    load_model_from_disk,
)
from pybuddy.optimization import (
    load_4bit_quantized_model,
    load_peft_model,
)
from pybuddy.training import train_model
from data.preprocess import create_dataset_from_json

logger = structlog.get_logger(__name__)


if __name__ == "__main__":
    DEVICE = "cpu"
    dataset_path = "data/data.json"
    model_path = "model/Qwen/Qwen2.5-Coder-1.5B-Instruct"
    output_dir = settings.FTMODEL_DIR
    
    # Create dataset, load pretrained model and tokenizer
    dataset = create_dataset_from_json(filepath=dataset_path, tokenizer_path=model_path)
    tokenizer = load_tokenizer_from_disk(model_path=model_path)
    pretrained_model, tokenizer = load_model_from_disk(
        model_path=model_path,
        device="cpu",
    )

    # Load 4-bit quantized model
    quantized_model = load_4bit_quantized_model(
        model_path=model_path,
        device=DEVICE,
    )
    
    # Load PEFT model
    peft_model = load_peft_model(quantized_model)
        
    # Train the model
    train_model(
        model=peft_model, 
        tokenizer=tokenizer, 
        dataset=dataset, 
        output_dir=output_dir,
    )
