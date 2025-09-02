from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
import torch

from typing import Optional, Union
from pybuddy import settings


def load_tokenizer_from_disk(
    model_path: str = f"{settings.MODEL_DIR}/{settings.BASEMODEL}",
) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_path)


def load_model_from_disk(
    model_path: str = f"{settings.MODEL_DIR}/{settings.BASEMODEL}",
    device: str = "auto",
    dtype: Union[torch.dtype, str] = "auto",
    quant_config: Union[BitsAndBytesConfig] = None,  # BitsAndBytesConfig for quantization
    local_files_only: bool = True,
    trust_remote_code: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device_map = "auto" if device != "cpu" else None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        quantization_config=quant_config,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if device == "cpu":
        model.to("cpu")
    return model, tokenizer


def show_model_info(model):
    """
    Display key information about a PyTorch model.
    """
    print("=" * 60)
    print(" Model Information")
    print("=" * 60)

    # Device
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = "No parameters"
    print(f"Device: {device}")

    # Memory footprint (if available)
    if hasattr(model, "get_memory_footprint"):
        mem_gb = model.get_memory_footprint() / 1e9
        print(f"Memory Footprint: {mem_gb:.2f} GB")
    else:
        print("Memory Footprint: Not available")

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Frozen Parameters    : {non_trainable_params:,}")

    # First parameter info
    for name, param in model.named_parameters():
        print(f"First Parameter      : {name}")
        print(f"  Shape: {tuple(param.shape)}")
        print(f"  Dtype: {param.dtype}")
        break

    print("=" * 60)
