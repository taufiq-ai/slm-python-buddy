import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

import structlog
from typing import Optional, Union
from pybuddy import settings


logger = structlog.get_logger(__name__)


def load_tokenizer_from_disk(
    model_path: str = f"{settings.MODEL_DIR}/{settings.BASEMODEL}",
) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_path)


def load_model_from_disk(
    model_path: str = f"{settings.MODEL_DIR}/{settings.BASEMODEL}",
    device: str = "auto",
    dtype: Union[torch.dtype, str] = "auto",
    quant_config: Union[
        BitsAndBytesConfig
    ] = None,  # BitsAndBytesConfig for quantization
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


def load_4bit_quantized_model(
    model_path: str = f"{settings.MODEL_DIR}/{settings.BASEMODEL}",
    device: str = "auto",  # Use "cpu" if you want to force CPU
    load_in_4bit: bool = True,
    use_double_quant: bool = True,
    quant_type: str = "nf4",
    dtype: torch.dtype = torch.float16,
):
    logger.info(f"Loading 4bit Quantized version of {model_path}")
    # Load with 4bit quantization
    USE_BF16 = torch.cuda.is_bf16_supported()
    dtype = (
        torch.bfloat16 if USE_BF16 else torch.float16
    )  # If your GPU supports BF16 → use it. Otherwise, F16

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=dtype,
    )
    # logger.info("BitsAndBytesConfig for 4bit quantization", bnb_cfg=bnb_cfg)

    model, tokenizer = load_model_from_disk(
        model_path=model_path,
        device=device,
        dtype=dtype,
        quant_config=bnb_cfg,  # Pass the BitsAndBytesConfig for quantization
        local_files_only=True,  # Ensure it loads from local files only
        trust_remote_code=True,  # Some models may require this
    )
    # Prepare for k-bit training (memory optimizations)
    model = prepare_model_for_kbit_training(model)
    logger.info(
        "Model loaded with 4bit quantization", model_path=model_path, device=device
    )
    show_model_info(model)
    return model


def load_model_with_peft_config(
    model: AutoModelForCausalLM,
    target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj"],
    r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> AutoModelForCausalLM:
    """
    Setup the PEFT configuration for LoRA.
    """
    logger.info(f"Setting up PEFT config for {model}")
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def load_ft_model_from_disk(
    base_model_path: str, lora_adapter_path: str, device: str = "auto"
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = load_tokenizer_from_disk(model_path=base_model_path)
    model = load_4bit_quantized_model(model_path=base_model_path, device=device)
    model = PeftModel.from_pretrained(model=model, model_id=lora_adapter_path)
    logger.info(
        "Loaded Fine-tuned Model",
        base_model_path=base_model_path,
        lora_adapter_path=lora_adapter_path,
        device=device,
    )
    show_model_info(model)
    return model, tokenizer


def merge_ft_model(
    base_model_path: str = f"{settings.MODEL_DIR}/{settings.BASEMODEL}", 
    lora_adapter_path: str = f"{settings.FTMODEL_DIR}/lora-adapter", 
    use_quantized_base_model: bool = False,
    output_path: str = f"{settings.FTMODEL_DIR}/merged", 
    device="auto",
    dtype="auto",
) -> None:
    if use_quantized_base_model:
        # Merging quantized base model and LoRA Adapter might be problematic to convert into gguf file using llama.cpp
        model, tokenizer = load_ft_model_from_disk(
            base_model_path=base_model_path,
            lora_adapter_path=lora_adapter_path,
            device=device 
        )
    else:
        # Merge BASE_MODEL and LoRA Adapter
        model, tokenizer = load_model_from_disk(
            model_path=base_model_path, 
            device=device,
            dtype=dtype,
        )
        model = PeftModel.from_pretrained(model, lora_adapter_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


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
