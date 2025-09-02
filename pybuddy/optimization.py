import torch
from transformers import (
    BitsAndBytesConfig, AutoModelForCausalLM
)
from peft import (
    LoraConfig, 
    get_peft_model,
    prepare_model_for_kbit_training,
)

import structlog

from pybuddy import settings
from pybuddy.utils import show_model_info, load_model_from_disk


logger = structlog.get_logger(__name__)


def load_4bit_quantized_model(
    model_path: str = f"{settings.MODEL_DIR}/{settings.BASEMODEL}",
    device: str = "auto", # Use "cpu" if you want to force CPU
    load_in_4bit: bool = True,
    use_double_quant: bool = True,
    quant_type: str = "nf4",
    dtype: torch.dtype = torch.float16
):
    logger.info(f"Loading 4bit Quantized version of {model_path}")
    # Load with 4bit quantization
    USE_BF16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if USE_BF16 else torch.float16  # If your GPU supports BF16 → use it. Otherwise, F16

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=dtype, 
    )
    logger.info("BitsAndBytesConfig for 4bit quantization", bnb_cfg=bnb_cfg)

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
    logger.info("Model loaded with 4bit quantization", model_path=model_path, device=device)
    show_model_info(model)
    return model


def load_peft_model(
    model: AutoModelForCausalLM,
    target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj"],
    r:int = 64,
    lora_alpha:int = 16,
    lora_dropout:float = 0.05,
    bias:str = "none",
    task_type:str = "CAUSAL_LM",
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


