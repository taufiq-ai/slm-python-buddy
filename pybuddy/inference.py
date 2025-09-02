from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import argparse
import structlog
from typing import Union

from pybuddy.utils import (
    load_model_from_disk,
    load_tokenizer_from_disk,
)
from pybuddy.optimization import load_4bit_quantized_model

logger = structlog.get_logger("__name__")


def infer_model(
    prompt: Union[str, dict],
    max_tokens: int = 1024,
    model: AutoModelForCausalLM = None,
    tokenizer: AutoTokenizer = None,
) -> str:
    if isinstance(prompt, str):
        messages = [
            {
                "role": "system",
                "content": "You're a helpfull code assistant. Help user in Python Programming Language. If you do not know something, let user know and say user to check online.",
            },
            {"role": "user", "content": prompt},
        ]
    else:
        messages = prompt

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    logger.info(
        "Generating Response",
        chat_template_text=text,
        # prompt=prompt,
        messages=messages,
        # tokenized_model_inputs=model_inputs,
    )

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_tokens)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logger.info(
        "Generation Done",
        response=response,
        # generated_ids=generated_ids,
    )
    return response


def load_ft_model(base_model_path: str, lora_adapter_path: str, device: str = "auto"):
    tokenizer = load_tokenizer_from_disk(model_path=base_model_path)
    model = load_4bit_quantized_model(model_path=base_model_path, device=device)
    model = PeftModel.from_pretrained(model=model, model_id=lora_adapter_path)
    return model, tokenizer


def cmd():
    # Usage: python inference.py "what is loop?" --max_tokens 512 --base-model model/Qwen/Qwen2.5-Coder-1.5B-Instruct --ftmodel model/ft_model/lora-adapter --device auto
    parser = argparse.ArgumentParser(description="Infer trained model")
    logger.info("Running Inference Script...")
    parser.add_argument("prompt", help="Prompt")
    parser.add_argument("--max_tokens", default=1024, help="max output tokens")
    parser.add_argument(
        "--base-model",
        default="model/Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Path to base model",
    )
    parser.add_argument(
        "--ftmodel",
        default="model/ft_model/lora-adapter",
        help="Path to the peft model's adapter",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run the model on (e.g., 'cpu', 'cuda', 'auto')",
    )
    args = parser.parse_args()
    model, tokenizer = load_ft_model(
        base_model_path=args.base_model,
        lora_adapter_path=args.ftmodel,
        device=args.device,
    )
    infer_model(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        model=model,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    cmd()
