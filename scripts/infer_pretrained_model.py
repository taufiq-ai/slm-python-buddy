from transformers import AutoModelForCausalLM, AutoTokenizer

from pybuddy.utils import (
    load_model_from_disk
)

import os
import argparse
import structlog
import time 
from typing import Union
from dotenv import load_dotenv

load_dotenv(interpolate=True)
logger = structlog.get_logger("__name__")

MODEL_DIR = os.getenv("MODEL_DIR")
BASEMODEL = os.getenv("BASEMODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")


def infer_pretrained(
    prompt: Union[str, dict],
    max_tokens: int = 1024,
    model: AutoModelForCausalLM = None,
    tokenizer: AutoTokenizer = None,
) -> str:
    if isinstance(prompt, str):
        messages = [
            {
                "role": "system", 
                "content": "You're a helpfull code assistant. Help user in Python Programming Language. If you do not know something, let user know and say user to check online."
            },
            {
                "role": "user", "content": prompt
            }
        ]
    else:
        messages = prompt

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    logger.info(
        "Generating Response", 
        chat_template_text=text, 
        prompt=prompt,
        accelerator=model.device,
    )

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(
        generated_ids, 
        skip_special_tokens=True
    )[0]
    logger.info(
        "Generation Done", 
        response=response, 
        # generated_ids=generated_ids,
    )
    return response



# Chat

def chat(model, tokenizer):
    max_tokens = int(input("Enter max_tokens limit: "))
    messages = [
        {
            "role": "system", 
            "content": "You're a helpfull code assistant. Help user in Python Programming Language. If you do not know something, let user know and say user to check online."
        },
    ]

    while True:
        start_time = time.time()
        
        prompt = input("User: ")
        user_message = {"role": "user", "content": prompt}
        messages.append(user_message)
        
        response = infer_pretrained(
            prompt=messages, max_tokens=max_tokens, 
            model=model, tokenizer=tokenizer
        )
        assitant_message = {"role": "assistant", "content": response}
        messages.append(assitant_message)

        runtime = time.time()-start_time
        print(f"\nBot: runtime -> {runtime}s; context_len -> {len(messages)}; \n{response}\n")
    


def cmd():
    # Usage: `uv run infer-pretrained.py "what is list comprehension?"`
    parser = argparse.ArgumentParser(description="Test pre-trained version of Qwen")
    logger.info("Running Inference Script...")
    parser.add_argument("prompt", type=str, help="Prompt")
    parser.add_argument("--max_tokens", type=int, default=1024, help="max output tokens")
    parser.add_argument("--model_path", type=str, default=f"{MODEL_DIR}/{BASEMODEL}", help="path to pretrained model directory")
    parser.add_argument("--device", type=str, default="auto", help="device to run the model on")
    args = parser.parse_args()

    p_model, tokenizer = load_model_from_disk(
        model_path=args.model_path, 
        device=args.device,
    )

    response = infer_pretrained(
        prompt=args.prompt, max_tokens=int(args.max_tokens), 
        model=p_model, tokenizer=tokenizer
    )
    print(f"Response: {response}")


if __name__ == "__main__":
    # p_model, tokenizer = load_model_from_disk(
    #     model_dir="models/Qwen/Qwen2.5-Coder-1.5B-Instruct", 
    #     device="auto"
    # )
    # chat(model=p_model, tokenizer=tokenizer)
    cmd()
