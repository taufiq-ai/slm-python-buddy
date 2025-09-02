from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model_from_disk(
    model_dir: str = "models/Qwen/Qwen2.5-Coder-1.5B-Instruct",
    device="auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if device=="cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, 
            torch_dtype="auto",
        ).to("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, 
            torch_dtype="auto", 
            device_map="auto",
        )
    return model, tokenizer
