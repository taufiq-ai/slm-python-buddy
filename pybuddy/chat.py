import time
import structlog
import argparse

from pybuddy.inference import infer_model, load_ft_model

logger = structlog.get_logger(__name__)


def chat(model, tokenizer, max_tokens=1024 * 2):
    # max_tokens = int(input("Enter max_tokens limit: "))
    messages = [
        {
            "role": "system",
            "content": "You're a helpfull code assistant. Help user in Python Programming Language. If you do not know something, let user know and say user to check online.",
        },
    ]
    while True:
        try:
            start_time = time.time()

            prompt = input("User: ")
            user_message = {"role": "user", "content": prompt}
            messages.append(user_message)

            response = infer_model(
                prompt=messages, max_tokens=max_tokens, model=model, tokenizer=tokenizer
            )
            assitant_message = {"role": "assistant", "content": response}
            messages.append(assitant_message)

            runtime = time.time() - start_time
            print(
                f"\nBot: runtime -> {runtime}s; context_len -> {len(messages)}; \n{response}\n"
            )
        except KeyboardInterrupt:
            logger.warning("Exiting chat loop")
            return


def cmd():
    # Usage: python chat.py --base-model model/Qwen/Qwen2.5-Coder-1.5B-Instruct --ftmodel model/ft_model/lora-adapter --device auto
    parser = argparse.ArgumentParser(description="Infer trained model")
    logger.info("Running Inference Script...")
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
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024 * 2,
        help="Maximum number of tokens to generate in the response",
    )
    args = parser.parse_args()
    model, tokenizer = load_ft_model(
        base_model_path=args.base_model,
        lora_adapter_path=args.ftmodel,
        device=args.device,
    )
    chat(model=model, tokenizer=tokenizer, max_tokens=max_tokens)


if __name__ == "__main__":
    cmd()
