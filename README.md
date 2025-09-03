## Quick start

```sh
pipx install uv
git clone https://github.com/taufiq-ai/slm-python-buddy.git
cp .env.example .env
uv venv
uv sync
export PYTHONPATH=.
```

## Usage
### Download a pre-trained model from HuggingFace
```bash
# Syntax: python scripts/download_pretrained_model.py --model-name <hf_model_name> --model-dir <dir_to_save_model> --device <device>
# example
uv run python scripts/download_pretrained_model.py --model-name Qwen/Qwen2.5-Coder-1.5B-Instruct --model-dir model --device auto
```

### infer pre-trained models
```bash
uv run infer-pretrained.py "what is list comprehension?" --max_tokens 500 --model_path <path_to_model>  --device auto
```

### Fine-tune on custom dataset
```bash
uv run python pybuddy/training.py
```

### Inference
1. Single prompt  
```bash
uv run python pybuddy/inference.py "what is loop?" --max_tokens 512 --base-model model/Qwen/Qwen2.5-Coder-1.5B-Instruct --ftmodel <path_to_ft_model> --device auto
```
2. Long chat context window
```bash
uv run python pybuddy/chat.py --base-model <path_to_base_model> --ftmodel <path_to_fine_tuned_model> --device "auto"
```


## Mobile Deployment
Convert into GGUF file for Mobile Phone
```

```
### Option X: MLC LLM
```bash
# install mlc llm
# cpu
uv pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu
# or cuda 12.3
uv pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cu123 mlc-ai-nightly-cu123
```