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
