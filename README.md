## Quick start

```sh
pipx install uv
git clone https://github.com/taufiq-ai/slm-python-buddy.git
uv venv
uv sync
```

## Usage
```bash
# Download a pre-trained model
# Syntax: uv run python src/download-model.py --model-name <hf_model_name> --model-dir <dir_to_save_model> --device <device>
uv run python scripts/download_pretrained_model.py --model-name Qwen/Qwen2.5-Coder-1.5B-Instruct --model-dir model --device auto
```

```bash
# infer pre-trained models
uv run pybuddy/inference.py
```
