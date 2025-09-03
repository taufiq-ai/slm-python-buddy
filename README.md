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
# Syntax: 
uv run python scripts/download_pretrained_model.py --model-name <hf_model_name> --model-dir <dir_to_save_model> --device <device>
# example
uv run python scripts/download_pretrained_model.py --model-name Qwen/Qwen2.5-Coder-1.5B-Instruct --model-dir model --device auto
```

### infer pre-trained models
```bash
uv run infer-pretrained.py "what is list comprehension?" --max_tokens 500 --model_path <path_to_model_dir>  --device auto
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


## Run on Llama Chat
1. Install llama.cpp
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release
```
2. Merge 4bit base model and lora adapter
```
uv run python
>> from pybuddy import utils
>> utils.merge_ft_model(base_model_path, lora_adapter_path, output_path)
```
3. Convert peft model into GGUF format using llama.cpp
```
uv run python llama.cpp/convert_lora_to_gguf.py \
  --base-model model/ft_model/merged \
  --outfile model/qwen2.5-1.5b-2bit.gguf \
  --outtype q2_k \
  --pad-vocab
```
4. Copy the `.gguf` file on your phone and load with **Llama Chat** App.
