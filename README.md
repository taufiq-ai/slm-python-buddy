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

### Infer pre-trained model
```bash
uv run scripts/infer_pretrained_model.py "what is list comprehension?" --max_tokens 500 --model_path <path_to_pretrained_model_dir>  --device <cpu_or_cuda>
```

### Fine-tune on custom dataset
```bash
# More customization TBA
uv run python pybuddy/training.py
```

### Inference
1. Single prompt  
```bash
uv run python pybuddy/inference.py "what is loop?" --max_tokens 512 --base-model <path_to_base_model> --ftmodel <path_to_ft_model> --device "<auto_or_cpu>"
```
2. Long chat context window
```bash
uv run python pybuddy/chat.py --base-model <path_to_base_model> --ftmodel <path_to_fine_tuned_model> --device "<auto_or_cpu>"
```


## Prepare finetuned model to run on Android 
We will use **Llama Chat** android app to load the model easily and do inference.

1. Install `llama.cpp` following their [installation guidelines](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j 8
```
- Troubleshooting: install the dependencies like `cmake, git, wget` etc.

2. Merge base model and lora adapter
```shell
uv run python
>> from pybuddy import utils
>> utils.merge_ft_model(base_model_path, lora_adapter_path, output_path)
```

3. Convert merged peft model into GGUF format using llama.cpp
```sh
# syntax
uv run python llama.cpp/convert_hf_to_gguf.py <ft-model-dir> --outfile <path_to_dir/filename.gguf> --outtype <quantization_type>

# example
uv run python llama.cpp/convert_hf_to_gguf.py model/Qwen/qwen2.5-1.5b-instruct-ft-merged --outfile model/Qwen/qwen2.5-1.5b-q8-03092025.gguf --outtype q8_0
```

4. Test inference with llama.cpp
```sh
# syntax
llama.cpp/build/bin/llama-cli -m <path_to_model/filename.gguf> -p "<prompt>"

# example
llama.cpp/build/bin/llama-cli -m model/Qwen/qwen2.5-1.5b-q8-03092025.gguf -p "What is list comprehension?"

```

5. Copy the `.gguf` file on your phone and load with **Llama Chat** App.


## Upload model to Hugging Face
```shell

```
