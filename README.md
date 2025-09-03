# SLM-Python-Buddy

A Python framework to build and deploy a fine-tuned, on-device small language model (SLM) for code assistance. Which is optimized for low-resource Android devices (e.g., Redmi Note 11, 4GB RAM).

## Why This Project?

- **Goal**: Enable an offline, lightweight Language Model based Python code assistant for mobile devices.
- **Process**: Fine-tune `Qwen2.5-Coder-1.5B-Instruct` model with PEFT, convert to GGUF (~800MB-1.5GB), and deploy for fast, offline inference.
- **Use Case**: Assist Python learners with coding queries on low-resource devices.
- **Why GGUF**: Compact format for mobile compatibility, balancing quality and size.

*NOTE: The finetuned guff model files are available here: https://huggingface.co/taufiq-ai/qwen2.5-coder-1.5-instruct-ft*


## Quick Start

1. **Install Dependencies**:
   ```bash
   pipx install uv
   git clone https://github.com/taufiq-ai/slm-python-buddy.git
   cd slm-python-buddy
   cp .env.example .env
   uv venv
   uv sync
   export PYTHONPATH=.
   ```

## Usage

### 1. Download Pre-trained Model
```bash
# Syntax:
uv run python scripts/download_pretrained_model.py --model-name <hf_model_name> --model-dir <dir> --device <auto|cpu|cuda>
# Example:
uv run python scripts/download_pretrained_model.py --model-name Qwen/Qwen2.5-Coder-1.5B-Instruct --model-dir model --device auto
```

### 2. Infer Pre-trained Model
```bash
# Syntax:
uv run scripts/infer_pretrained_model.py "<prompt>" --max_tokens <tokens> --model_path <path_to_model> --device <cpu|cuda>
# Example:
uv run scripts/infer_pretrained_model.py "What is list comprehension?" --max_tokens 500 --model_path model/Qwen/Qwen2.5-Coder-1.5B-Instruct --device cpu
```

### 3. Fine-tune with 4bit-Quantization and LoRA Config
1. Finetuning BASEMODEL  
```bash
# Customize params in pybuddy/training.py (more options TBD)
uv run pybuddy/train.py --dataset_path data/data.json --model_path model/Qwen/Qwen2.5-Coder-1.5B-Instruct --output_dir model/ft_model
```
2. Finetune a finetuned model
```bash
uv run pybuddy/train.py --dataset_path data/python-code-instruction-dataset-kaggle-devastator.json --model_path model/Qwen/qwen2.5-1.5b-instruct-ft-merged --output_dir model/ft_model_04092025_0400
```

### 4. Inference with Fine-tuned Model
- **Single Prompt**:
  ```bash
  # Syntax:
  uv run python pybuddy/inference.py "<prompt>" --max_tokens <tokens> --base-model <path_to_base_model> --ftmodel <path_to_lora> --device <auto|cpu>
  # Example:
  uv run python pybuddy/inference.py "What is a loop?" --max_tokens 512 --base-model model/Qwen/Qwen2.5-Coder-1.5B-Instruct --ftmodel model/ft_model/lora-adapter --device auto
  ```

- **Chat Mode (Long Context)**:
  ```bash
  # Syntax:
  uv run python pybuddy/chat.py --base-model <path_to_base_model> --ftmodel <path_to_lora> --device <auto|cpu>
  # Example:
  uv run python pybuddy/chat.py --base-model model/Qwen/Qwen2.5-Coder-1.5B-Instruct --ftmodel model/ft_model/lora-adapter --device auto
  ```

### 5. Deploy on Android (Offline)
Convert the fine-tuned model to **GGUF** for offline inference on Android.

1. **Install llama.cpp**:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   cmake -B build
   cmake --build build --config Release -j 8
   ```

2. **Merge Base Model and LoRA**:
   ```python
   # uv run python
   from pybuddy import utils
   utils.merge_ft_model(
       base_model_path="model/Qwen/Qwen2.5-Coder-1.5B-Instruct",
       lora_adapter_path="model/ft_model/lora-adapter",
       output_path="model/Qwen/qwen2.5-1.5b-instruct-ft-merged"
   )
   ```

3. **Convert to GGUF (4-bit or 8-bit)**:
   ```bash
   # Syntax:
   uv run python llama.cpp/convert_hf_to_gguf.py <ft_model_dir> --outfile <gguf_models/filename.gguf> --outtype <q4_0|q8_0>
   # Example (8-bit, ~1.5GB):
   uv run python llama.cpp/convert_hf_to_gguf.py model/Qwen/qwen2.5-1.5b-instruct-ft-merged \
     --outfile model/gguf/qwen2.5-1.5b-q8-03092025.gguf \
     --outtype q8_0
   ```

4. **Test GGUF File on Computer**:
   ```bash
   # Syntax:
   llama.cpp/build/bin/llama-cli -m <model/gguf/filename.gguf> -p "<prompt>"
   # Example:
   llama.cpp/build/bin/llama-cli -m model/gguf/qwen2.5-1.5b-q8-03092025.gguf -p "What is list comprehension?"
   ```

5. **Run on Android**:
   - Copy the GGUF file (~800MB for `q4_0`, ~1.5GB for `q8_0`) to your Android device.
   - Install an App that can load local llm model such as **Llama Chat**.
   - Import the GGUF file for offline inference (~10-15 tokens/s on Snapdragon 685, 8GB RAM).

### 6. Share Models on Hugging Face
1. **Set Up Token**:
   - Create a Write token at [Hugging Face Settings](https://huggingface.co/settings/tokens).
   - Add to `.env` as `HF_TOKEN`.

2. **Upload Models**:
   ```bash
   # Syntax:
   uv run scripts/upload_to_hf.py --local_model_dir <path_to_dir> --hf_repo_id <username/repo_id> --repo_type model
   # Example:
   uv run scripts/upload_to_hf.py --local_model_dir model/gguf --hf_repo_id taufiq-ai/qwen2.5-coder-1.5-instruct-ft --repo_type model
   ```
   Alternatively use for simplicity
   ```bash
   export HF_TOKEN=<your_hf_token>
   huggingface-cli upload taufiq-ai/qwen2.5-coder-1.5-instruct-ft model/gguf --repo-type model
   ```

## Project Structure
- `pybuddy/`: Core scripts for training, inference, and utilities.
- `scripts/`: Tools for downloading/uploading models.
- `.env`: Environment variables (e.g., `HF_TOKEN`).

## Notes
- **Quantization**: Use `q4_0` (~800MB) or `q8_0` (~1.5GB) for Android. Avoid `tq2_0` due to quality issues.
- **Performance**: ~5-10 tokens/s on low-resource devices (Redmi Note 11 - 4GB RAM).
- **Compatibility**: Llama Chat works well with `q8_0` GGUF files.
- **Contributing**: PRs and issues welcome!.