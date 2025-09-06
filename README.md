# SLM-Python-Buddy

An offline Python coding assistant that runs on your phone. Fine-tune small language models (like Qwen2.5-Coder-1.5B) and deploy them as lightweight GGUF files for mobile devices.

## What it does

- **Fine-tune** small language models on Python instruction datasets using LoRA  
- **Convert & Quantize** models to GGUF format (Q8: ~1.5GB) for mobile deployment  
- **Deploy** offline on Android devices for Python learning and coding help  
- **Answer** questions like "What is a loop?" or "Explain list comprehension" with ~10-15 tokens/s on mid-range phones  

Perfect for students and developers who want AI coding assistance on their phone without internet dependency or privacy concerns.  

**Finetuned GGUF models are available here:**
- **HuggingFace:** https://huggingface.co/taufiq-ai/qwen2.5-coder-1.5-instruct-ft *(preferred)*  
- **Ollama:** https://ollama.com/taufiq-ai/qwen2.5-coder-1.5b-instruct-ft-taufiq-04092025  

## Demo

Runs offline on mid-range phones. Tested on the following devices:  
- **Redmi Note 13 (Snapdragon 685, 8GB RAM):** ~5-10 tokens/s  
- **Redmi Note 11 (Snapdragon 680, 4GB RAM):** ~5–7 tokens/s  

🎥 **[Click here to watch the demo on youtube](https://www.youtube.com/watch?v=GkIkqUldQak)**



## Quick Start  

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

### 1. Download Pre-trained Model from HuggingFace
```sh
# syntax
uv run python -m scripts.download_pretrained_model --model-name <hf_model_name> --model-dir <dir> --device <auto|cpu|cuda>
```
```bash
# example:
uv run python -m scripts.download_pretrained_model --model-name Qwen/Qwen2.5-Coder-1.5B-Instruct --model-dir model --device auto
```

### 2. Infer Pre-trained Model
```sh
# syntax
uv run -m scripts.infer_pretrained_model "<prompt>" --max_tokens <tokens> --model_path <path_to_model> --device <auto|cpu|cuda>
```
```bash
# example
uv run -m scripts.infer_pretrained_model "What is list comprehension?" --max_tokens 500 --model_path model/Qwen/Qwen2.5-Coder-1.5B-Instruct --device auto
```

### 3. Fine-tune with 4bit-Quantization and LoRA Config
**1. Finetuning Pretrained Model**  
```bash
# syntax
uv run -m pybuddy.train \
  --dataset_path <path/to/dataset.json> \
  --model_path <path/to/base/model> \
  --output_dir <path/to/output> \
  --batch_size <int> \
  --grad_accum <int> \
  --lr <float> \
  --epochs <int> \
  --max_length <int>
```
```bash
# example
uv run -m pybuddy.train \
  --dataset_path data/data.json \
  --model_path model/Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --output_dir model/ft_model \
  --batch_size 2 \
  --grad_accum 4 \
  --lr 2e-4 \
  --epochs 3 \
  --max_length 4096
```
**2. Finetuning Finetuned Model**  
```bash
# Syntax
uv run -m pybuddy.train \
 --dataset_path <path/to/dataset.json> \
 --model_path <path/to/merged/model> \
 --output_dir <path/to/output> \
 --batch_size <int> \
 --grad_accum <int> \
 --lr <float> \
 --epochs <int> \
 --max_length <int>
```
```bash
# example
uv run -m pybuddy.train \
 --dataset_path data/python-code-inst-dataset.json \
 --model_path model/Qwen/qwen2.5-1.5b-instruct-ft-merged \
 --output_dir model/ft_model_04092025_0400 \
 --batch_size 2 \
 --grad_accum 4 \
 --lr 2e-4 \
 --epochs 3 \
 --max_length 4096
```

### 4. Inference Fine-tuned Model
**1. Single Prompt**:  
```sh
# syntax
uv run python -m pybuddy.inference "<prompt>" --max_tokens <max_tokens> --base-model <path/to/base_model> --ftmodel <path_to_finetuned_lora_adapter> --device <auto|cpu|cuda>
```
```bash
# example
uv run python -m pybuddy.inference \
 "What is a loop?" \
 --max_tokens 512 \
 --base-model model/Qwen/Qwen2.5-Coder-1.5B-Instruct \
 --ftmodel model/ft_model_04092025_0400/lora-adapter \
 --device auto
```

**2. Chat Mode (Long Context)**:  
```sh
# syntax
uv run python -m pybuddy.chat --base-model <path/to/base_model> --ftmodel <path/to/lora_adapter> --device <auto|cpu|cuda>
```
```bash
# example
uv run python -m pybuddy.chat \
  --base-model model/Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --ftmodel model/ft_model_04092025_0400/lora-adapter \
  --device auto
```

### 5. Run on Android (Offline)
Convert your fine-tuned model to GGUF format for offline Android inference.

**Step 1: Setup llama.cpp**  
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j 8
```

**Step 2: Merge base model with LoRA adapter**  
```py
# uv run python
from pybuddy import utils
utils.merge_ft_model(
    base_model_path="model/Qwen/Qwen2.5-Coder-1.5B-Instruct",
    lora_adapter_path="model/ft_model/lora-adapter", 
    output_path="model/Qwen/qwen2.5-1.5b-instruct-ft-merged"
)
```

**Step 3: Convert to GGUF**  
```bash
# 8-bit (~1.5GB, better quality)
uv run python llama.cpp/convert_hf_to_gguf.py \
  model/Qwen/qwen2.5-1.5b-instruct-ft-merged \
  --outfile model/gguf/qwen2.5-1.5b-q8.gguf \
  --outtype q8_0
```

**Step 4: Test locally**  
```sh
llama.cpp/build/bin/llama-cli \
  -m model/gguf/qwen2.5-1.5b-q8.gguf \
  -p "What is list comprehension?"
```

**Step 5: Deploy to Android**  
- Copy **GGUF** file to your Android device  
- Install **Llama Chat** or similar local LLM app  
- Import the GGUF file  
- Enjoy offline inference (~10-15 tokens/s on mid-range devices)  


### 6. Share Models on Hugging Face

**Step 1: Setup HF Token**  
- Create a Write token at [Hugging Face Settings](https://huggingface.co/settings/tokens)  
- Add `HF_TOKEN=your_token_here` to your `.env` file  

**Step 2: Upload Models**

*Option A: Using custom script*  
```bash
uv run -m scripts.upload_to_hf \
 --local_model_dir model/gguf \
 --hf_repo_id username/repo-name \
 --repo_type model
```

*Option B: Using HF CLI (simpler)*
```sh
export HF_TOKEN=your_hf_token
huggingface-cli upload username/repo-name model/gguf --repo-type model
```

## Acknowledgments

Special thanks to **[LalokaLabs](https://lalokalabs.co/)** for providing GPU compute resources that made this project possible. The fine-tuning was conducted on their **Quadro RTX 8000 (46GB VRAM)** infrastructure.

Without access to high-end GPU resources, training and experimenting with language models would be significantly more challenging for individual developers and researchers.
