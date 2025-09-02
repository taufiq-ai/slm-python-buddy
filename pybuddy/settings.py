from dotenv import load_dotenv
import os

# Load .env file with variable interpolation
load_dotenv(interpolate=True)

MODEL_DIR = os.getenv("MODEL_DIR")
QMODEL_DIR = os.getenv("QMODEL_DIR")
FTMODEL_DIR = os.getenv("FTMODEL_DIR")
BASEMODEL= os.getenv("BASEMODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
DEVICE=os.getenv("DEVICE", "auto")


