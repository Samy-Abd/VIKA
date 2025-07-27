#!/usr/bin/env python
# generate.py – Step 8: run the LLM locally and emit an answer (with fallback model download)

import pathlib, sys, os
from llama_cpp import Llama

# Optional: install huggingface_hub if not present
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise SystemExit("❌ huggingface_hub not installed. Install with:\n  pip install huggingface_hub")

ROOT = pathlib.Path(__file__).parent
MODEL_DIR = ROOT / "models"
MODEL_NAME = "mistral-7b-instruct-v0.2.Q5_K_M.gguf"
MODEL_PATH = MODEL_DIR / MODEL_NAME

PROMPT_PATH = ROOT / "prompt.txt"
OUT_PATH = ROOT / "answer.txt"

# Fallback download if model is missing
if not MODEL_PATH.exists():
    print("🔽 Model not found locally. Downloading from Hugging Face…")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_path = hf_hub_download(
        repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename=MODEL_NAME,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False  # ensure a real file copy
    )
    print(f"✅ Model downloaded → {downloaded_path}")
    MODEL_PATH = pathlib.Path(downloaded_path)

PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

# 1. Spin up the model (7B in 4-bit = ~6GB)
llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=4096,
    n_threads=max(1, os.cpu_count() - 1),
    n_gpu_layers=32  
)

# 2. Run inference
print("▶ Generating …")
tokens = llm.create_completion(
    PROMPT,
    temperature=0.7,
    top_p=0.95,
    max_tokens=800,
    stop=["Source:", "Answer:"],
    stream=True,
)

with OUT_PATH.open("w", encoding="utf-8") as out_f:
    for chunk in tokens:
        txt = chunk["choices"][0]["text"]
        sys.stdout.write(txt)
        out_f.write(txt)

print(f"\n✅  answer.txt written → {OUT_PATH}")
