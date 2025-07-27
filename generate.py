#!/usr/bin/env python
# generate.py – Step 8: run the LLM locally and emit an answer

import pathlib, sys, textwrap
from llama_cpp import Llama
import os

ROOT = pathlib.Path(__file__).parent
MODEL = ROOT / "models" / "mistral-7b-instruct-v0.2.Q5_K_M.gguf"
PROMPT_PATH = ROOT / "prompt.txt"
OUT_PATH = ROOT / "answer.txt"

PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

# 1. Spin up the model (7-B params in 4-bit ≈ 6.2 GB)
llm = Llama(
    model_path=str(MODEL),
    n_ctx=4096,            # Mistral can use long contexts
    n_threads=max(1, os.cpu_count() - 2),  # leave one core free
    n_gpu_layers=-1        # keep everything on CPU
)

# 2. Run the inference 💡
print("▶ Generating …")
tokens = llm.create_completion(
    PROMPT,
    temperature=0.7,
    top_p=0.95,
    max_tokens=800,
    stop=["Source:", "Answer:"],  # mirrors the prompt format
    stream=True,                 # yields chunks immediately
)

with OUT_PATH.open("w", encoding="utf-8") as out_f:
    for chunk in tokens:
        txt = chunk["choices"][0]["text"]
        sys.stdout.write(txt)    # live console stream
        out_f.write(txt)

print(f"\n✅  answer.txt written → {OUT_PATH}")