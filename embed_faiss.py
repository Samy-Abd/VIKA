#!/usr/bin/env python
"""
embed_faiss.py – Embed chunk files into a FAISS index **incrementally**.
-----------------------------------------------------------------------
This rewrite keeps the original CLI behaviour (build an index from scratch) *and*
adds a public helper `add_chunks_to_index()` that lets other Python code append
new documents without rebuilding everything – essential for live RAG services.

Highlights
~~~~~~~~~~
* **Incremental updates**: detect already‑indexed `doc_id`s and skip them.
* **Re‑usable helpers**: `encode_texts()` + `load_or_create_index()` so external
  apps (e.g. `app.py`) can piggy‑back on the same logic.
* **Safer metadata**: always flush metadata *after* FAISS writes succeed.
* **Configurable**: environment variables `EMBED_MODEL` and `BATCH_SIZE` override
  defaults at runtime.

CLI
---
```
# (re)build full index
python embed_faiss.py ./data/pdfs --out ./data/index --full

# append new *.chunks.jsonl files into existing index
python embed_faiss.py ./incoming/mydoc.chunks.jsonl --out ./data/index
```
If `--full` is omitted the script operates **incrementally**.

Dependencies
~~~~~~~~~~~~
    pip install sentence-transformers faiss-cpu tqdm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import faiss  # type: ignore
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

###############################################################################
# Configuration
###############################################################################

DEFAULT_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "256"))

###############################################################################
# Low‑level helpers
###############################################################################

def encode_texts(model: SentenceTransformer, texts: Sequence[str]) -> np.ndarray:
    """Return *L2‑normalised* embeddings as float32 ndarray (shape: n×d)."""
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return emb.astype("float32")


def load_or_create_index(index_path: Path, dim: int) -> faiss.Index:
    """Load existing IndexFlatIP or create an empty one with dimension *dim*."""
    if index_path.exists():
        return faiss.read_index(str(index_path))
    return faiss.IndexFlatIP(dim)

###############################################################################
# Incremental public API
###############################################################################

def add_chunks_to_index(
    chunk_paths: List[Path],
    index_dir: Path,
    model: SentenceTransformer | None = None,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Embed each *.chunks.jsonl* in *chunk_paths* **once** and append to index.

    Parameters
    ----------
    chunk_paths : list[Path]
        Files produced by *chunker.py*.
    index_dir : Path
        Directory containing `faiss.index` + `metadata.jsonl` (created if missing).
    model : SentenceTransformer or None, optional
        Pre‑loaded embedder (will be loaded automatically if *None*).
    batch_size : int, optional
        Encode texts in batches of this size.

    Returns
    -------
    int
        Number of new vectors added (0 ➜ nothing changed).
    """
    if not chunk_paths:
        return 0

    # Load model lazily ------------------------------------------------------
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(DEFAULT_MODEL, device=device)
    dim = model.get_sentence_embedding_dimension()

    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "faiss.index"
    meta_path = index_dir / "metadata.jsonl"

    index = load_or_create_index(index_path, dim)

    # Read existing metadata to skip duplicates -----------------------------
    existing_doc_ids: set[str] = set()
    metadata: List[Dict] = []
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                metadata.append(obj)
                existing_doc_ids.add(obj["doc_id"])

    # ---- iterate over new docs -------------------------------------------
    texts_batch: List[str] = []
    meta_batch: List[Dict] = []
    new_vectors = 0

    def _flush():
        nonlocal new_vectors
        if not texts_batch:
            return
        emb = encode_texts(model, texts_batch)
        index.add(emb)
        with meta_path.open("a", encoding="utf-8") as f:
            for rec in meta_batch:
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")
        new_vectors += emb.shape[0]
        texts_batch.clear()
        meta_batch.clear()

    for p in chunk_paths:
        doc_id = p.stem.replace(".chunks", "")
        if doc_id in existing_doc_ids:
            continue  # already present → skip entire file
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                texts_batch.append(obj["text"].strip())
                meta_batch.append({
                    "doc_id": doc_id,
                    "chunk_id": obj.get("id"),
                    "page": obj.get("page"),
                })
                if len(texts_batch) >= batch_size:
                    _flush()
        _flush()

    if new_vectors:
        faiss.write_index(index, str(index_path))
    return new_vectors

###############################################################################
# Full rebuild – retain original behaviour
###############################################################################

def build_index_full(input_targets: List[Path], out_dir: Path, batch_size: int = BATCH_SIZE):
    if not input_targets:
        raise SystemExit("❌ No .chunks.jsonl files found to index.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(DEFAULT_MODEL, device=device)
    dim = model.get_sentence_embedding_dimension()

    texts: List[str] = []
    meta_records: List[Dict] = []

    for doc_id, chunk in tqdm(_iter_all_chunks(input_targets), desc="Embedding"):
        texts.append(chunk["text"].strip())
        meta_records.append({
            "doc_id": doc_id,
            "chunk_id": chunk.get("id"),
            "page": chunk.get("page"),
        })

    embeddings = encode_texts(model, texts)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for rec in meta_records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ Built index with {index.ntotal} vectors → {out_dir}")

###############################################################################
# Shared iter helper
###############################################################################

def _load_chunks(jsonl_path: Path) -> List[Dict]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _iter_all_chunks(targets: List[Path]):
    for p in targets:
        doc_id = p.stem.replace(".chunks", "")
        for chunk in _load_chunks(p):
            yield doc_id, chunk

###############################################################################
# CLI
###############################################################################

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Embed chunk files into a FAISS index (incremental).")
    p.add_argument("input", type=Path, help="A .chunks.jsonl file or directory containing them.")
    p.add_argument("--out", type=Path, default=Path("./data/index"), help="Index folder (default: ./data/index)")
    p.add_argument("--full", action="store_true", help="Rebuild index from scratch instead of appending.")
    p.add_argument("--batch", type=int, default=BATCH_SIZE, help="Embedding batch size (default: 256)")
    return p


def _gather_targets(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.chunks.jsonl"))


def main(argv: List[str] | None = None):  # pragma: no cover
    args = _build_arg_parser().parse_args(argv)
    targets = _gather_targets(args.input)

    if args.full:
        build_index_full(targets, args.out, args.batch)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(DEFAULT_MODEL, device=device)
        added = add_chunks_to_index(targets, args.out, model, args.batch)
        print(f"✅ Added {added} new vector(s) to index")


if __name__ == "__main__":
    main()
