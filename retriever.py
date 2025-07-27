#!/usr/bin/env python
"""
retriever.py – Lightweight semantic search over a FAISS index (now with *optional* empty‑index handling).
------------------------------------------------------------------------------------------------------
This rewrite keeps 100 % CLI compatibility **but** returns Python exceptions (instead of
`SystemExit`) from library helpers so GUI apps can import the module safely when an index
is not yet built.

Key changes
~~~~~~~~~~~
* `load_faiss_index(allow_empty: bool = False)` – if `allow_empty=True` and the
  index is missing, an *empty* `IndexFlatIP` plus empty `metadata` list is
  returned rather than aborting.
* Extracted `print_result()` to DRY the CLI pretty‑print.
* Added type annotations + docstrings cleanup.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import faiss  # type: ignore
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "all-MiniLM-L6-v2"  # Must match embed_faiss.py
EMBED_DIM = 384  # Dimension of all‑MiniLM‑L6‑v2

###############################################################################
# Helpers
###############################################################################

def load_faiss_index(index_dir: Path, allow_empty: bool = False):
    """Return `(index, metadata)`.

    If *allow_empty* is **True** the function yields an *empty* IndexFlatIP & list
    when the files don’t exist, letting calling code continue gracefully.
    """
    index_path = index_dir / "faiss.index"
    meta_path = index_dir / "metadata.jsonl"

    if not index_path.exists() or not meta_path.exists():
        if allow_empty:
            return faiss.IndexFlatIP(EMBED_DIM), []
        raise FileNotFoundError(f"No index found under {index_dir}")

    index = faiss.read_index(str(index_path))

    metadata: List[Dict] = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines() if line]

    if index.ntotal != len(metadata):
        raise ValueError("Vector count mismatch between FAISS index and metadata.jsonl")

    return index, metadata


def embed_queries(model: SentenceTransformer, queries: List[str]) -> np.ndarray:
    emb = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
    return emb.astype("float32")


def search(index: faiss.Index,
           metadata: List[Dict],
           query_vec: np.ndarray,
           top_k: int):
    # 1) Rien à chercher si l’index est vide
    if index.ntotal == 0 or not metadata:
        return []

    scores, idxs = index.search(query_vec, top_k)

    results = []
    for s, i in zip(scores[0], idxs[0]):
        # FAISS renvoie -1 quand il n’y a pas de voisin
        if i < 0 or i >= len(metadata):
            continue
        results.append((float(s), metadata[i]))
    return results



def load_chunk_text(chunks_dir: Path, doc_id: str, chunk_id: int) -> str | None:
    jsonl_path = chunks_dir / f"{doc_id}.chunks.jsonl"
    if not jsonl_path.exists():
        return None
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        if obj.get("id") == chunk_id:
            return obj.get("text")
    return None

###############################################################################
# CLI helpers
###############################################################################

def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Semantic search over a FAISS index.")
    p.add_argument("--query", "-q", help="Text of the query (omit in --interactive mode)")
    p.add_argument("--k", type=int, default=5, help="Number of neighbors to return (default: 5)")
    p.add_argument("--index_dir", type=Path, default=Path("./data/index"))
    p.add_argument("--chunks_dir", type=Path, default=Path("./data/pdfs"))
    p.add_argument("--interactive", action="store_true", help="Run REPL if set (ignores --query)")
    return p.parse_args()


def _pretty_print(results: List[Tuple[float, Dict]], chunks_dir: Path):
    for rank, (score, meta) in enumerate(results, 1):
        doc_id = meta["doc_id"]
        chunk_id = meta["chunk_id"]
        page = meta.get("page")
        snippet = load_chunk_text(chunks_dir, doc_id, chunk_id)
        print("#" * 72)
        print(f"Rank {rank} – score: {score:.4f}")
        print(f"Document: {doc_id}   Chunk: {chunk_id}   Page: {page}")
        if snippet:
            print("\n" + snippet[:300].replace("\n", " ") + ("…" if len(snippet) > 300 else ""))

###############################################################################
# Main
###############################################################################

def main():  # pragma: no cover
    args = _build_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)

    try:
        index, metadata = load_faiss_index(args.index_dir, allow_empty=False)
    except FileNotFoundError as e:
        sys.exit(str(e))

    if args.interactive:
        print("▶️  Interactive mode – type a query and hit Enter (Ctrl‑C to exit)")
        try:
            while True:
                q = input("🔍 ")
                if not q.strip():
                    continue
                q_vec = embed_queries(model, [q])
                hits = search(index, metadata, q_vec, args.k)
                _pretty_print(hits, args.chunks_dir)
        except KeyboardInterrupt:
            print("\nBye! 👋")
        return

    if not args.query:
        sys.exit("❌ --query is required unless --interactive is used")

    q_vec = embed_queries(model, [args.query])
    hits = search(index, metadata, q_vec, args.k)

    # Write compact JSON to STDOUT (pipeline‑friendly)
    out = []
    for score, meta in hits:
        obj = {"score": score, **meta}
        snippet = load_chunk_text(args.chunks_dir, meta["doc_id"], meta["chunk_id"])
        if snippet:
            obj["text"] = snippet
        out.append(obj)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
