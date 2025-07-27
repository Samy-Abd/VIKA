#!/usr/bin/env python
"""
reranker.py – Refine initial semantic‑search results with a local cross‑encoder.

Pipeline position
-----------------
This script is intended to follow **retriever.py**.  Feed it the JSON list
printed by *retriever.py* (either via a file or STDIN) and it will score each
(result‑query) pair with a cross‑encoder such as
`cross‑encoder/ms‑marco‑MiniLM‑L‑6‑v2`, returning the passages sorted by the
more accurate relevance estimate.

Quick start
-----------
# Retrieve top‑20 candidate chunks, then re‑rank and print the best 5
python retriever.py --query "hydrogen storage materials" --k 20 \
  | python reranker.py --query "hydrogen storage materials" --top 5

# Or operate on a saved file:
python retriever.py --query "EU AI Act timeline" --k 15 > hits.json
python reranker.py --query "EU AI Act timeline" --input hits.json --top 10

Dependencies
------------
> pip install sentence‑transformers tqdm numpy

The chosen model (~65 MB) runs comfortably on CPU; if a GPU is available the
script will use it automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

###############################################################################
# Core class
###############################################################################

class CrossEncoderReranker:
    """Lightweight wrapper around a sentence‑transformers CrossEncoder."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size

    @torch.inference_mode()
    def score(self, query: str, passages: List[str]) -> np.ndarray:
        """Return relevance scores (higher = more relevant)."""
        pairs = [[query, p] for p in passages]
        scores = self.model.predict(pairs, batch_size=self.batch_size, convert_to_numpy=True)
        return scores.astype(float)

    def rerank(self, query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach a *rerank_score* to each item and return sorted list (desc)."""
        passages = [item.get("text", "") for item in items]
        scores = self.score(query, passages)
        for item, s in zip(items, scores):
            item["rerank_score"] = float(s)
        return sorted(items, key=lambda x: x["rerank_score"], reverse=True)

###############################################################################
# CLI helpers
###############################################################################

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re‑rank chunks using a cross‑encoder.")
    p.add_argument("--query", required=True, help="The search query string.")
    p.add_argument("--input", type=Path, default=None, help="JSON file with retrieval hits (reads STDIN if omitted).")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Cross‑encoder name (default: {DEFAULT_MODEL}).")
    p.add_argument("--top", type=int, default=5, help="Number of top passages to output (default: 5). Use 0 to output all.")
    p.add_argument("--batch", type=int, default=32, help="Batch size for prediction (default: 32).")
    p.add_argument("--no_snippet", action="store_true", help="Drop 'text' field to save space in output JSON.")
    return p.parse_args()


def read_items(path: Path | None) -> List[Dict[str, Any]]:
    data = sys.stdin.read() if path is None else path.read_text(encoding="utf-8")
    try:
        items = json.loads(data)
    except json.JSONDecodeError as e:
        raise SystemExit(f"❌ Failed to parse JSON input: {e}") from e
    if not isinstance(items, list):
        raise SystemExit("❌ Input JSON must be a list of objects (retriever hits).")
    return items


def main() -> None:
    args = build_args()

    items = read_items(args.input)
    if not items:
        raise SystemExit("❌ No passages found in the input – nothing to rerank.")

    reranker = CrossEncoderReranker(model_name=args.model, batch_size=args.batch)
    results = reranker.rerank(args.query, items)

    top_n = args.top if args.top > 0 else len(results)
    out = results[:top_n]

    if args.no_snippet:
        for obj in out:
            obj.pop("text", None)

    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":  # pragma: no cover
    main()