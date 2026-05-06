"""CrossEncoder reranking for retrieved passages."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - declared in requirements
    torch = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover - declared in requirements
    CrossEncoder = None


DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Small wrapper around sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32):
        if CrossEncoder is None:
            raise RuntimeError("sentence-transformers is required for CrossEncoder reranking.")
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.asarray([], dtype="float32")
        pairs = [[query, passage] for passage in passages]
        if torch is None:
            scores = self.model.predict(pairs, batch_size=self.batch_size, convert_to_numpy=True)
        else:
            with torch.inference_mode():
                scores = self.model.predict(pairs, batch_size=self.batch_size, convert_to_numpy=True)
        return np.asarray(scores, dtype="float32")

    def rerank(self, query: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        passages = [item.get("text", "") for item in items]
        scores = self.score(query, passages)
        reranked: list[dict[str, Any]] = []
        for item, score in zip(items, scores):
            enriched = dict(item)
            enriched["rerank_score"] = float(score)
            reranked.append(enriched)
        return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank retrieval hits with a CrossEncoder.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--input", type=Path, default=None, help="JSON hits file; reads stdin if omitted.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    return parser.parse_args()


def read_items(path: Path | None) -> list[dict[str, Any]]:
    data = sys.stdin.read() if path is None else path.read_text(encoding="utf-8")
    parsed = json.loads(data)
    if not isinstance(parsed, list):
        raise SystemExit("Input JSON must be a list of retrieval hit objects.")
    return parsed


def main() -> None:
    args = build_args()
    items = read_items(args.input)
    reranker = CrossEncoderReranker(model_name=args.model, batch_size=args.batch)
    results = reranker.rerank(args.query, items)
    top_n = args.top if args.top > 0 else len(results)
    json.dump(results[:top_n], sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
