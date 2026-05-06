"""Dense, BM25, and hybrid retrieval for VIKA."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - declared in requirements
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - declared in requirements
    SentenceTransformer = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - requirements install rank_bm25
    BM25Okapi = None


MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
RRF_K = 60
RETRIEVAL_MODES = ("dense", "bm25", "hybrid")


@dataclass
class BM25Bundle:
    records: list[dict[str, Any]]
    tokenized_corpus: list[list[str]]
    model: Any


@dataclass
class RetrievalResult:
    candidates: list[dict[str, Any]]
    reranked: list[dict[str, Any]]
    query_vector: np.ndarray
    retrieval_mode: str
    reranked_candidates: list[dict[str, Any]]


class SimpleBM25:
    """Small fallback used only when rank_bm25 is unavailable locally."""

    def __init__(self, corpus: list[list[str]]):
        self.corpus = corpus
        self.doc_count = max(len(corpus), 1)
        self.avgdl = sum(len(doc) for doc in corpus) / self.doc_count
        self.df: dict[str, int] = {}
        for doc in corpus:
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = []
        k1 = 1.5
        b = 0.75
        for doc in self.corpus:
            freqs: dict[str, int] = {}
            for token in doc:
                freqs[token] = freqs.get(token, 0) + 1
            score = 0.0
            doc_len = max(len(doc), 1)
            for token in query_tokens:
                if token not in freqs:
                    continue
                df = self.df.get(token, 0)
                idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
                tf = freqs[token]
                denom = tf + k1 * (1 - b + b * doc_len / max(self.avgdl, 1e-9))
                score += idf * (tf * (k1 + 1)) / denom
            scores.append(score)
        return np.asarray(scores, dtype="float32")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\-]+", (text or "").lower(), flags=re.UNICODE)


def load_faiss_index(index_dir: Path, allow_empty: bool = False) -> tuple[faiss.Index, list[dict[str, Any]]]:
    index_path = index_dir / "faiss.index"
    meta_path = index_dir / "metadata.jsonl"

    if not index_path.exists() or not meta_path.exists():
        if allow_empty:
            return faiss.IndexFlatIP(EMBED_DIM), []
        raise FileNotFoundError(f"No FAISS index found under {index_dir}")

    index = faiss.read_index(str(index_path))
    metadata = [
        json.loads(line)
        for line in meta_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if index.ntotal != len(metadata):
        raise ValueError("Vector count mismatch between FAISS index and metadata.jsonl")
    return index, metadata


def embed_queries(model: Any, queries: list[str]) -> np.ndarray:
    embeddings = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype="float32")


def load_chunk_record(chunks_dir: Path, doc_id: str, chunk_id: int) -> dict[str, Any] | None:
    jsonl_path = chunks_dir / f"{doc_id}.chunks.jsonl"
    if not jsonl_path.exists():
        return None
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("id") == chunk_id or record.get("chunk_id") == chunk_id:
                return record
    return None


def load_chunk_text(chunks_dir: Path, doc_id: str, chunk_id: int) -> str | None:
    record = load_chunk_record(chunks_dir, doc_id, chunk_id)
    return record.get("text") if record else None


def _record_with_text(record: dict[str, Any], chunks_dir: Path, index_position: int) -> dict[str, Any] | None:
    item = dict(record)
    item["index_position"] = index_position
    if "chunk_id" not in item:
        item["chunk_id"] = item.get("id")
    if not item.get("text"):
        chunk = load_chunk_record(chunks_dir, item["doc_id"], int(item["chunk_id"]))
        if chunk:
            item.update({key: value for key, value in chunk.items() if key not in {"id"}})
            item["text"] = chunk.get("text", "")
    if not item.get("text"):
        return None
    if item.get("page") is None:
        item["page"] = 1
    item["page_type"] = item.get("page_type") or "text"
    item["lang"] = item.get("lang") or "en"
    return item


def build_bm25_index(metadata: list[dict[str, Any]], chunks_dir: Path) -> BM25Bundle:
    records: list[dict[str, Any]] = []
    corpus: list[list[str]] = []

    for position, record in enumerate(metadata):
        item = _record_with_text(record, chunks_dir, position)
        if item is None:
            continue
        records.append(item)
        corpus.append(tokenize(item["text"]))

    if not corpus:
        model = SimpleBM25([])
    else:
        model = BM25Okapi(corpus) if BM25Okapi is not None else SimpleBM25(corpus)
    return BM25Bundle(records=records, tokenized_corpus=corpus, model=model)


def search_dense(
    index: faiss.Index,
    metadata: list[dict[str, Any]],
    chunks_dir: Path,
    query_vector: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    if index.ntotal == 0 or not metadata:
        return []
    k = min(top_k, index.ntotal)
    scores, indexes = index.search(query_vector, k)
    results: list[dict[str, Any]] = []
    for rank, (score, position) in enumerate(zip(scores[0], indexes[0]), start=1):
        if position < 0 or position >= len(metadata):
            continue
        item = _record_with_text(metadata[int(position)], chunks_dir, int(position))
        if item is None:
            continue
        item["dense_score"] = float(score)
        item["dense_rank"] = rank
        item["retrieval_sources"] = ["dense"]
        item["score"] = float(score)
        results.append(item)
    return results


def search_bm25(bundle: BM25Bundle, query: str, top_k: int) -> list[dict[str, Any]]:
    if not bundle.records:
        return []
    scores = np.asarray(bundle.model.get_scores(tokenize(query)), dtype="float32")
    order = np.argsort(scores)[::-1]
    results: list[dict[str, Any]] = []
    for rank, position in enumerate(order[:top_k], start=1):
        score = float(scores[position])
        if score <= 0:
            continue
        item = dict(bundle.records[int(position)])
        item["bm25_score"] = score
        item["bm25_rank"] = rank
        item["retrieval_sources"] = ["bm25"]
        item["score"] = score
        results.append(item)
    return results


def _candidate_key(item: dict[str, Any]) -> tuple[str, int]:
    return str(item["doc_id"]), int(item.get("chunk_id", item.get("id", 0)))


def rrf_fuse(
    dense_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    fused: dict[tuple[str, int], dict[str, Any]] = {}

    for rank, item in enumerate(dense_hits, start=1):
        key = _candidate_key(item)
        fused[key] = dict(item)
        fused[key]["retrieval_sources"] = {"dense"}
        fused[key]["rrf_score"] = 1.0 / (rrf_k + rank)
        fused[key]["dense_rank"] = rank

    for rank, item in enumerate(bm25_hits, start=1):
        key = _candidate_key(item)
        if key not in fused:
            fused[key] = dict(item)
            fused[key]["retrieval_sources"] = set()
            fused[key]["rrf_score"] = 0.0
        fused[key]["retrieval_sources"].add("bm25")
        fused[key]["bm25_rank"] = rank
        fused[key]["bm25_score"] = item.get("bm25_score", 0.0)
        fused[key]["rrf_score"] += 1.0 / (rrf_k + rank)

    results = []
    for item in fused.values():
        item["retrieval_sources"] = sorted(item["retrieval_sources"])
        item["score"] = float(item["rrf_score"])
        results.append(item)
    return sorted(results, key=lambda item: item["score"], reverse=True)


def retrieve(
    query: str,
    index: faiss.Index,
    metadata: list[dict[str, Any]],
    chunks_dir: Path,
    embed_model: Any,
    reranker: Any | None = None,
    retrieval_mode: str = "hybrid",
    bm25_index: BM25Bundle | None = None,
    candidate_k: int = 20,
    final_k: int = 5,
) -> RetrievalResult:
    if retrieval_mode not in RETRIEVAL_MODES:
        raise ValueError(f"retrieval_mode must be one of {RETRIEVAL_MODES}")

    query_vector = embed_queries(embed_model, [query])
    dense_hits: list[dict[str, Any]] = []
    bm25_hits: list[dict[str, Any]] = []

    if retrieval_mode in {"dense", "hybrid"}:
        dense_hits = search_dense(index, metadata, chunks_dir, query_vector, candidate_k)
    if retrieval_mode in {"bm25", "hybrid"}:
        bm25_index = bm25_index or build_bm25_index(metadata, chunks_dir)
        bm25_hits = search_bm25(bm25_index, query, candidate_k)

    if retrieval_mode == "dense":
        candidates = dense_hits
    elif retrieval_mode == "bm25":
        candidates = bm25_hits
    else:
        candidates = rrf_fuse(dense_hits, bm25_hits)

    candidates = candidates[:candidate_k]
    if reranker is not None and candidates:
        reranked_candidates = reranker.rerank(query, [dict(item) for item in candidates])
    else:
        reranked_candidates = [
            dict(item, rerank_score=float(item.get("score", 0.0)))
            for item in candidates
        ]

    return RetrievalResult(
        candidates=candidates,
        reranked=reranked_candidates[:final_k],
        query_vector=query_vector,
        retrieval_mode=retrieval_mode,
        reranked_candidates=reranked_candidates,
    )


def mean_cosine_similarity(
    index: faiss.Index,
    query_vector: np.ndarray,
    items: list[dict[str, Any]],
) -> float:
    if not items:
        return 0.0
    values: list[float] = []
    for item in items:
        position = item.get("index_position")
        if position is None:
            continue
        try:
            vector = np.asarray(index.reconstruct(int(position)), dtype="float32")
            values.append(float(np.dot(query_vector[0], vector)))
        except Exception:
            continue
    return float(np.mean(values)) if values else 0.0


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search a VIKA FAISS/BM25 index.")
    parser.add_argument("--query", "-q", help="Text query.")
    parser.add_argument("--k", type=int, default=5, help="Number of final results.")
    parser.add_argument("--candidate_k", type=int, default=20, help="Candidate pool size.")
    parser.add_argument("--mode", choices=RETRIEVAL_MODES, default="hybrid")
    parser.add_argument("--index_dir", type=Path, default=Path("./data/index"))
    parser.add_argument("--chunks_dir", type=Path, default=Path("./data/pdfs"))
    return parser.parse_args()


def main() -> None:
    args = _build_args()
    if not args.query:
        sys.exit("--query is required")

    if SentenceTransformer is None:
        raise SystemExit("sentence-transformers is required for the retriever CLI.")
    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    index, metadata = load_faiss_index(args.index_dir, allow_empty=False)
    result = retrieve(
        args.query,
        index,
        metadata,
        args.chunks_dir,
        model,
        retrieval_mode=args.mode,
        candidate_k=args.candidate_k,
        final_k=args.k,
    )
    json.dump(result.reranked, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
