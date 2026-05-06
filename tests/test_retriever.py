from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

import retriever


class KeywordEmbeddingModel:
    def __init__(self):
        self.vocab = ["rareterm", "semantic", "alpha", "beta"]

    def get_sentence_embedding_dimension(self) -> int:
        return len(self.vocab)

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        rows = []
        for text in texts:
            lowered = text.lower()
            if lowered.strip() == "rareterm":
                vector = np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")
            else:
                vector = np.array([float(term in lowered) for term in self.vocab], dtype="float32")
            if normalize_embeddings and np.linalg.norm(vector) > 0:
                vector = vector / np.linalg.norm(vector)
            rows.append(vector)
        return np.vstack(rows).astype("float32")


def _build_index(chunks: list[dict]):
    model = KeywordEmbeddingModel()
    vectors = model.encode([chunk["text"] for chunk in chunks])
    index = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())
    index.add(vectors)
    metadata = []
    for chunk in chunks:
        metadata.append(
            {
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "page": chunk["page"],
                "page_type": "text",
                "lang": "en",
            }
        )
    return model, index, metadata


def test_known_relevant_chunk_retrieves_in_top_three():
    chunks = [
        {"doc_id": "doc-a", "chunk_id": 0, "text": "alpha kinase pathway", "page": 1},
        {"doc_id": "doc-b", "chunk_id": 0, "text": "beta unrelated material", "page": 2},
        {"doc_id": "doc-c", "chunk_id": 0, "text": "semantic background", "page": 3},
    ]
    model, index, metadata = _build_index(chunks)

    result = retriever.retrieve(
        "alpha",
        index,
        metadata,
        Path("."),
        model,
        retrieval_mode="hybrid",
        candidate_k=3,
        final_k=3,
    )

    assert any(item["doc_id"] == "doc-a" for item in result.reranked[:3])


def test_hybrid_mode_differs_from_dense_only_when_bm25_matches():
    chunks = [
        {"doc_id": "dense-doc", "chunk_id": 0, "text": "semantic concept only", "page": 1},
        {"doc_id": "bm25-doc", "chunk_id": 0, "text": "rareterm exact lexical match", "page": 2},
        {"doc_id": "other-doc", "chunk_id": 0, "text": "beta background material", "page": 3},
    ]
    model, index, metadata = _build_index(chunks)
    bm25 = retriever.build_bm25_index(metadata, Path("."))

    dense = retriever.retrieve(
        "rareterm",
        index,
        metadata,
        Path("."),
        model,
        retrieval_mode="dense",
        candidate_k=3,
        final_k=3,
    )
    hybrid = retriever.retrieve(
        "rareterm",
        index,
        metadata,
        Path("."),
        model,
        retrieval_mode="hybrid",
        bm25_index=bm25,
        candidate_k=3,
        final_k=3,
    )

    assert dense.reranked[0]["doc_id"] != hybrid.reranked[0]["doc_id"]
    assert "bm25" in hybrid.reranked[0]["retrieval_sources"]
