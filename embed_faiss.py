"""FAISS embedding pipeline with language-aware model routing."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import faiss
import numpy as np
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - declared in requirements
    def tqdm(iterable, **_: Any):  # type: ignore[no-redef]
        return iterable

try:
    import torch
except ImportError:  # pragma: no cover - declared in requirements
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - declared in requirements
    SentenceTransformer = None


EN_EMBED_MODEL = os.getenv("VIKA_EMBED_MODEL_EN", "all-MiniLM-L6-v2")
MULTILINGUAL_EMBED_MODEL = os.getenv(
    "VIKA_EMBED_MODEL_MULTI",
    "paraphrase-multilingual-MiniLM-L12-v2",
)
BATCH_SIZE = int(os.getenv("VIKA_EMBED_BATCH_SIZE", "256"))


def embedding_key_for_lang(lang: str | None) -> str:
    return "en" if (lang or "en").lower() == "en" else "multi"


def embedding_model_name_for_lang(lang: str | None) -> str:
    return EN_EMBED_MODEL if embedding_key_for_lang(lang) == "en" else MULTILINGUAL_EMBED_MODEL


def load_embedding_models(device: str | None = None) -> dict[str, Any]:
    """Load both embedders once and keep them available for routing."""
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is required to load embedding models.")
    device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    return {
        "en": SentenceTransformer(EN_EMBED_MODEL, device=device),
        "multi": SentenceTransformer(MULTILINGUAL_EMBED_MODEL, device=device),
    }


def select_embedding_model(
    models: dict[str, SentenceTransformer],
    lang: str | None,
) -> SentenceTransformer:
    return models[embedding_key_for_lang(lang)]


def encode_texts(model: Any, texts: Sequence[str]) -> np.ndarray:
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype="float32")


def load_or_create_index(index_path: Path, dim: int) -> faiss.Index:
    if index_path.exists():
        return faiss.read_index(str(index_path))
    return faiss.IndexFlatIP(dim)


def _load_chunks(jsonl_path: Path) -> list[dict[str, Any]]:
    with jsonl_path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _normalise_chunk(doc_id: str, chunk: dict[str, Any]) -> dict[str, Any]:
    page = chunk.get("page")
    return {
        "id": int(chunk.get("id", 0)),
        "chunk_id": int(chunk.get("id", 0)),
        "text": (chunk.get("text") or "").strip(),
        "doc_id": chunk.get("doc_id") or doc_id,
        "page": int(page) if page is not None else 1,
        "char_start": int(chunk.get("char_start", 0)),
        "char_end": int(chunk.get("char_end", len(chunk.get("text") or ""))),
        "section_title": chunk.get("section_title"),
        "page_type": chunk.get("page_type") or "text",
        "lang": chunk.get("lang") or "en",
    }


def _iter_chunks(targets: Iterable[Path]):
    for path in targets:
        doc_id = path.stem.replace(".chunks", "")
        for chunk in _load_chunks(path):
            record = _normalise_chunk(doc_id, chunk)
            if record["text"]:
                yield record


def _existing_doc_ids(meta_path: Path) -> set[str]:
    if not meta_path.exists():
        return set()
    doc_ids: set[str] = set()
    with meta_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                doc_ids.add(json.loads(line)["doc_id"])
    return doc_ids


def _default_single_model() -> SentenceTransformer:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is required to load embedding models.")
    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    return SentenceTransformer(EN_EMBED_MODEL, device=device)


def add_chunks_to_index(
    chunk_paths: list[Path],
    index_dir: Path,
    model: Any | None = None,
    batch_size: int = BATCH_SIZE,
    models: dict[str, Any] | None = None,
) -> int:
    """Embed new chunk files once and append them to the FAISS index."""
    if not chunk_paths:
        return 0

    if models is None and model is None:
        models = load_embedding_models()
    if model is not None:
        models = {"en": model, "multi": model}

    assert models is not None
    dim = models["en"].get_sentence_embedding_dimension()

    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "faiss.index"
    meta_path = index_dir / "metadata.jsonl"
    index = load_or_create_index(index_path, dim)
    existing = _existing_doc_ids(meta_path)

    texts_batch: list[str] = []
    meta_batch: list[dict[str, Any]] = []
    active_key: str | None = None
    added = 0

    def flush() -> None:
        nonlocal added, active_key
        if not texts_batch or active_key is None:
            return
        embeddings = encode_texts(models[active_key], texts_batch)
        index.add(embeddings)
        with meta_path.open("a", encoding="utf-8") as file:
            for record in meta_batch:
                json.dump(record, file, ensure_ascii=False)
                file.write("\n")
        added += int(embeddings.shape[0])
        texts_batch.clear()
        meta_batch.clear()
        active_key = None

    for chunk_path in chunk_paths:
        doc_id = chunk_path.stem.replace(".chunks", "")
        if doc_id in existing:
            continue
        for record in _iter_chunks([chunk_path]):
            key = embedding_key_for_lang(record["lang"])
            if active_key is not None and key != active_key:
                flush()
            active_key = key
            metadata = dict(record)
            metadata["embedding_model"] = embedding_model_name_for_lang(record["lang"])
            texts_batch.append(record["text"])
            meta_batch.append(metadata)
            if len(texts_batch) >= batch_size:
                flush()
        flush()
        existing.add(doc_id)

    if added:
        faiss.write_index(index, str(index_path))
    return added


def build_index_full(
    input_targets: list[Path],
    out_dir: Path,
    batch_size: int = BATCH_SIZE,
    models: dict[str, Any] | None = None,
) -> int:
    if not input_targets:
        raise SystemExit("No .chunks.jsonl files found to index.")

    models = models or load_embedding_models()
    dim = models["en"].get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)
    records = list(_iter_chunks(input_targets))

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"
    if meta_path.exists():
        meta_path.unlink()

    by_key: dict[str, list[dict[str, Any]]] = {"en": [], "multi": []}
    for record in records:
        by_key[embedding_key_for_lang(record["lang"])].append(record)

    metadata_records: list[dict[str, Any]] = []
    for key, key_records in by_key.items():
        for offset in tqdm(range(0, len(key_records), batch_size), desc=f"Embedding {key}"):
            batch = key_records[offset : offset + batch_size]
            embeddings = encode_texts(models[key], [record["text"] for record in batch])
            index.add(embeddings)
            for record in batch:
                metadata = dict(record)
                metadata["embedding_model"] = embedding_model_name_for_lang(record["lang"])
                metadata_records.append(metadata)

    faiss.write_index(index, str(out_dir / "faiss.index"))
    with meta_path.open("w", encoding="utf-8") as file:
        for record in metadata_records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")

    print(f"Built index with {index.ntotal} vectors at {out_dir}")
    return int(index.ntotal)


def _gather_targets(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.chunks.jsonl"))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed chunk files into a FAISS index.")
    parser.add_argument("input", type=Path, help="A chunk JSONL file or directory.")
    parser.add_argument("--out", type=Path, default=Path("./data/index"), help="Index directory.")
    parser.add_argument("--full", action="store_true", help="Rebuild the index from scratch.")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Embedding batch size.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    targets = _gather_targets(args.input)
    if args.full:
        build_index_full(targets, args.out, args.batch)
    else:
        added = add_chunks_to_index(targets, args.out, model=_default_single_model(), batch_size=args.batch)
        print(f"Added {added} vector(s) to index")


if __name__ == "__main__":
    main()
