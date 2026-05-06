from __future__ import annotations

import json
from pathlib import Path

import faiss
import fitz
import numpy as np

import chunker
import embed_faiss
from parser_utils import extract_pdf_text


class DummyEmbeddingModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        rows = []
        for index, _ in enumerate(texts, start=1):
            vector = np.array([index, 1.0, 0.0, 0.0], dtype="float32")
            if normalize_embeddings:
                vector = vector / np.linalg.norm(vector)
            rows.append(vector)
        return np.vstack(rows).astype("float32")


def _write_chunks(path: Path, chunks: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            json.dump(chunk, file)
            file.write("\n")
    return path


def test_index_size_increases_by_expected_chunk_count(workspace_tmp_path: Path):
    tmp_path = workspace_tmp_path
    chunks_path = _write_chunks(
        tmp_path / "doc1.chunks.jsonl",
        [
            {
                "id": 0,
                "text": "first chunk",
                "doc_id": "doc1",
                "page": 1,
                "char_start": 0,
                "char_end": 11,
                "section_title": None,
                "page_type": "text",
                "lang": "en",
            },
            {
                "id": 1,
                "text": "second chunk",
                "doc_id": "doc1",
                "page": 2,
                "char_start": 0,
                "char_end": 12,
                "section_title": None,
                "page_type": "text",
                "lang": "en",
            },
        ],
    )

    added = embed_faiss.add_chunks_to_index([chunks_path], tmp_path / "index", model=DummyEmbeddingModel())
    index = faiss.read_index(str(tmp_path / "index" / "faiss.index"))

    assert added == 2
    assert index.ntotal == 2


def test_page_field_is_never_none_for_standard_text_pdf(workspace_tmp_path: Path):
    tmp_path = workspace_tmp_path
    pdf_path = tmp_path / "standard.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Standard text PDF page with extractable text. " * 10)
    doc.save(pdf_path)
    doc.close()

    extraction = extract_pdf_text(pdf_path)
    txt_path = tmp_path / "standard.txt"
    txt_path.write_text(extraction["text"], encoding="utf-8")
    with txt_path.with_suffix(".pages.jsonl").open("w", encoding="utf-8") as file:
        for page_record in extraction["pages"]:
            json.dump(page_record, file)
            file.write("\n")

    chunker.process_file(txt_path, None, 120, 20, "jsonl")
    embed_faiss.add_chunks_to_index(
        [txt_path.with_suffix(".chunks.jsonl")],
        tmp_path / "index",
        model=DummyEmbeddingModel(),
    )

    metadata = [
        json.loads(line)
        for line in (tmp_path / "index" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metadata
    assert all(record["page"] is not None for record in metadata)
