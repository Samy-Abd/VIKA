from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import fitz
from PIL import Image

import chunker
from document_intake import PDFHandler
from page_classifier import classify_page
from parser_utils import extract_pdf_text


def _make_text_pdf(path: Path, page_texts: list[str]) -> Path:
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    doc.save(path)
    doc.close()
    return path


def _write_extraction_sidecars(pdf_path: Path, txt_path: Path) -> None:
    extraction = extract_pdf_text(pdf_path)
    txt_path.write_text(extraction["text"], encoding="utf-8")
    with txt_path.with_suffix(".pages.jsonl").open("w", encoding="utf-8") as file:
        for page in extraction["pages"]:
            json.dump(page, file)
            file.write("\n")


def test_three_page_pdf_chunks_have_non_null_page(workspace_tmp_path: Path):
    tmp_path = workspace_tmp_path
    pdf_path = _make_text_pdf(
        tmp_path / "three_pages.pdf",
        [
            "Page one contains alpha concepts. " * 8,
            "Page two contains beta concepts. " * 8,
            "Page three contains gamma concepts. " * 8,
        ],
    )
    txt_path = tmp_path / "doc.txt"
    _write_extraction_sidecars(pdf_path, txt_path)

    chunks = chunker.build_chunks_for_file(txt_path, chunk_size=120, chunk_overlap=20)

    assert chunks
    assert all(chunk["page"] is not None for chunk in chunks)
    assert {chunk["page"] for chunk in chunks} == {1, 2, 3}


def test_duplicated_document_is_not_ingested_twice(workspace_tmp_path: Path):
    tmp_path = workspace_tmp_path
    store = tmp_path / "store"
    manifest = tmp_path / "manifest.csv"
    first = _make_text_pdf(tmp_path / "first.pdf", ["Duplicate document text."])
    second = tmp_path / "second.pdf"
    second.write_bytes(first.read_bytes())

    cfg = SimpleNamespace(store=store, manifest=manifest, sleep=0, retries=2)
    handler = PDFHandler(cfg)
    handler.process(first)
    handler.process(second)

    assert len(list(store.glob("*.pdf"))) == 1
    assert len(list(store.glob("*.txt"))) == 1
    assert not second.exists()


def test_text_only_page_is_classified_as_text(workspace_tmp_path: Path):
    tmp_path = workspace_tmp_path
    pdf_path = _make_text_pdf(tmp_path / "text.pdf", ["Text-only page. " * 50])
    with fitz.open(pdf_path) as doc:
        assert classify_page(doc[0]) == "text"


def test_scanned_page_is_classified_as_scanned(workspace_tmp_path: Path):
    tmp_path = workspace_tmp_path
    image_path = tmp_path / "scan.png"
    Image.new("RGB", (200, 100), "white").save(image_path)

    pdf_path = tmp_path / "scan.pdf"
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_image(fitz.Rect(40, 40, 260, 160), filename=str(image_path))
    doc.save(pdf_path)
    doc.close()

    with fitz.open(pdf_path) as doc:
        assert classify_page(doc[0]) == "scanned"
