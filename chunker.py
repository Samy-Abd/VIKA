"""Page-aware chunking for VIKA documents."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from parser_utils import detect_language


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def chunk_text_with_offsets(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[tuple[str, int, int]]:
    """Split text into overlapping chunks while preserving character offsets."""
    text = text or ""
    if not text.strip():
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[tuple[str, int, int]] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            window = text[start:end]
            split_at = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
            if split_at > chunk_size // 2:
                end = start + split_at

        raw = text[start:end]
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        adjusted_start = start + leading
        adjusted_end = start + trailing
        chunk = text[adjusted_start:adjusted_end]
        if chunk:
            chunks.append((chunk, adjusted_start, adjusted_end))

        if end >= text_len:
            break
        next_start = max(end - chunk_overlap, start + 1)
        start = next_start

    return chunks


def extract_section_title(text: str) -> str | None:
    """Find a lightweight section title candidate from page text."""
    for raw_line in (text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or len(line) > 120:
            continue
        words = line.split()
        if len(words) > 14:
            continue
        if re.match(r"^\d+(\.\d+)*\s+\S+", line):
            return line
        if line.isupper() and any(ch.isalpha() for ch in line):
            return line
        if len(words) >= 2 and line[:1].isupper() and not line.endswith("."):
            return line
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_page_records(txt_path: Path) -> list[dict[str, Any]]:
    pages_path = txt_path.with_suffix(".pages.jsonl")
    meta = _read_json(txt_path.with_suffix(".meta.json"))
    doc_lang = meta.get("lang")

    if pages_path.exists():
        records: list[dict[str, Any]] = []
        with pages_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                page = json.loads(line)
                page_number = page.get("page") or line_number
                records.append(
                    {
                        "page": int(page_number),
                        "text": page.get("text", ""),
                        "page_type": page.get("page_type") or "text",
                        "lang": page.get("lang") or doc_lang,
                    }
                )
        if records:
            fallback_lang = doc_lang or detect_language("\n".join(r["text"] for r in records))
            for record in records:
                record["lang"] = record.get("lang") or fallback_lang
            return records

    text = txt_path.read_text(encoding="utf-8", errors="replace")
    lang = doc_lang or detect_language(text)
    return [{"page": 1, "text": text, "page_type": "text", "lang": lang}]


def build_chunks_for_file(
    txt_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Build page-aware chunk records for one extracted document."""
    doc_id = txt_path.stem
    chunks: list[dict[str, Any]] = []
    chunk_id = 0

    for page_record in _load_page_records(txt_path):
        page_text = page_record.get("text", "")
        section_title = extract_section_title(page_text)
        for chunk_text, char_start, char_end in chunk_text_with_offsets(
            page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ):
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "page": int(page_record["page"]),
                    "char_start": int(char_start),
                    "char_end": int(char_end),
                    "section_title": section_title,
                    "page_type": page_record.get("page_type") or "text",
                    "lang": page_record.get("lang") or "en",
                }
            )
            chunk_id += 1

    return chunks


def write_chunks_jsonl(chunks: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            json.dump(chunk, file, ensure_ascii=False)
            file.write("\n")


def write_chunks_txt(chunks: list[dict[str, Any]], base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        out = base_path.parent / f"{base_path.stem}_{chunk['id']:03d}.txt"
        out.write_text(chunk["text"], encoding="utf-8")


def process_file(
    txt_path: Path,
    out_dir: Path | None,
    chunk_size: int,
    chunk_overlap: int,
    fmt: str,
) -> int:
    chunks = build_chunks_for_file(txt_path, chunk_size, chunk_overlap)
    target_dir = out_dir or txt_path.parent

    if fmt == "jsonl":
        write_chunks_jsonl(chunks, target_dir / f"{txt_path.stem}.chunks.jsonl")
    else:
        write_chunks_txt(chunks, target_dir / txt_path.name)

    return len(chunks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split extracted PDF text into page-aware chunks.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--fmt", choices=["jsonl", "txt"], default="jsonl")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    targets = [args.input] if args.input.is_file() else sorted(args.input.glob("*.txt"))

    if not targets:
        print("No .txt files found to process.", file=sys.stderr)
        sys.exit(1)

    for txt_path in targets:
        count = process_file(txt_path, args.out, args.chunk_size, args.overlap, args.fmt)
        print(f"{txt_path.name}: {count} chunks written")


if __name__ == "__main__":
    main()
