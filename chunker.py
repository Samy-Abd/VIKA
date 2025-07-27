#!/usr/bin/env python
"""
chunker.py – Split large text documents into overlapping chunks for downstream processing.

This utility is meant to follow the PDF‑intake pipeline: once a PDF has been
extracted to <hash>.txt, call this script to produce <hash>.chunks.jsonl (or
numbered *.txt files) containing manageable text segments.

Example
-------
> python chunker.py ./data/pdfs/abcd1234.txt --chunk_size 1200 --overlap 200

Dependencies (minimal)
----------------------
> pip install "langchain>=0.2"

Only the *text‑splitter* portion of LangChain is used, so installing the full
stack is **not** required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    # LangChain ≥0.2 ships the text splitter as part of the core package
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError as e:
    raise SystemExit(
        "Missing dependency: langchain. Install with\n  pip install 'langchain>=0.2'"
    ) from e

###############################################################################
# Core splitter helpers
###############################################################################

def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: tuple[str, ...] | None = None,
) -> list[str]:
    """Return *text* split into overlapping chunks.

    Parameters
    ----------
    text : str
        Raw document text.
    chunk_size : int, default ``1000``
        Maximum characters (UTF‑8 code points) per chunk.
    chunk_overlap : int, default ``200``
        Number of characters of overlap between consecutive chunks.
    separators : tuple[str, ...] | None
        Custom separator hierarchy passed to RecursiveCharacterTextSplitter.
        Defaults to ("\n\n", "\n", " ", "") matching LangChain defaults.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    return splitter.split_text(text)

###############################################################################
# I/O helpers
###############################################################################

def write_chunks_jsonl(chunks: list[str], out_path: Path):
    """Write *chunks* to *out_path* as newline‑delimited JSON objects."""
    with out_path.open("w", encoding="utf‑8") as f:
        for i, ch in enumerate(chunks):
            json.dump({"id": i, "text": ch}, f, ensure_ascii=False)
            f.write("\n")


def write_chunks_txt(chunks: list[str], base_path: Path):
    """Write each chunk to ``<base>_<idx>.txt`` side‑by‑side."""
    for i, ch in enumerate(chunks):
        (base_path.parent / f"{base_path.stem}_{i:03d}.txt").write_text(ch, encoding="utf‑8")

###############################################################################
# CLI wrapper
###############################################################################

def process_file(
    txt_path: Path,
    out_dir: Path | None,
    chunk_size: int,
    chunk_overlap: int,
    fmt: str,
) -> int:
    text = txt_path.read_text(encoding="utf‑8", errors="replace")
    chunks = chunk_text(text, chunk_size, chunk_overlap)

    # Choose output strategy
    if fmt == "jsonl":
        target = (out_dir or txt_path.parent) / f"{txt_path.stem}.chunks.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        write_chunks_jsonl(chunks, target)
    else:  # fmt == "txt"
        target_dir = out_dir or txt_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        write_chunks_txt(chunks, txt_path if out_dir is None else target_dir / txt_path.name)

    return len(chunks)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Split .txt documents into fixed‑size overlapping chunks.")
    p.add_argument(
        "input",
        type=Path,
        help="Path to a .txt file or directory containing .txt files.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (defaults to the same folder as each input).",
    )
    p.add_argument("--chunk_size", type=int, default=1000, help="Maximum characters per chunk (default: 1000).")
    p.add_argument("--overlap", type=int, default=200, help="Character overlap between chunks (default: 200).")
    p.add_argument("--fmt", choices=["jsonl", "txt"], default="jsonl", help="Output format (default: jsonl).")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    targets: list[Path]
    if args.input.is_file():
        targets = [args.input]
    else:
        targets = sorted(args.input.glob("*.txt"))

    if not targets:
        print("No .txt files found to process.", file=sys.stderr)
        sys.exit(1)

    for txt in targets:
        n = process_file(txt, args.out, args.chunk_size, args.overlap, args.fmt)
        print(f"✅ {txt.name}: {n} chunks written")


if __name__ == "__main__":  # pragma: no cover
    main()
