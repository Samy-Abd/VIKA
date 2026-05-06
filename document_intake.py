"""PDF intake service for runtime uploads.

The intake layer validates PDFs, deduplicates by SHA-256, stores each upload
under a content-addressed filename, and writes page-aware extraction sidecars for
the chunker.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Set

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:  # pragma: no cover - declared in requirements
    class FileSystemEventHandler:  # type: ignore[no-redef]
        pass

    Observer = None

from parser_utils import extract_pdf_text

try:
    import pypdf
except ImportError:  # pragma: no cover - optional deeper validation
    pypdf = None


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch a staging folder for uploaded PDFs and ingest them."
    )
    parser.add_argument("--staging", default="./incoming", help="Folder to watch for new PDFs")
    parser.add_argument("--store", default="./data/pdfs", help="Canonical PDF store")
    parser.add_argument("--manifest", default="./data/manifest.csv", help="CSV manifest path")
    parser.add_argument("--log", default="./intake.log", help="Log file path")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between size checks")
    parser.add_argument("--retries", type=int, default=3, help="Stable-size checks")
    parser.add_argument("--service", action="store_true", help="Deprecated compatibility flag")
    return parser.parse_args()


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_pdf_header(path: Path) -> bool:
    with path.open("rb") as file:
        return b"%PDF-" in file.read(1024)


def pdf_loadable(path: Path) -> bool:
    if pypdf is None:
        return True
    try:
        reader = pypdf.PdfReader(str(path))
        _ = reader.pages[0]
        return True
    except Exception:
        return False


def wait_for_stable_size(path: Path, sleep: float, retries: int) -> bool:
    previous = -1
    for _ in range(retries):
        size = path.stat().st_size
        if size == previous:
            return True
        previous = size
        time.sleep(sleep)
    return False


def load_manifest(manifest: Path) -> Set[str]:
    if not manifest.exists():
        return set()
    with manifest.open(newline="", encoding="utf-8") as file:
        return {row["hash"] for row in csv.DictReader(file) if row.get("hash")}


def _record_in_manifest(manifest: Path, pdf_hash: str, original_name: str) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    exists = manifest.exists()
    known = load_manifest(manifest) if exists else set()
    if pdf_hash in known:
        return

    with manifest.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["hash", "original"])
        if not exists:
            writer.writeheader()
        writer.writerow({"hash": pdf_hash, "original": original_name})


def _write_pages_sidecar(path: Path, pages: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for page in pages:
            json.dump(page, file, ensure_ascii=False)
            file.write("\n")


class PDFHandler(FileSystemEventHandler):
    """React to PDF files appearing in the staging directory."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.known = load_manifest(cfg.manifest)
        logging.info("Loaded %d known document hashes", len(self.known))

    def on_created(self, event: Any) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() == ".pdf":
            self.process(path)

    def process(self, path: Path) -> None:
        logging.info("Processing upload: %s", path)

        if not wait_for_stable_size(path, self.cfg.sleep, self.cfg.retries):
            logging.warning("File never stabilized: %s", path)
            return
        if not is_pdf_header(path):
            logging.warning("Missing PDF header: %s", path)
            return
        if not pdf_loadable(path):
            logging.warning("PDF validation failed: %s", path)
            return

        pdf_hash = sha256(path)
        if pdf_hash in self.known:
            logging.info("Duplicate upload skipped: %s", path)
            path.unlink(missing_ok=True)
            return

        dest = self.cfg.store / f"{pdf_hash}.pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        path.replace(dest)

        extraction = extract_pdf_text(dest)
        text_path = dest.with_suffix(".txt")
        text_path.write_text(extraction["text"], encoding="utf-8")
        _write_pages_sidecar(dest.with_suffix(".pages.jsonl"), extraction["pages"])

        meta = {
            "hash": pdf_hash,
            "original_name": path.name,
            "acquired": datetime.now(timezone.utc).isoformat(),
            "bytes": dest.stat().st_size,
            "extract_method": extraction["method"],
            "lang": extraction["lang"],
        }
        with dest.with_suffix(".meta.json").open("w", encoding="utf-8") as file:
            json.dump(meta, file, ensure_ascii=False, indent=2)

        _record_in_manifest(self.cfg.manifest, pdf_hash, path.name)
        self.known.add(pdf_hash)
        logging.info("Stored %s with %d page records", dest, len(extraction["pages"]))


def main() -> None:
    if Observer is None:
        raise SystemExit("watchdog is required to run document_intake.py as a service.")

    cfg = build_args()
    cfg.staging = Path(cfg.staging).expanduser().resolve()
    cfg.store = Path(cfg.store).expanduser().resolve()
    cfg.manifest = Path(cfg.manifest).expanduser().resolve()
    cfg.log = Path(cfg.log).expanduser().resolve()

    cfg.staging.mkdir(parents=True, exist_ok=True)
    cfg.store.mkdir(parents=True, exist_ok=True)

    setup_logging(cfg.log)
    handler = PDFHandler(cfg)

    for pdf in sorted(cfg.staging.glob("*.pdf")):
        handler.process(pdf)

    observer = Observer()
    observer.schedule(handler, str(cfg.staging), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
