"""PDF intake service
~~~~~~~~~~~~~~~~~~~~
Monitors a *staging* directory for new PDF files, validates/­deduplicates them via
SHA‑256, then moves each file into a canonical *store* under its hash‑based
filename.  For provenance, a side‑car `<hash>.meta.json` and a central
`manifest.csv` are maintained.  The manifest keeps a **minimal two‑column**
(mapping *hash → original filename*) so that downstream UIs (e.g. Gradio
sidebar) can always recover the human‑friendly name without parsing JSON.

Run standalone:
    python document_intake.py  --staging ./incoming  --store ./data/pdfs
"""
from __future__ import annotations

###############################################################################
# Imports
###############################################################################
import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Set

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer  # cross‑platform default

from parser_utils import extract_pdf_text

try:
    import pypdf  # optional deep validation
except ImportError:  # graceful degradation: just skip the extra check
    pypdf = None

###############################################################################
# CLI helpers
###############################################################################

def build_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Watch a staging folder for new PDFs, deduplicate via SHA‑256, "
            "validate integrity, and move them into a canonical store with "
            "hash‑based filenames.  A manifest.csv and <hash>.meta.json "
            "side‑car are maintained for provenance."
        )
    )
    parser.add_argument("--staging", default="./incoming", help="Folder to watch for new PDFs")
    parser.add_argument("--store", default="./data/pdfs", help="Canonical PDF store")
    parser.add_argument("--manifest", default="./data/manifest.csv", help="CSV manifest path")
    parser.add_argument("--log", default="./intake.log", help="Log file path")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between size checks")
    parser.add_argument(
        "--retries", type=int, default=3, help="Stable‑size checks before processing"
    )
    # "--service" kept for CI/backwards compatibility; has no effect now.
    parser.add_argument("--service", action="store_true", help="(Deprecated) No longer needed")
    return parser.parse_args()

###############################################################################
# Logging helpers
###############################################################################

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

###############################################################################
# Generic utilities
###############################################################################

def sha256(path: Path) -> str:
    """Return the SHA‑256 digest of *path* (binary)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_pdf_header(path: Path) -> bool:
    """Quick check for a %PDF‑ header in the first KB."""
    with path.open("rb") as f:
        return b"%PDF-" in f.read(1024)


def pdf_loadable(path: Path) -> bool:
    """Deep validation using *pypdf* (if available)."""
    if not pypdf:
        return True
    try:
        reader = pypdf.PdfReader(str(path))
        # attempt to load first page – catches most corruptions
        _ = reader.pages[0]
        return True
    except Exception:
        return False


def wait_for_stable_size(path: Path, sleep: float, retries: int) -> bool:
    """Return *True* once file size stops growing for *retries* checks."""
    prev = -1
    for _ in range(retries):
        size = path.stat().st_size
        if size == prev:
            return True
        prev = size
        time.sleep(sleep)
    return False

###############################################################################
# Manifest helpers  – *minimal mapping* (hash ↔ original filename)
###############################################################################

def _record_in_manifest(manifest: Path, pdf_hash: str, original_name: str) -> None:
    """Append *(hash, original)* to *manifest* if that hash is not present yet."""
    manifest.parent.mkdir(parents=True, exist_ok=True)
    exists = manifest.exists()

    # Avoid quadratic scans on every write – keep whole set in memory if needed.
    already: Set[str]
    if exists:
        with manifest.open(newline="", encoding="utf-8") as fp:
            already = {row["hash"] for row in csv.DictReader(fp)}
    else:
        already = set()

    if pdf_hash in already:
        return

    with manifest.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["hash", "original"])
        if not exists:
            writer.writeheader()
        writer.writerow({"hash": pdf_hash, "original": original_name})


def load_manifest(manifest: Path) -> Set[str]:
    """Return the set of *hashes* that are already listed in *manifest*."""
    if not manifest.exists():
        return set()
    with manifest.open(newline="", encoding="utf-8") as fp:
        return {row["hash"] for row in csv.DictReader(fp)}

###############################################################################
# Watchdog event handler
###############################################################################

class PDFHandler(FileSystemEventHandler):
    """React to new PDFs appearing in the *staging* directory."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.known = load_manifest(cfg.manifest)
        logging.info("Loaded %d hashes from manifest", len(self.known))

    # ---- watchdog callback -----------------------------------------------
    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() != ".pdf":
            return
        self.process(path)

    # ---- core pipeline ----------------------------------------------------
    def process(self, path: Path) -> None:
        """Full ingestion routine for a single *path*."""
        logging.info("[NEW] %s", path)

        # 1) wait until the file upload is complete (size no longer changes)
        if not wait_for_stable_size(path, self.cfg.sleep, self.cfg.retries):
            logging.warning("File never stabilised: %s", path)
            return

        # 2) lightweight syntactic validation
        if not is_pdf_header(path):
            logging.warning("Missing %PDF‑ header: %s", path)
            return
        if not pdf_loadable(path):
            logging.warning("pypdf failed to load: %s", path)
            return

        # 3) deduplication via SHA‑256
        h = sha256(path)
        if h in self.known:
            logging.info("Duplicate (hash=%s) – deleting %s", h, path)
            path.unlink(missing_ok=True)
            return

        # 4) move into canonical store (hash‑based filename)
        dest = self.cfg.store / f"{h}.pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        path.replace(dest)

        # 5) provenance side‑car metadata
        meta = {
            "hash": h,
            "original_name": path.name,
            "acquired": datetime.utcnow().isoformat() + "Z",
            "bytes": dest.stat().st_size,
        }
        meta_path = dest.with_suffix(".meta.json")
        with meta_path.open("w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2)

        # 6) extract raw text (stored beside PDF for downstream chunker)
        text_info = extract_pdf_text(dest)
        text_path = dest.with_suffix(".txt")
        with text_path.open("w", encoding="utf-8") as fp:
            fp.write(text_info["text"])
        logging.info("Text extracted using %s", text_info["method"])

        # 7) update manifest (hash ↔ original filename) and in‑memory cache
        _record_in_manifest(self.cfg.manifest, h, path.name)
        self.known.add(h)

        logging.info("Stored ✅ %s (%.1f KB)", dest, meta["bytes"] / 1024)

###############################################################################
# Entrypoint – run as a service
###############################################################################

def main() -> None:
    cfg = build_args()

    # Canonicalise/expand user paths
    cfg.staging = Path(cfg.staging).expanduser().resolve()
    cfg.store = Path(cfg.store).expanduser().resolve()
    cfg.manifest = Path(cfg.manifest).expanduser().resolve()
    cfg.log = Path(cfg.log).expanduser().resolve()

    cfg.staging.mkdir(parents=True, exist_ok=True)
    cfg.store.mkdir(parents=True, exist_ok=True)

    setup_logging(cfg.log)
    logging.info("PDF intake service starting…")
    logging.info("Staging: %s", cfg.staging)
    logging.info("Store:   %s", cfg.store)

    handler = PDFHandler(cfg)

    # -- bootstrap: process any PDFs already present in *staging* -----------
    pre_existing = list(cfg.staging.glob("*.pdf"))
    if pre_existing:
        logging.info("Scanning %d existing PDFs in staging…", len(pre_existing))
    for f in pre_existing:
        handler.process(f)

    # -- live monitoring ----------------------------------------------------
    observer = Observer()
    observer.schedule(handler, str(cfg.staging), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down…")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
