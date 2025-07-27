import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer  # Use the cross‑platform default


from parser_utils import extract_pdf_text

try:
    import pypdf  # optional deep‑validation
except ImportError:
    pypdf = None

###############################################################################
# CLI helpers
###############################################################################

def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch a staging folder for new PDFs, deduplicate via SHA‑256, "
                    "validate integrity, and move them into a canonical store with "
                    "hash‑based filenames. A manifest.csv and <hash>.meta.json sidecar "
                    "are maintained for provenance."
    )
    parser.add_argument("--staging", default="./incoming", help="Folder to watch for new PDFs")
    parser.add_argument("--store", default="./data/pdfs", help="Canonical PDF store")
    parser.add_argument("--manifest", default="./data/manifest.csv", help="CSV manifest path")
    parser.add_argument("--log", default="./intake.log", help="Log file path")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between size checks")
    parser.add_argument("--retries", type=int, default=3, help="Stable‑size checks before processing")
    # `--service` flag kept for compatibility; with the new simplified observer it is ignored.
    parser.add_argument("--service", action="store_true", help="(Deprecated) No longer needed")
    return parser.parse_args()

###############################################################################
# Logging
###############################################################################

def setup_logging(log_path: Path):
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
# Utility functions
###############################################################################

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_pdf_header(path: Path) -> bool:
    with path.open("rb") as f:
        header = f.read(1024)
    return b"%PDF-" in header


def pdf_loadable(path: Path) -> bool:
    if not pypdf:
        return True
    try:
        reader = pypdf.PdfReader(str(path))
        _ = reader.pages[0]
        return True
    except Exception:
        return False


def wait_for_stable_size(path: Path, sleep: float, retries: int) -> bool:
    prev = -1
    for _ in range(retries):
        size = path.stat().st_size
        if size == prev:
            return True
        prev = size
        time.sleep(sleep)
    return False

###############################################################################
# Manifest helpers
###############################################################################

def load_manifest(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["hash"] for row in reader}


def append_manifest(manifest: Path, row: dict):
    is_new = not manifest.exists()
    with manifest.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["hash", "original_name", "acquired", "bytes"])
        if is_new:
            writer.writeheader()
        writer.writerow(row)

###############################################################################
# Watchdog event handler
###############################################################################

class PDFHandler(FileSystemEventHandler):
    def __init__(self, cfg):
        self.cfg = cfg
        self.known = load_manifest(cfg.manifest)
        logging.info("Loaded %d hashes from manifest", len(self.known))

    # --- watchdog callback --------------------------------------------------
    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() != ".pdf":
            return
        self.process(path)

    # --- core pipeline ------------------------------------------------------
    def process(self, path: Path):
        logging.info("[NEW] %s", path)
        if not wait_for_stable_size(path, self.cfg.sleep, self.cfg.retries):
            logging.warning("File never stabilized: %s", path)
            return
        if not is_pdf_header(path):
            logging.warning("Missing %PDF- header: %s", path)
            return
        if not pdf_loadable(path):
            logging.warning("pypdf failed to load: %s", path)
            return
        h = sha256(path)
        if h in self.known:
            logging.info("Duplicate (hash=%s) – deleting %s", h, path)
            path.unlink(missing_ok=True)
            return
        dest = self.cfg.store / f"{h}.pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        path.replace(dest)
        meta = {
            "hash": h,
            "original_name": path.name,
            "acquired": datetime.utcnow().isoformat() + "Z",
            "bytes": dest.stat().st_size,
        }
        meta_path = dest.with_suffix(".meta.json")

        text_info = extract_pdf_text(dest)
        text_path = dest.with_suffix(".txt")
        with text_path.open("w", encoding="utf-8") as f:
            f.write(text_info["text"])

        logging.info("Text extracted using %s", text_info["method"])

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        append_manifest(self.cfg.manifest, meta)
        self.known.add(h)
        logging.info("Stored ✅ %s (%.1f KB)", dest, meta["bytes"] / 1024)

###############################################################################
# Entrypoint
###############################################################################

def main():
    cfg = build_args()
    cfg.staging = Path(cfg.staging).expanduser().resolve()
    cfg.store = Path(cfg.store).expanduser().resolve()
    cfg.manifest = Path(cfg.manifest).expanduser().resolve()
    cfg.log = Path(cfg.log).expanduser().resolve()

    cfg.staging.mkdir(parents=True, exist_ok=True)
    cfg.store.mkdir(parents=True, exist_ok=True)

    setup_logging(cfg.log)
    logging.info("PDF intake service starting…")
    logging.info("Staging:   %s", cfg.staging)
    logging.info("Store:     %s", cfg.store)

    handler = PDFHandler(cfg)

    # -------- new: bootstrap existing PDFs ----------------------------------
    pre_existing = list(cfg.staging.glob("*.pdf"))
    if pre_existing:
        logging.info("Scanning %d existing PDFs in staging…", len(pre_existing))
    for f in pre_existing:
        handler.process(f)

    # ------------------------------------------------------------------------
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
