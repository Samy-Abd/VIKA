"""PDF extraction utilities with page classification and targeted OCR."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import fitz

from page_classifier import classify_page

try:
    from PIL import Image
except ImportError:  # pragma: no cover - dependency is declared for the app
    Image = None

try:
    import pytesseract
except ImportError:  # pragma: no cover - dependency is declared for the app
    pytesseract = None

try:
    from langdetect import detect
except ImportError:  # pragma: no cover - dependency is declared for the app
    detect = None


OCR_LANG = "fra+eng"


def detect_language(text: str) -> str:
    """Detect document language, defaulting to English on short/invalid text."""
    if detect is None:
        return "en"
    try:
        sample = (text or "").strip()
        if not sample:
            return "en"
        return detect(sample)
    except Exception:
        return "en"


def _pixmap_to_image(pixmap: fitz.Pixmap) -> Any:
    if Image is None:
        return None
    return Image.open(io.BytesIO(pixmap.tobytes("png")))


def _ocr_pixmap(pixmap: fitz.Pixmap) -> str:
    if pytesseract is None:
        return ""
    image = _pixmap_to_image(pixmap)
    if image is None:
        return ""
    try:
        return pytesseract.image_to_string(image, lang=OCR_LANG).strip()
    except Exception:
        return ""


def _ocr_page(page: fitz.Page) -> str:
    pixmap = page.get_pixmap(dpi=300)
    return _ocr_pixmap(pixmap)


def _image_bboxes(page: fitz.Page) -> list[fitz.Rect]:
    """Return image bounding boxes for a page, tolerating PyMuPDF API variants."""
    boxes: list[fitz.Rect] = []
    seen: set[tuple[int, int, int, int]] = set()

    for image_info in page.get_images(full=True) or []:
        candidates = [image_info]
        if len(image_info) > 7:
            candidates.append(image_info[7])

        rect = None
        for candidate in candidates:
            try:
                rect = page.get_image_bbox(candidate)
                break
            except Exception:
                continue

        if rect is None or rect.is_empty or rect.is_infinite:
            continue

        key = (round(rect.x0), round(rect.y0), round(rect.x1), round(rect.y1))
        if key in seen:
            continue
        seen.add(key)
        boxes.append(rect)

    return boxes


def _ocr_image_zones(page: fitz.Page) -> list[tuple[float, float, str]]:
    entries: list[tuple[float, float, str]] = []
    for rect in _image_bboxes(page):
        try:
            pixmap = page.get_pixmap(dpi=300, clip=rect)
        except Exception:
            continue
        text = _ocr_pixmap(pixmap)
        if text:
            entries.append((float(rect.y0), float(rect.x0), text))
    return entries


def _text_block_entries(page: fitz.Page) -> list[tuple[float, float, str]]:
    entries: list[tuple[float, float, str]] = []
    for block in page.get_text("blocks") or []:
        if len(block) >= 7 and block[6] != 0:
            continue
        text = str(block[4]).strip() if len(block) > 4 else ""
        if text:
            entries.append((float(block[1]), float(block[0]), text))
    return entries


def extract_page_text(page: fitz.Page, page_type: str) -> str:
    """Extract text from one classified PDF page."""
    if page_type in {"text", "illustrative"}:
        return (page.get_text("text") or "").strip()

    if page_type == "scanned":
        return _ocr_page(page)

    if page_type == "mixed":
        entries = _text_block_entries(page)
        entries.extend(_ocr_image_zones(page))
        entries.sort(key=lambda item: (item[0], item[1]))
        return "\n".join(text for _, _, text in entries if text).strip()

    return (page.get_text("text") or "").strip()


def extract_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract page-aware records from a PDF."""
    records: list[dict[str, Any]] = []
    with fitz.open(str(pdf_path)) as doc:
        for page_index, page in enumerate(doc, start=1):
            page_type = classify_page(page)
            text = extract_page_text(page, page_type)
            records.append(
                {
                    "page": page_index,
                    "text": text,
                    "page_type": page_type,
                    "char_start": 0,
                    "char_end": len(text),
                }
            )
    return records


def extract_pdf_text(pdf_path: Path) -> dict[str, Any]:
    """Extract a PDF into page records plus a flat text compatibility view."""
    pages = extract_pdf_pages(pdf_path)
    full_text = "\n\n".join(page["text"] for page in pages if page.get("text"))
    lang = detect_language(full_text)
    for page in pages:
        page["lang"] = lang

    return {
        "method": "pymupdf+tesseract",
        "text": full_text,
        "pages": pages,
        "lang": lang,
    }
