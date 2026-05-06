"""PDF page classification helpers for VIKA."""
from __future__ import annotations

from typing import Any

TEXT_DENSITY_THRESHOLD = 0.01
LOW_TEXT_DENSITY_THRESHOLD = 0.001


def classify_page_from_features(text: str, page_area: float, image_count: int) -> str:
    """Classify a PDF page from raw text, page area, and image count.

    The thresholds intentionally follow the project specification exactly.
    """
    area = max(float(page_area or 0.0), 1.0)
    text_density = len((text or "").strip()) / area
    has_images = image_count > 0

    if text_density >= TEXT_DENSITY_THRESHOLD and not has_images:
        return "text"
    if text_density >= TEXT_DENSITY_THRESHOLD and has_images:
        return "illustrative"
    if text_density < LOW_TEXT_DENSITY_THRESHOLD and has_images:
        return "scanned"
    if LOW_TEXT_DENSITY_THRESHOLD <= text_density < TEXT_DENSITY_THRESHOLD and has_images:
        return "mixed"
    return "text"


def page_features(page: Any) -> dict[str, Any]:
    """Return the PyMuPDF-derived features used for classification."""
    text = page.get_text("text") or ""
    images = page.get_images(full=True) or []
    area = float(getattr(page.rect, "area", 0.0) or 0.0)
    return {
        "text": text,
        "page_area": area,
        "image_count": len(images),
        "text_density": len(text.strip()) / max(area, 1.0),
        "has_images": len(images) > 0,
    }


def classify_page(page: Any) -> str:
    """Classify a PyMuPDF page as text, illustrative, scanned, or mixed."""
    features = page_features(page)
    return classify_page_from_features(
        features["text"],
        features["page_area"],
        features["image_count"],
    )
