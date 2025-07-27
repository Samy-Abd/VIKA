from pathlib import Path
import numpy as np

import fitz  # PyMuPDF
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

def extract_text_fitz(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    return "\n".join(page.get_text() for page in doc)

def extract_text_ocr(pdf_path: Path) -> str:
    images = convert_from_path(str(pdf_path), dpi=300)
    text_blocks = []
    for image in images:
        result = ocr_engine.ocr(np.array(image), cls=True)
        lines = [" ".join([word_info[1][0] for word_info in line]) for line in result[0]]
        text_blocks.append("\n".join(lines))
    return "\n\n".join(text_blocks)

def extract_pdf_text(pdf_path: Path) -> dict:
    text = extract_text_fitz(pdf_path)
    if len(text.strip()) > 100:
        return {"method": "fitz", "text": text}
    else:
        return {"method": "ocr", "text": extract_text_ocr(pdf_path)}
