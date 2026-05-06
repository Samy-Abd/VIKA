"""Prompt construction for page-cited RAG answers."""
from __future__ import annotations

import csv
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from jinja2 import Template
except ImportError:  # pragma: no cover - dependency is declared
    Template = None


ROOT = Path(__file__).parent
MANIFEST = ROOT / "data" / "manifest.csv"
MAX_CITATION_LABEL_LEN = 48
_manifest_lock = threading.Lock()


def _load_manifest_mapping() -> dict[str, str]:
    if not MANIFEST.exists():
        return {}

    mapping: dict[str, str] = {}
    with _manifest_lock:
        with MANIFEST.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                doc_hash = (row.get("hash") or "").strip()
                original = (row.get("original") or "").strip()
                if doc_hash and original:
                    mapping[doc_hash] = original
    return mapping


def truncate_citation_label(label: str, max_len: int = MAX_CITATION_LABEL_LEN) -> str:
    label = " ".join((label or "unknown").replace("[", "(").replace("]", ")").split())
    if len(label) <= max_len:
        return label

    suffix = Path(label).suffix
    if suffix and len(suffix) + 8 < max_len:
        keep = max_len - len(suffix) - 3
        return f"{label[:keep]}...{suffix}"
    return f"{label[: max_len - 3]}..."


def citation_label_for_doc(doc_id: str) -> str:
    original = _load_manifest_mapping().get(doc_id, doc_id)
    return truncate_citation_label(original)


SYSTEM_PROMPT = """\
You are VIKA, a scientific-document assistant.

Grounding rules:
- Use only the retrieved context passages to answer.
- Cite every factual claim that comes from context with [source filename p.PAGE].
- If the answer is not supported by the retrieved context, say: "I cannot answer from the uploaded documents."
- Do not invent sources, page numbers, or document details.

Style:
- Answer in the same language as the user.
- Be concise, rigorous, and useful for students.
- Use equations or scientific notation when they improve clarity.
"""

DEFAULT_TEMPLATE = """\
{{ system_prompt }}

{% if history %}
Conversation so far:
{% for turn in history %}
User: {{ turn[0] }}
Assistant: {{ turn[1] }}
{% endfor %}
{% endif %}
{% if chunks %}
Retrieved context:
{% for c in chunks %}
Source: [{{ c.source_label }} p.{{ c.page }}]
Page type: {{ c.page_type }}
{{ c.text }}
---
{% endfor %}
{% else %}
No context passages were retrieved. The assistant must say: "I cannot answer from the uploaded documents."
{% endif %}
Question: {{ query }}

Answer:
"""


@dataclass
class Chunk:
    text: str
    doc_id: str
    page: int | str
    page_type: str = "text"
    source_label: str = "unknown"

    @classmethod
    def from_mapping(cls, obj: dict) -> "Chunk":
        page = obj.get("page")
        doc_id = str(obj.get("doc_id") or obj.get("source") or "unknown")
        return cls(
            text=obj.get("text", ""),
            doc_id=doc_id,
            page=page if page is not None else "?",
            page_type=obj.get("page_type") or "text",
            source_label=citation_label_for_doc(doc_id),
        )


def _render_template(
    query: str,
    chunks: Sequence[Chunk],
    history: Sequence[Sequence[str]],
    template_str: str | None = None,
) -> str:
    template_str = template_str or DEFAULT_TEMPLATE
    history = history[-5:]

    if Template is not None:
        template = Template(template_str, autoescape=False)
        return template.render(
            query=query.strip(),
            chunks=chunks,
            history=history,
            system_prompt=SYSTEM_PROMPT,
        )

    lines = [SYSTEM_PROMPT, ""]
    if history:
        lines.append("Conversation so far:")
        for user, assistant in history:
            lines.append(f"User: {user}")
            lines.append(f"Assistant: {assistant}")
    if chunks:
        lines.append("Retrieved context:")
        for chunk in chunks:
            lines.append(f"Source: [{chunk.source_label} p.{chunk.page}]")
            lines.append(f"Page type: {chunk.page_type}")
            lines.append(chunk.text.rstrip())
            lines.append("---")
    else:
        lines.append(
            'No context passages were retrieved. The assistant must say: '
            '"I cannot answer from the uploaded documents."'
        )
    lines.append(f"Question: {query.strip()}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def build_prompt(
    query: str,
    raw_chunks: Iterable[dict] | Iterable[Chunk],
    template: str | None = None,
    history: Sequence[Sequence[str]] | None = None,
) -> str:
    chunks = [chunk if isinstance(chunk, Chunk) else Chunk.from_mapping(chunk) for chunk in raw_chunks]
    return _render_template(query, chunks, history or [], template)
