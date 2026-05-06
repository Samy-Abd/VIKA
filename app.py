from __future__ import annotations

import csv
import html
import json
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import faiss
import gradio as gr
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import chunker
import embed_faiss
import llm_router
import retriever
from document_intake import PDFHandler
from parser_utils import detect_language
from prompt_builder import build_prompt
from reranker import CrossEncoderReranker


ROOT = Path(__file__).parent
PDF_STORE = ROOT / "data" / "pdfs"
INDEX_DIR = ROOT / "data" / "index"
MANIFEST = ROOT / "data" / "manifest.csv"
REPORT_PATH = Path(tempfile.gettempdir()) / "vika_session_evaluation.json"

DEFAULT_RETRIEVAL_MODE = "hybrid"
TOP_CANDIDATES = 20
TOP_CONTEXT_CHUNKS = 5

PDF_STORE.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


CSS = """
.gradio-container { max-width: 100% !important; }
footer { display: none !important; }
#sidebar {
    background: var(--color-background-secondary);
    border-right: 1px solid var(--color-border-tertiary);
    border-radius: 8px 0 0 8px;
    padding: 16px 12px;
    min-height: 80vh;
}
#doc-list { font-size: 13px; line-height: 1.8; }
#topbar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid var(--color-border-tertiary);
    margin-bottom: 8px;
}
#chatbot {
    border: 0.5px solid var(--color-border-tertiary) !important;
    border-radius: 8px !important;
}
#chatbot .message { font-size: 14px; line-height: 1.6; }
.source-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: var(--color-background-secondary);
    border: 0.5px solid var(--color-border-tertiary);
    border-radius: 16px;
    padding: 2px 10px;
    font-size: 11px;
    color: var(--color-text-secondary);
    margin: 2px 3px 0 0;
}
#ctx-panel {
    background: var(--color-background-secondary);
    border-left: 1px solid var(--color-border-tertiary);
    border-radius: 0 8px 8px 0;
    padding: 14px 12px;
    min-height: 80vh;
}
#ctx-panel .stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 14px;
}
#ctx-panel .stat-card {
    background: var(--color-background-primary);
    border: 0.5px solid var(--color-border-tertiary);
    border-radius: 8px;
    padding: 8px 10px;
    text-align: center;
}
#ctx-panel .stat-num { font-size: 22px; font-weight: 500; }
#ctx-panel .stat-lbl { font-size: 11px; color: var(--color-text-secondary); }
#ctx-panel .chunk-card {
    background: var(--color-background-primary);
    border: 0.5px solid var(--color-border-tertiary);
    border-radius: 8px;
    padding: 8px 10px;
    margin-bottom: 6px;
    font-size: 12px;
}
#ctx-panel .chunk-header {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 4px;
    font-size: 11px;
}
#ctx-panel .chunk-score { font-weight: 500; color: #1d8f6f; }
#ctx-panel .chunk-name { color: var(--color-text-secondary); overflow-wrap: anywhere; }
#ctx-panel .score-bar {
    height: 3px;
    background: var(--color-border-tertiary);
    border-radius: 2px;
    margin: 4px 0;
}
#ctx-panel .score-fill { height: 3px; border-radius: 2px; background: #1d8f6f; }
#ctx-panel .chunk-text {
    color: var(--color-text-secondary);
    font-size: 11px;
    line-height: 1.4;
    margin-top: 4px;
}
#ctx-panel .chunk-meta {
    color: var(--color-text-tertiary);
    font-size: 10px;
    margin-top: 3px;
}
#ctx-panel .section-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--color-text-tertiary);
    margin-bottom: 8px;
}
"""


_manifest_lock = threading.Lock()
_runtime_lock = threading.RLock()
_eval_lock = threading.Lock()

_embed_models: dict[str, SentenceTransformer] | None = None
_index: faiss.Index | None = None
_meta: list[dict[str, Any]] | None = None
_bm25: retriever.BM25Bundle | None = None
_reranker: CrossEncoderReranker | None = None
_eval_records: list[dict[str, Any]] = []

if not REPORT_PATH.exists():
    REPORT_PATH.write_text("[]", encoding="utf-8")


def _load_manifest() -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not MANIFEST.exists():
        return mapping
    with _manifest_lock:
        with MANIFEST.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                doc_hash = (row.get("hash") or "").strip()
                original = (row.get("original") or "").strip()
                if doc_hash and original:
                    mapping[doc_hash] = original
    return mapping


def list_documents_md() -> str:
    pdfs = sorted(PDF_STORE.glob("*.pdf"))
    if not pdfs:
        return "*No documents indexed yet.*"
    mapping = _load_manifest()
    lines = ["### Documents"]
    for pdf in pdfs:
        lines.append(f"- {mapping.get(pdf.stem, pdf.stem)}")
    return "\n".join(lines)


def _empty_ctx_html() -> str:
    return (
        "<div style='color:var(--color-text-tertiary);font-size:13px;"
        "margin-top:20px;text-align:center'>Ask a question to see retrieved context here.</div>"
    )


def _score_percent(score: float) -> int:
    if 0 <= score <= 1:
        return min(int(score * 100), 100)
    return max(0, min(int((score + 10) / 20 * 100), 100))


def _build_ctx_html(reranked: list[dict[str, Any]], total_hits: int) -> str:
    stats = f"""
    <div class='stat-grid'>
        <div class='stat-card'>
            <div class='stat-num'>{len(reranked)}</div>
            <div class='stat-lbl'>chunks used</div>
        </div>
        <div class='stat-card'>
            <div class='stat-num'>{total_hits}</div>
            <div class='stat-lbl'>retrieved</div>
        </div>
    </div>
    """

    chunks_html = "<div class='section-title'>Top chunks</div>"
    for item in reranked:
        doc_id = str(item.get("doc_id", "unknown"))
        page = item.get("page") if item.get("page") is not None else "?"
        score = float(item.get("rerank_score", item.get("score", 0.0)))
        pct = _score_percent(score)
        color = "#1d8f6f" if pct >= 70 else "#b7791f" if pct >= 40 else "#c2413b"
        snippet = html.escape((item.get("text") or "")[:160])
        page_type = html.escape(str(item.get("page_type") or "text"))
        sources = ", ".join(item.get("retrieval_sources") or [])
        citation = html.escape(f"[{doc_id} p.{page}]")

        chunks_html += f"""
        <div class='chunk-card'>
            <div class='chunk-header'>
                <span class='chunk-name'>{citation}</span>
                <span class='chunk-score' style='color:{color}'>{score:.2f}</span>
            </div>
            <div class='score-bar'>
                <div class='score-fill' style='width:{pct}%;background:{color}'></div>
            </div>
            <div class='chunk-text'>{snippet}...</div>
            <div class='chunk-meta'>page_type: {page_type} | source: {html.escape(sources)}</div>
        </div>
        """

    return stats + chunks_html


def _format_sources(reranked: list[dict[str, Any]]) -> str:
    if not reranked:
        return ""
    pills = ""
    seen: set[tuple[str, Any]] = set()
    for item in reranked:
        doc_id = str(item.get("doc_id", "unknown"))
        page = item.get("page") if item.get("page") is not None else "?"
        key = (doc_id, page)
        if key in seen:
            continue
        seen.add(key)
        pills += f"<span class='source-pill'>[{html.escape(doc_id)} p.{html.escape(str(page))}]</span>"
    return f"<div style='margin-top:8px'>{pills}</div>"


def get_embed_models() -> dict[str, SentenceTransformer]:
    global _embed_models
    with _runtime_lock:
        if _embed_models is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _embed_models = embed_faiss.load_embedding_models(device=device)
    return _embed_models


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    with _runtime_lock:
        if _reranker is None:
            _reranker = CrossEncoderReranker()
    return _reranker


def load_index() -> tuple[faiss.Index, list[dict[str, Any]], retriever.BM25Bundle]:
    global _index, _meta, _bm25
    with _runtime_lock:
        if _index is None or _meta is None:
            try:
                _index, _meta = retriever.load_faiss_index(INDEX_DIR, allow_empty=True)
            except Exception:
                dim = get_embed_models()["en"].get_sentence_embedding_dimension()
                _index = faiss.IndexFlatIP(dim)
                _meta = []
        if _bm25 is None:
            _bm25 = retriever.build_bm25_index(_meta, PDF_STORE)
    return _index, _meta, _bm25


def refresh_index() -> None:
    global _index, _meta, _bm25
    with _runtime_lock:
        _index, _meta = retriever.load_faiss_index(INDEX_DIR, allow_empty=True)
        _bm25 = retriever.build_bm25_index(_meta, PDF_STORE)


def process_upload(files: list[Any] | None, progress=gr.Progress()):
    if not files:
        return "No PDF selected.", list_documents_md()

    progress(0, desc="Starting intake")
    cfg = SimpleNamespace(store=PDF_STORE, manifest=MANIFEST, sleep=0.1, retries=3)
    handler = PDFHandler(cfg)
    stage = Path(tempfile.mkdtemp(prefix="vika_upload_"))
    failed: list[str] = []

    for file_obj in files:
        try:
            source_path = Path(getattr(file_obj, "name", file_obj))
            staged = stage / source_path.name
            staged.write_bytes(source_path.read_bytes())
            handler.process(staged)
        except Exception as exc:
            failed.append(f"{Path(getattr(file_obj, 'name', file_obj)).name}: {exc}")

    progress(0.35, desc="Chunking")
    new_chunks: list[Path] = []
    for txt_path in sorted(PDF_STORE.glob("*.txt")):
        chunks_path = txt_path.with_suffix(".chunks.jsonl")
        if chunks_path.exists():
            continue
        try:
            count = chunker.process_file(
                txt_path,
                None,
                chunker.DEFAULT_CHUNK_SIZE,
                chunker.DEFAULT_CHUNK_OVERLAP,
                "jsonl",
            )
            if count:
                new_chunks.append(chunks_path)
        except Exception as exc:
            failed.append(f"Chunk {txt_path.name}: {exc}")

    progress(0.7, desc="Embedding")
    added = 0
    if new_chunks:
        try:
            added = embed_faiss.add_chunks_to_index(
                new_chunks,
                INDEX_DIR,
                models=get_embed_models(),
            )
            refresh_index()
        except Exception as exc:
            failed.append(f"Embed: {exc}")

    progress(1.0, desc="Done")
    message = f"Indexed {len(files)} PDF(s); added {added} vector(s)."
    if failed:
        message += "\nWarnings:\n" + "\n".join(failed)
    return message, list_documents_md()


def retrieve_and_rerank(query: str, retrieval_mode: str) -> tuple[retriever.RetrievalResult, float]:
    start = time.time()
    models = get_embed_models()
    query_lang = detect_language(query)
    embed_model = embed_faiss.select_embedding_model(models, query_lang)
    index, metadata, bm25_index = load_index()
    result = retriever.retrieve(
        query=query,
        index=index,
        metadata=metadata,
        chunks_dir=PDF_STORE,
        embed_model=embed_model,
        reranker=get_reranker(),
        retrieval_mode=retrieval_mode,
        bm25_index=bm25_index,
        candidate_k=TOP_CANDIDATES,
        final_k=TOP_CONTEXT_CHUNKS,
    )
    return result, (time.time() - start) * 1000


EVAL_COLUMNS = [
    "Q#",
    "Query",
    "Retrieval (ms)",
    "Generation (ms)",
    "Total (ms)",
    "Chunks retrieved",
    "Chunks used",
    "Reranker mean",
    "Reranker min",
    "Cosine sim",
    "BM25 %",
    "Mode",
]


def _truncate_query(query: str) -> str:
    query = " ".join(query.split())
    return query if len(query) <= 80 else query[:77] + "..."


def _write_report() -> str:
    with _eval_lock:
        REPORT_PATH.write_text(json.dumps(_eval_records, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(REPORT_PATH)


def _eval_table_rows() -> list[list[Any]]:
    with _eval_lock:
        records = list(_eval_records)
    return [
        [
            record["query_index"],
            record["query"],
            round(record["retrieval_latency_ms"], 2),
            round(record["generation_latency_ms"], 2),
            round(record["total_latency_ms"], 2),
            record["chunks_retrieved"],
            record["chunks_used"],
            round(record["reranker_score_mean"], 4),
            round(record["reranker_score_min"], 4),
            round(record["cosine_sim_mean"], 4),
            round(record["bm25_contribution_pct"], 2),
            record["retrieval_mode"],
        ]
        for record in records
    ]


def _eval_summary_values() -> tuple[int, float, float, float]:
    with _eval_lock:
        records = list(_eval_records)
    if not records:
        return 0, 0.0, 0.0, 0.0
    return (
        len(records),
        round(float(np.mean([record["total_latency_ms"] for record in records])), 2),
        round(float(np.mean([record["reranker_score_mean"] for record in records])), 4),
        round(float(np.mean([record["cosine_sim_mean"] for record in records])), 4),
    )


def _eval_outputs():
    total, mean_latency, mean_rerank, mean_cosine = _eval_summary_values()
    report = _write_report()
    return (
        total,
        mean_latency,
        mean_rerank,
        mean_cosine,
        _eval_table_rows(),
        gr.update(value=report),
    )


def _record_evaluation(
    query: str,
    retrieval_result: retriever.RetrievalResult,
    retrieval_latency_ms: float,
    generation_latency_ms: float,
    total_latency_ms: float,
) -> None:
    used = retrieval_result.reranked
    reranker_scores = [float(item.get("rerank_score", item.get("score", 0.0))) for item in used]
    index, _, _ = load_index()
    cosine_mean = retriever.mean_cosine_similarity(index, retrieval_result.query_vector, used)
    bm25_count = sum(1 for item in used if "bm25" in (item.get("retrieval_sources") or []))
    page_types = Counter(str(item.get("page_type") or "text") for item in used)

    with _eval_lock:
        record = {
            "query_index": len(_eval_records) + 1,
            "query": _truncate_query(query),
            "retrieval_latency_ms": float(retrieval_latency_ms),
            "generation_latency_ms": float(generation_latency_ms),
            "total_latency_ms": float(total_latency_ms),
            "chunks_retrieved": len(retrieval_result.candidates),
            "chunks_used": len(used),
            "reranker_score_mean": float(np.mean(reranker_scores)) if reranker_scores else 0.0,
            "reranker_score_min": float(np.min(reranker_scores)) if reranker_scores else 0.0,
            "cosine_sim_mean": float(cosine_mean),
            "bm25_contribution_pct": float((bm25_count / len(used)) * 100) if used else 0.0,
            "page_types_used": dict(page_types),
            "retrieval_mode": retrieval_result.retrieval_mode,
        }
        _eval_records.append(record)


def chat_stream(
    user_message: str,
    chat_history: list[list[str]],
    model_name: str,
    retrieval_mode: str,
    ctx_html: str,
):
    eval_state = _eval_outputs()
    if not user_message or not user_message.strip():
        yield chat_history, "", ctx_html, *eval_state
        return

    total_start = time.time()
    retrieval_result, retrieval_latency_ms = retrieve_and_rerank(user_message, retrieval_mode)
    reranked = retrieval_result.reranked
    ctx_html = _build_ctx_html(reranked, len(retrieval_result.candidates))

    chat_history = chat_history or []
    chat_history.append([user_message, ""])
    prompt = build_prompt(user_message, reranked, history=chat_history[:-1])

    answer = ""
    generation_start = time.time()
    try:
        for token in llm_router.stream_response(prompt, model_name):
            answer += token
            chat_history[-1][1] = answer
            yield chat_history, "", ctx_html, *eval_state
    except Exception as exc:
        answer += f"\n\nRuntime error: {type(exc).__name__}: {exc}"
        chat_history[-1][1] = answer
        yield chat_history, "", ctx_html, *eval_state

    generation_latency_ms = (time.time() - generation_start) * 1000
    total_latency_ms = (time.time() - total_start) * 1000
    if reranked:
        chat_history[-1][1] = answer + _format_sources(reranked)
    _record_evaluation(
        user_message,
        retrieval_result,
        retrieval_latency_ms,
        generation_latency_ms,
        total_latency_ms,
    )
    yield chat_history, "", ctx_html, *_eval_outputs()


with gr.Blocks(
    title="VIKA - RAG Assistant",
    theme=gr.themes.Soft(primary_hue="emerald"),
    css=CSS,
) as demo:
    with gr.Tabs():
        with gr.Tab("Chat"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=190, elem_id="sidebar"):
                    gr.Markdown("## VIKA")
                    docs_md = gr.Markdown(list_documents_md(), elem_id="doc-list")
                    gr.Button("Refresh", size="sm").click(fn=list_documents_md, outputs=docs_md)

                    gr.Markdown("---")
                    with gr.Accordion("Upload PDFs", open=True):
                        upload = gr.File(file_types=[".pdf"], file_count="multiple")
                        ingest_btn = gr.Button("Index", size="sm", variant="primary")
                        ingest_log = gr.Textbox(label="Log", lines=3, interactive=False)
                        ingest_btn.click(
                            fn=process_upload,
                            inputs=[upload],
                            outputs=[ingest_log, docs_md],
                        )

                with gr.Column(scale=5):
                    with gr.Row(elem_id="topbar"):
                        gr.Markdown("### Chat")
                        model_dd = gr.Dropdown(
                            choices=llm_router.MODEL_NAMES,
                            value=llm_router.DEFAULT_MODEL,
                            label="Model",
                            scale=2,
                            container=False,
                        )
                        retrieval_mode_dd = gr.Dropdown(
                            choices=list(retriever.RETRIEVAL_MODES),
                            value=DEFAULT_RETRIEVAL_MODE,
                            label="Retrieval",
                            scale=1,
                            container=False,
                        )

                    chatbot = gr.Chatbot(
                        height=500,
                        elem_id="chatbot",
                        label="",
                        show_label=False,
                        render_markdown=True,
                    )

                    with gr.Row():
                        txt = gr.Textbox(
                            placeholder="Ask anything about your uploaded documents...",
                            scale=5,
                            show_label=False,
                            container=False,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Column(scale=2, min_width=220, elem_id="ctx-panel"):
                    gr.Markdown("### Context")
                    ctx_panel = gr.HTML(_empty_ctx_html())

        with gr.Tab("Session Evaluation"):
            with gr.Row():
                total_queries_box = gr.Number(
                    label="Total queries this session",
                    value=0,
                    precision=0,
                    interactive=False,
                )
                mean_latency_box = gr.Number(
                    label="Mean total latency (ms)",
                    value=0.0,
                    precision=2,
                    interactive=False,
                )
                mean_rerank_box = gr.Number(
                    label="Mean reranker score",
                    value=0.0,
                    precision=4,
                    interactive=False,
                )
                mean_cosine_box = gr.Number(
                    label="Mean cosine similarity",
                    value=0.0,
                    precision=4,
                    interactive=False,
                )

            eval_table = gr.Dataframe(
                headers=EVAL_COLUMNS,
                value=[],
                interactive=False,
                wrap=True,
            )
            report_download = gr.DownloadButton(
                "Download session report",
                value=str(REPORT_PATH),
            )

    submit_inputs = [txt, chatbot, model_dd, retrieval_mode_dd, ctx_panel]
    submit_outputs = [
        chatbot,
        txt,
        ctx_panel,
        total_queries_box,
        mean_latency_box,
        mean_rerank_box,
        mean_cosine_box,
        eval_table,
        report_download,
    ]

    txt.submit(chat_stream, submit_inputs, submit_outputs)
    send_btn.click(chat_stream, submit_inputs, submit_outputs)


threading.Thread(
    target=lambda: (get_embed_models(), get_reranker(), load_index()),
    daemon=True,
).start()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
