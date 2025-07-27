# app.py – Gradio interface for VIKA RAG assistant (sidebar edition)
# -------------------------------------------------------------------
"""Run with:
    python app.py

Left‑hand sidebar now shows the **original filenames** (from `manifest.csv`) instead
of SHA‑256 hashes.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import gradio as gr
import torch
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import faiss  # type: ignore

# ── Project modules ---------------------------------------------------------
from document_intake import PDFHandler, sha256
import chunker
import embed_faiss
import retriever
from reranker import CrossEncoderReranker
from prompt_builder import build_prompt

# ── Paths & constants -------------------------------------------------------
ROOT = Path(__file__).parent
PDF_STORE = ROOT / "data" / "pdfs"
INDEX_DIR = ROOT / "data" / "index"
MANIFEST = ROOT / "data" / "manifest.csv"
MODEL_PATH = ROOT / "models" / "mistral-7b-instruct-v0.2.Q5_K_M.gguf"

PDF_STORE.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################
# Utility – build sidebar markdown (original filenames)
###############################################################################

def _load_manifest_mapping() -> dict[str, str]:
    """Return {hash: original_filename}.  Ignores rows that miss either."""
    mapping: dict[str, str] = {}
    if not MANIFEST.exists():
        return mapping
    with MANIFEST.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            h = (row.get("hash") or row.get("sha256") or "").strip()
            name = (row.get("original") or row.get("filename") or "").strip()
            if h and name:
                mapping[h] = name
    return mapping


def list_documents_md() -> str:
    pdfs = sorted(PDF_STORE.glob("*.pdf"))
    if not pdfs:
        return "*(no documents indexed yet)*"

    mapping = _load_manifest_mapping()
    items = [
        f"- **{mapping.get(p.stem, '(unknown‑name)')}**"
        for p in pdfs
    ]
    return "### 🗂️ Indexed documents\n" + "\n".join(items)


###############################################################################
# Wrapper around embed_faiss for incremental adds
###############################################################################

def add_chunks_to_index(chunks_paths: List[Path], embed_model: SentenceTransformer):
    return embed_faiss.add_chunks_to_index(chunks_paths, INDEX_DIR, embed_model)

###############################################################################
# Lazy singletons (thread‑safe)
###############################################################################

_lock = threading.Lock()
_llm: Llama | None = None
_embed: SentenceTransformer | None = None
_index: faiss.Index | None = None
_meta: List[Dict] | None = None
_reranker: CrossEncoderReranker | None = None


def get_embed():
    global _embed
    with _lock:
        if _embed is None:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            _embed = SentenceTransformer("all-MiniLM-L6-v2", device=dev)
        return _embed


def get_reranker():
    global _reranker
    with _lock:
        if _reranker is None:
            _reranker = CrossEncoderReranker()
        return _reranker


def load_index():
    global _index, _meta
    with _lock:
        if _index is None or _meta is None:
            try:
                _index, _meta = retriever.load_faiss_index(INDEX_DIR, allow_empty=True)
            except Exception:
                dim = get_embed().get_sentence_embedding_dimension()
                _index, _meta = faiss.IndexFlatIP(dim), []
        return _index, _meta


def refresh_index():
    global _index, _meta
    with _lock:
        _index, _meta = retriever.load_faiss_index(INDEX_DIR, allow_empty=True)


def get_llm():
    global _llm
    with _lock:
        if _llm is None:
            gpu_layers = 40 if torch.cuda.is_available() else 0
            _llm = Llama(
                model_path=str(MODEL_PATH),
                n_ctx=4096,
                n_gpu_layers=gpu_layers,
                n_threads=max(1, os.cpu_count() - 1),
            )
        return _llm

###############################################################################
# PDF upload → intake → chunk → embed
###############################################################################

def process_upload(files: List[Path], progress: gr.Progress = gr.Progress()):
    progress(0, desc="Starting intake…")

    cfg = SimpleNamespace(store=PDF_STORE, manifest=MANIFEST, sleep=0.1, retries=3)
    handler = PDFHandler(cfg)

    stage = Path(tempfile.mkdtemp())
    for f in files:
        tmp = stage / Path(f.name).name
        tmp.write_bytes(Path(f).read_bytes())
        handler.process(tmp)
    progress(0.3, desc="Text extracted")

    # chunk new txts
    new_chunks = []
    for txt in PDF_STORE.glob("*.txt"):
        chunks_path = txt.with_suffix(".chunks.jsonl")
        if not chunks_path.exists():
            chunker.process_file(txt, None, 1000, 200, "jsonl")
            new_chunks.append(chunks_path)
    progress(0.6, desc="Chunked")

    # embed
    added_vecs = 0
    if new_chunks:
        added_vecs = add_chunks_to_index(new_chunks, get_embed())
        if added_vecs:
            refresh_index()
    progress(1.0, desc="Index updated ✅")

    return (
        f"Added {len(files)} PDF(s) → {added_vecs} new vectors.",
        list_documents_md(),
    )

###############################################################################
# Chat generator
###############################################################################

def token_stream(query: str, history: Sequence[Tuple[str, str]]):
    embed = get_embed()
    index, meta = load_index()

    q_vec = retriever.embed_queries(embed, [query])
    hits = retriever.search(index, meta, q_vec, top_k=20)
    hits = [(s, m) for (s, m) in hits if s > 0.4]

    passages = []
    for score, m in hits:
        text = retriever.load_chunk_text(PDF_STORE, m["doc_id"], m["chunk_id"])
        passages.append({"score": score, **m, "text": text or ""})
    if not passages:
        reranked = []
    else:
        reranked = get_reranker().rerank(query, passages)[:5]

    prompt = build_prompt(query, reranked, history=history)

    llm = get_llm()
    for chunk in llm.create_completion(prompt, temperature=0.7, top_p=0.95, max_tokens=800, stream=True):
        yield chunk["choices"][0]["text"]


def chat_stream(user_message: str, chat_hist: List[Tuple[str, str]]):
    chat_hist = chat_hist or []
    chat_hist.append((user_message, ""))
    answer = ""
    for tok in token_stream(user_message, chat_hist[:-1]):
        answer += tok
        chat_hist[-1] = (user_message, answer)
        yield chat_hist, ""  # second output clears textbox

###############################################################################
# Gradio UI
###############################################################################

with gr.Blocks(title="VIKA – PDF RAG Assistant") as demo:
    with gr.Row():
        # Sidebar column ----------------------------------------------------
        with gr.Column(scale=1, min_width=220):
            docs_md = gr.Markdown(list_documents_md(), label="Documents")
            refresh_btn = gr.Button("🔄 Refresh list")
            refresh_btn.click(fn=lambda: list_documents_md(), outputs=docs_md)

        # Main column -------------------------------------------------------
        with gr.Column(scale=4):
            gr.Markdown(
                """# 📚 VIKA – PDF RAG Assistant\nUpload PDFs → build knowledge base → chat and get cited answers."""
            )

            with gr.Accordion("📂 Upload & index", open=not any(PDF_STORE.glob("*.pdf"))):
                upload = gr.File(file_types=[".pdf"], file_count="multiple", label="Choose PDF(s)")
                ingest_btn = gr.Button("Upload & Index")
                ingest_log = gr.Textbox(label="Log", interactive=False)
                ingest_btn.click(
                    fn=process_upload,
                    inputs=[upload],
                    outputs=[ingest_log, docs_md],
                )

            chatbot = gr.Chatbot(height=450, label="Chat")
            txt = gr.Textbox(scale=4, placeholder="Ask anything… (SHIFT+ENTER for newline)")
            txt.submit(chat_stream, [txt, chatbot], [chatbot, txt])

    # warm‑up in background to avoid UI freeze on first interaction
    threading.Thread(target=lambda: (get_embed(), get_reranker(), get_llm(), load_index()), daemon=True).start()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
