# 🧠 VIKA – Vision Knowledge Assistant

**VIKA** is a lightweight, local-first Retrieval-Augmented Generation (RAG) system that transforms your PDF documents into a searchable knowledge base and provides accurate, source-cited answers via a local LLM.

> ⚡️ No cloud dependencies – everything runs locally on CPU or GPU  
> 🧾 Ideal for lecture notes, research papers, or any scientific documents

---

## 🚀 Features

- 📂 Upload PDF documents via a web UI
- 🔍 Perform semantic search using FAISS
- 🧠 Rerank results with a local cross-encoder
- 🤖 Query a quantized Mistral-7B-Instruct LLM
- 🧾 Source citations shown in answers
- 🔁 Incremental indexing (no reprocessing needed)
- 🧑‍🏫 Optimized for student tutoring and educational QA

---

## 📦 Project Structure

```bash
.
├── app.py              # Gradio interface with sidebar, chat, and upload
├── chunker.py          # Text splitter (LangChain text splitter)
├── document_intake.py  # Deduplicates and extracts text from PDFs
├── embed_faiss.py      # Embeds chunks into FAISS index
├── retriever.py        # Vector search over FAISS index
├── reranker.py         # Reranks search results with a cross-encoder
├── prompt_builder.py   # Builds the final prompt with citations + chat history
├── generate.py         # Runs the LLM locally and saves response
├── pipeline.py         # Orchestrates the pipeline manually
├── parser_utils.py     # OCR/text extraction utilities (PaddleOCR, PyMuPDF) 
└── data/
    └── pdfs/           # Canonical store of processed PDF files and metadata
```

---

## 🛠️ Requirements

```bash
pip install -r requirements.txt
```

### Key Dependencies
- `sentence-transformers`
- `faiss-cpu`
- `torch`
- `langchain`
- `llama-cpp-python`
- `gradio`
- `pdf2image`, `fitz`, `paddleocr`
- `huggingface_hub`

---

## 🖥️ Running the App

```bash
python app.py
```

Then open http://localhost:7860 in your browser.

---

## 🧾 How It Works

1. **Upload PDFs**  
   Deduplicated by SHA-256, then text is extracted using PyMuPDF or PaddleOCR fallback.

2. **Chunking**  
   The text is split into overlapping segments using a recursive splitter.

3. **Embedding**  
   Chunks are embedded using `all-MiniLM-L6-v2` and added to a FAISS index.

4. **Retrieval + Reranking**  
   The system retrieves the top-k chunks via FAISS and reranks them using `cross-encoder/ms-marco-MiniLM-L-6-v2`.

5. **Prompt Building**  
   Results are formatted using a Jinja2-based prompt builder, with human-readable source citations.

6. **Generation**  
   A quantized Mistral-7B-Instruct model is used to generate the final answer locally.

---

## 🧪 Testing Manually

### Full Index Build
```bash
python embed_faiss.py ./data/pdfs --out ./data/index --full
```

### Retrieval + Reranking + Prompt
```bash
python retriever.py --query "What is a CNN?" --k 20 > hits.json
python reranker.py --query "What is a CNN?" --input hits.json --top 5 > top_hits.json
python prompt_builder.py --query "What is a CNN?" --hits top_hits.json > prompt.txt
```

### Generation (CLI)
```bash
python generate.py
```

---

## 📌 Notes

- Files are stored by SHA-256 to avoid duplication.
- Original filenames are preserved in `manifest.csv` and shown in the UI.
- Supports both CPU and GPU execution.
- Gracefully handles empty indexes or missing model files.

---

## 📚 Example Use Case

> Upload your course PDFs, ask "What is gradient descent?" and receive a cited, accurate answer directly grounded in your own material.

---

## 👨‍🔧 Future Improvements

- [ ] Add support for extracting/captioning images from PDFs
- [ ] Extend language support beyond English
- [ ] Add notebook support (.ipynb → markdown chunks)

---

## 🧠 Credits

Created with ❤️ for local-first, citation-focused document QA.  
System name: **VIKA – Vision Knowledge Assistant**
