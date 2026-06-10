# Smart Document Q&A Assistant

A production-quality RAG (Retrieval-Augmented Generation) chatbot that lets you upload PDFs and ask questions about them. Answers are grounded strictly in document content and cite the source page.

This project is designed for local execution and evaluation.

## Features

- Upload and index PDF documents
- Retrieval-Augmented Generation (RAG)
- Semantic search using embeddings
- ChromaDB persistent vector storage
- Source citations with page numbers
- Document management (upload / list / delete)
- FastAPI REST backend
- Streamlit frontend
- Automated test suite (33 tests)

## Architecture

```
PDF Upload
  → FastAPI /documents/upload
  → PyMuPDF   (extract text per page)
  → LangChain RecursiveCharacterTextSplitter  (chunk: 800 chars, 150 overlap)
  → SentenceTransformers all-MiniLM-L6-v2     (embed locally, 384-dimensional vectors)
  → ChromaDB PersistentClient                 (store with cosine distance)

Question
  → FastAPI /chat/ask
  → SentenceTransformers  (embed query)
  → ChromaDB              (top-5 nearest chunks, distance ≤ 0.7)
  → Gemini 2.5 Flash      (temperature 0.2, grounded prompt)
  → JSON response: { answer, sources: [{document_name, page}] }
```

```
smart-document-qa/
├── src/                  Core pipeline (no API or UI knowledge)
│   ├── pdf_loader.py     PyMuPDF extraction
│   ├── chunker.py        LangChain text splitting
│   ├── embeddings.py     SentenceTransformer singleton
│   ├── vector_store.py   All ChromaDB operations
│   ├── retriever.py      Embed + query + distance filter
│   ├── llm.py            Gemini prompt construction + generation
│   └── rag_pipeline.py   ingest_document() / answer_question()
├── api/
│   ├── upload.py         POST /documents/upload, GET /documents/list, DELETE /documents/{name}
│   └── chat.py           POST /chat/ask
├── ui/
│   └── streamlit_app.py  Browser UI (pure HTTP client, no src/ imports)
├── tests/
│   ├── test_pdf.py       pdf_loader + chunker (in-memory PDF fixtures)
│   ├── test_retrieval.py embeddings + vector_store + retriever (isolated ChromaDB)
│   └── test_llm.py       llm (mocked generate_content, no API calls)
├── main.py               FastAPI entry point
└── config.py             All configuration, env-overridable
```

## Local Setup

### Prerequisites

- Python 3.11+
- A [Gemini API key](https://aistudio.google.com/app/apikey)

### 1. Clone and install

```bash
git clone <your-repo-url>
cd smart-document-qa
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env   # or edit .env directly
```

Set your key in `.env`:

```
GEMINI_API_KEY=your_key_here
```

### 3. Run the API

```bash
uvicorn main:app --reload
```

API is available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### 4. Run the UI (separate terminal)

```bash
streamlit run ui/streamlit_app.py
```

UI is available at `http://localhost:8501`.

### 5. Run tests

```bash
pytest tests/ -v
```

No API key is consumed during tests. The embedding model (`all-MiniLM-L6-v2`) is downloaded once on first run and cached locally (~90 MB).

**Current test status: 33/33 tests passing.**

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Google AI Studio API key |
| `CHROMA_DB_PATH` | `./database/chroma_db` | ChromaDB storage directory |
| `UPLOAD_DIR` | `./documents/uploaded_pdfs` | Uploaded PDF storage directory |
| `CHUNK_SIZE` | `800` | Max characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.7` | Max cosine distance to include a chunk |
| `MAX_FILE_SIZE_MB` | `50` | Max PDF upload size |

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/documents/upload` | Upload and index a PDF (multipart/form-data, field: `file`) |
| `GET` | `/documents/list` | List all indexed documents and chunk counts |
| `DELETE` | `/documents/{document_name}` | Remove a document from the index |
| `POST` | `/chat/ask` | Ask a question (`{"question": "...", "top_k": 5}`) |
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |

### Example: upload a PDF

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@my_report.pdf"
```

### Example: ask a question

```bash
curl -X POST http://localhost:8000/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key financial risks?"}'
```

Response:

```json
{
  "answer": "According to page 12, the key financial risks include...",
  "sources": [
    {"document_name": "my_report.pdf", "page": 12},
    {"document_name": "my_report.pdf", "page": 15}
  ],
  "retrieved_chunks_count": 4,
  "question": "What are the key financial risks?"
}
```

## Notes

- **Scanned PDFs** (image-only, no embedded text layer) are not supported. The API returns a clear error message.
- **Multi-document**: All indexed documents are queried together. The answer cites which document and page each excerpt came from.
