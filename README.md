# Smart Document Q&A Assistant

A production-quality RAG (Retrieval-Augmented Generation) chatbot that lets you upload PDFs and ask questions about them. Answers are grounded strictly in document content and cite the source page.

This project is designed for local execution and evaluation.

## Features

- Chat sessions — one chat, one document
- Per-session retrieval (questions only search the attached document)
- SQLite chat history persistence
- Session management (create / list / view / delete)
- Retrieval-Augmented Generation (RAG)
- Semantic search using embeddings
- ChromaDB persistent vector storage
- Source citations with page numbers
- FastAPI REST backend
- Streamlit frontend
- Automated test suite (52 tests)

## Architecture

```
PDF Upload
  → FastAPI POST /sessions                     (multipart: file + optional title)
  → PyMuPDF                                    (extract text per page)
  → LangChain RecursiveCharacterTextSplitter   (chunk: 800 chars, 150 overlap)
  → SentenceTransformers all-MiniLM-L6-v2      (embed locally, 384-dimensional vectors)
  → ChromaDB PersistentClient                  (store with cosine distance, tagged with chat_id)
  → SQLite                                     (persist session metadata)

Question
  → FastAPI POST /sessions/{id}/ask
  → SentenceTransformers  (embed query)
  → ChromaDB              (top-5 nearest chunks, distance ≤ 1.0, filtered by chat_id)
  → Gemini 2.5 Flash      (temperature 0.2, grounded prompt)
  → SQLite                (persist user + assistant messages)
  → JSON response: { answer, sources: [{document_name, page}], retrieved_chunks_count, question }
```

```
smart-document-qa/
├── src/                  Core pipeline (no API or UI knowledge)
│   ├── pdf_loader.py     PyMuPDF extraction
│   ├── chunker.py        LangChain text splitting
│   ├── embeddings.py     SentenceTransformer singleton
│   ├── vector_store.py   ChromaDB operations (scoped by chat_id)
│   ├── retriever.py      Embed + query + distance filter
│   ├── llm.py            Gemini prompt construction + generation
│   ├── database.py       SQLite session + message CRUD
│   └── rag_pipeline.py   ingest_document() / answer_question()
├── api/
│   └── sessions.py       All session and chat endpoints
├── ui/
│   └── streamlit_app.py  Browser UI (pure HTTP client, no src/ imports)
├── tests/
│   ├── test_pdf.py       pdf_loader + chunker (in-memory PDF fixtures)
│   ├── test_retrieval.py embeddings + vector_store + retriever (isolated ChromaDB)
│   ├── test_llm.py       llm (mocked generate_content, no API calls)
│   └── test_sessions.py  database CRUD (isolated SQLite per test)
├── main.py               FastAPI entry point
├── config.py             All configuration, env-overridable
└── .env.example          Environment variable template
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

**Current test status: 52/52 tests passing.**

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Google AI Studio API key |
| `CHROMA_DB_PATH` | `./database/chroma_db` | ChromaDB storage directory |
| `SQLITE_DB_PATH` | `./database/chat_history.db` | SQLite file for sessions and message history |
| `UPLOAD_DIR` | `./documents/uploaded_pdfs` | Uploaded PDF storage directory |
| `CHUNK_SIZE` | `800` | Max characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `1.0` | Max cosine distance to include a chunk |
| `MAX_FILE_SIZE_MB` | `50` | Max PDF upload size |

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/sessions` | Create a session and ingest its PDF (multipart: `file`, optional `title`) |
| `GET` | `/sessions` | List all sessions, newest first |
| `GET` | `/sessions/{id}` | Get session metadata and full message history |
| `DELETE` | `/sessions/{id}` | Delete session, messages, embeddings, and PDF file |
| `POST` | `/sessions/{id}/ask` | Ask a question scoped to this session's document |
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |

### Example: create a session (upload PDF)

```bash
curl -X POST http://localhost:8000/sessions \
  -F "file=@my_report.pdf" \
  -F "title=My Report"
```

Response:

```json
{
  "id": "3f2e1a...",
  "title": "My Report",
  "document_name": "my_report.pdf",
  "created_at": "2026-06-11T10:00:00+00:00",
  "pages_processed": 5,
  "chunks_added": 23
}
```

### Example: ask a question

```bash
curl -X POST http://localhost:8000/sessions/3f2e1a.../ask \
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

- **One chat = one document**: Each session is scoped to a single PDF. Questions only retrieve chunks from that session's document, preventing answer contamination across documents.
- **Scanned PDFs** (image-only, no embedded text layer) are not supported. The API returns a clear error message.
