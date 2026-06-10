import os
import re
import logging
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

import config
from src import database, vector_store
from src.llm import LLMError
from src.rag_pipeline import ingest_document, answer_question

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["sessions"])


def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name)
    return re.sub(r"[^\w\-.]", "_", name)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SessionSummary(BaseModel):
    id: str
    title: str
    document_name: str
    created_at: str


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class SessionCreateResponse(BaseModel):
    id: str
    title: str
    document_name: str
    created_at: str
    pages_processed: int
    chunks_added: int


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    sources: list[dict] | None = None
    timestamp: str


class SessionDetailResponse(BaseModel):
    id: str
    title: str
    document_name: str
    created_at: str
    messages: list[MessageOut]


class QuestionRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceCitation(BaseModel):
    document_name: str
    page: int


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    retrieved_chunks_count: int
    question: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=SessionCreateResponse, status_code=201)
async def create_session(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
):
    """Create a chat session and ingest its PDF in one step.

    Rolls back the session row if ingestion fails so the DB stays consistent.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"File must have a .pdf extension. Got: '{file.filename}'.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if not content.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400, detail="File does not appear to be a valid PDF."
        )

    max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {config.MAX_FILE_SIZE_MB} MB limit.",
        )

    document_name = _sanitize_filename(file.filename)

    # Create the session row first — we need session_id before we can write the file
    # and before chunk IDs are minted.
    session = database.create_session(document_name=document_name, title=title)
    session_id = session["id"]

    # Each session gets its own subdirectory so the same filename can appear in
    # multiple chats without collision.
    session_dir = os.path.join(config.UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    file_path = os.path.join(session_dir, document_name)

    with open(file_path, "wb") as fh:
        fh.write(content)

    try:
        result = ingest_document(file_path, document_name, session_id)
    except ValueError as exc:
        _rollback_session(session_id, file_path)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _rollback_session(session_id, file_path)
        logger.error("Ingestion failed for session %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return SessionCreateResponse(
        id=session_id,
        title=session["title"],
        document_name=document_name,
        created_at=session["created_at"],
        pages_processed=result["pages_processed"],
        chunks_added=result["chunks_added"],
    )


@router.get("", response_model=SessionListResponse)
def list_sessions():
    """Return all chat sessions ordered newest first."""
    sessions = database.list_sessions()
    return SessionListResponse(sessions=[SessionSummary(**s) for s in sessions])


@router.get("/{session_id}", response_model=SessionDetailResponse)
def get_session(session_id: str):
    """Return a session's metadata and its full message history."""
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    messages = database.get_messages(session_id)
    return SessionDetailResponse(
        **session,
        messages=[MessageOut(**m) for m in messages],
    )


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: str):
    """Delete a session: removes messages, ChromaDB chunks, and the PDF file."""
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Order matters: ChromaDB first, then SQLite, then filesystem.
    vector_store.delete_by_chat(session_id)
    database.delete_session(session_id)

    session_dir = os.path.join(config.UPLOAD_DIR, session_id)
    pdf_path = os.path.join(session_dir, session["document_name"])
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    if os.path.isdir(session_dir) and not os.listdir(session_dir):
        os.rmdir(session_dir)


@router.post("/{session_id}/ask", response_model=AskResponse)
def ask(session_id: str, request: QuestionRequest):
    """Answer a question using only the document attached to this session."""
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    top_k = request.top_k or config.TOP_K

    database.add_message(session_id, role="user", content=request.question)

    try:
        result = answer_question(request.question, chat_id=session_id, top_k=top_k)
    except LLMError as exc:
        logger.error("LLM error in session %s: %s", session_id, exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected error in session %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")

    database.add_message(
        session_id,
        role="assistant",
        content=result["answer"],
        sources=result["sources"],
    )

    return AskResponse(
        answer=result["answer"],
        sources=[SourceCitation(**s) for s in result["sources"]],
        retrieved_chunks_count=result["retrieved_chunks_count"],
        question=request.question,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rollback_session(session_id: str, file_path: str) -> None:
    """Remove an orphaned session row and its partially-written PDF."""
    database.delete_session(session_id)
    if os.path.exists(file_path):
        os.remove(file_path)
