import os
import re
import logging
from fastapi import APIRouter, File, HTTPException, UploadFile

import config
from src import vector_store
from src.rag_pipeline import ingest_document

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r"[^\w\-.]", "_", name)
    return name


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Ingest a PDF into the vector store.

    - Validates extension and PDF magic bytes.
    - Enforces MAX_FILE_SIZE_MB.
    - Idempotent: returns early if the document is already indexed.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"File must have a .pdf extension. Got: '{file.filename}'.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Magic-byte check — more reliable than MIME type alone.
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

    if vector_store.document_exists(document_name):
        return {
            "status": "already_exists",
            "document_name": document_name,
            "message": "Document already indexed. Delete it first to re-ingest.",
        }

    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(config.UPLOAD_DIR, document_name)
    with open(file_path, "wb") as fh:
        fh.write(content)

    try:
        result = ingest_document(file_path, document_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Ingestion failed for '%s': %s", document_name, exc)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return {
        "status": "success",
        "document_name": result["document_name"],
        "pages_processed": result["pages_processed"],
        "chunks_added": result["chunks_added"],
        "already_existed": False,
    }


@router.get("/list")
def list_documents():
    """Return all ingested documents and their chunk counts."""
    return {"documents": vector_store.list_documents()}


@router.delete("/{document_name}")
def delete_document(document_name: str):
    """Remove a document and all its chunks from the vector store."""
    if not vector_store.document_exists(document_name):
        raise HTTPException(
            status_code=404, detail=f"Document '{document_name}' not found."
        )

    vector_store.delete_document(document_name)

    file_path = os.path.join(config.UPLOAD_DIR, document_name)
    if os.path.exists(file_path):
        os.remove(file_path)

    return {"status": "deleted", "document_name": document_name}
