import logging
from src.pdf_loader import load_pdf
from src.chunker import chunk_pages
from src.embeddings import embed_texts
from src import vector_store
from src.retriever import retrieve
from src.llm import generate_answer

logger = logging.getLogger(__name__)


def ingest_document(file_path: str, document_name: str) -> dict:
    """Full ingestion pipeline: PDF → pages → chunks → embeddings → ChromaDB.

    Returns:
        document_name    (str)
        pages_processed  (int)
        chunks_added     (int)
    """
    pages = load_pdf(file_path)
    chunks = chunk_pages(pages, document_name)

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    vector_store.add_chunks(chunks, embeddings)

    logger.info(
        "Ingested '%s': %d pages, %d chunks.", document_name, len(pages), len(chunks)
    )
    return {
        "document_name": document_name,
        "pages_processed": len(pages),
        "chunks_added": len(chunks),
    }


def answer_question(question: str, top_k: int | None = None) -> dict:
    """Full query pipeline: question → retrieval → LLM → answer with citations.

    Returns:
        answer                 (str)
        sources                (list of {document_name, page}, deduplicated)
        retrieved_chunks_count (int)
    """
    retrieved_chunks = retrieve(question, top_k=top_k)
    answer = generate_answer(question, retrieved_chunks)

    seen: set[tuple] = set()
    sources = []
    for chunk in retrieved_chunks:
        key = (chunk.document_name, chunk.page)
        if key not in seen:
            seen.add(key)
            sources.append({"document_name": chunk.document_name, "page": chunk.page})

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks_count": len(retrieved_chunks),
    }
