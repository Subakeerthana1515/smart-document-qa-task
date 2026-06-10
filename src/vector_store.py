import os
import re
import chromadb
import config

# Ensure the storage directory exists before ChromaDB tries to open it.
os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)

_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
_collection = _client.get_or_create_collection(
    name=config.COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def _make_id(document_name: str, page: int, chunk_index: int) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", document_name)
    return f"{safe}_{page}_{chunk_index}"


def add_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """Persist a batch of chunks and their embeddings to the collection."""
    ids = [
        _make_id(c["metadata"]["document_name"], c["metadata"]["page"], c["metadata"]["chunk_index"])
        for c in chunks
    ]
    _collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def query_collection(query_embedding: list[float], n_results: int) -> dict:
    """Return the top-n closest chunks to the query embedding."""
    return _collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )


def document_exists(document_name: str) -> bool:
    """Return True if any chunks for this document are already stored."""
    result = _collection.get(where={"document_name": document_name}, limit=1)
    return len(result["ids"]) > 0


def delete_document(document_name: str) -> None:
    """Remove all chunks belonging to a document."""
    _collection.delete(where={"document_name": document_name})


def list_documents() -> list[dict]:
    """Return each unique document name and its chunk count."""
    result = _collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in result["metadatas"]:
        name = meta["document_name"]
        counts[name] = counts.get(name, 0) + 1
    return [{"document_name": name, "chunk_count": count} for name, count in counts.items()]
