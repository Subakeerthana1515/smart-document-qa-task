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


def _make_id(chat_id: str, document_name: str, page: int, chunk_index: int) -> str:
    safe_chat = re.sub(r"[^a-zA-Z0-9_-]", "_", chat_id)
    safe_doc = re.sub(r"[^a-zA-Z0-9_-]", "_", document_name)
    return f"{safe_chat}_{safe_doc}_{page}_{chunk_index}"


def add_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """Persist a batch of chunks and their embeddings to the collection."""
    ids = [
        _make_id(
            c["metadata"]["chat_id"],
            c["metadata"]["document_name"],
            c["metadata"]["page"],
            c["metadata"]["chunk_index"],
        )
        for c in chunks
    ]
    _collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def query_collection(
    query_embedding: list[float],
    n_results: int,
    chat_id: str,
) -> dict:
    """Return the top-n closest chunks scoped to a specific chat session.

    Guards against the ChromaDB error that fires when the filtered set is empty
    (zero chunks exist for this chat_id yet).
    """
    check = _collection.get(where={"chat_id": chat_id}, limit=1)
    if not check["ids"]:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    return _collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"chat_id": chat_id},
        include=["documents", "metadatas", "distances"],
    )


def chat_has_document(chat_id: str) -> bool:
    """Return True if any chunks for this chat session are already stored."""
    result = _collection.get(where={"chat_id": chat_id}, limit=1)
    return len(result["ids"]) > 0


def delete_by_chat(chat_id: str) -> None:
    """Remove all chunks belonging to a chat session."""
    _collection.delete(where={"chat_id": chat_id})
