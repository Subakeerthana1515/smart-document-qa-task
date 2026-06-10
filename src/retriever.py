from dataclasses import dataclass
import config
from src.embeddings import embed_query
from src.vector_store import query_collection


@dataclass
class RetrievedChunk:
    text: str
    document_name: str
    page: int
    distance: float


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Embed the query, search ChromaDB, and return chunks below the distance threshold.

    Lower cosine distance = more similar. Chunks with distance > SIMILARITY_THRESHOLD
    are discarded here before being passed to the LLM.
    """
    k = top_k if top_k is not None else config.TOP_K
    query_embedding = embed_query(query)
    results = query_collection(query_embedding, n_results=k)

    if not results["ids"] or not results["ids"][0]:
        return []

    chunks = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if distance <= config.SIMILARITY_THRESHOLD:
            chunks.append(RetrievedChunk(
                text=doc,
                document_name=meta["document_name"],
                page=meta["page"],
                distance=distance,
            ))

    return chunks
