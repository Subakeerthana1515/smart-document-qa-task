from sentence_transformers import SentenceTransformer
import config

# Loaded once at import time — reused across all calls.
_model = SentenceTransformer(config.EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings. Returns list of 384-dim float vectors."""
    return _model.encode(texts, show_progress_bar=False).tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string. Returns a 384-dim float vector."""
    return _model.encode([query], show_progress_bar=False)[0].tolist()
