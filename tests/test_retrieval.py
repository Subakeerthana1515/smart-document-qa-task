"""Tests for src/embeddings.py, src/vector_store.py, and src/retriever.py.

Each test that touches ChromaDB receives an `isolated_collection` fixture that
replaces the module-level _collection with a fresh in-memory-backed instance
stored in pytest's tmp_path. The real database is never written to.
"""
import chromadb
import pytest

import config
import src.vector_store as vs
from src.embeddings import embed_query, embed_texts
from src.retriever import retrieve

CHAT_A = "session-alpha"
CHAT_B = "session-beta"


# ---------------------------------------------------------------------------
# Fixture — isolated ChromaDB collection per test
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_collection(tmp_path, monkeypatch):
    """Redirect vector_store._collection to a fresh tmp_path-backed collection."""
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection(
        name="test_chunks",
        metadata={"hnsw:space": "cosine"},
    )
    monkeypatch.setattr(vs, "_collection", collection)
    return collection


# ---------------------------------------------------------------------------
# Embeddings tests
# ---------------------------------------------------------------------------

def test_embed_texts_returns_correct_count():
    vectors = embed_texts(["Hello world", "Test sentence"])
    assert len(vectors) == 2


def test_embed_texts_dimensionality():
    vectors = embed_texts(["Hello world"])
    assert len(vectors[0]) == 384


def test_embed_query_dimensionality():
    vector = embed_query("What is the capital of France?")
    assert len(vector) == 384


def test_embed_query_returns_float_list():
    vector = embed_query("test")
    assert all(isinstance(v, float) for v in vector)


# ---------------------------------------------------------------------------
# Vector store tests
# ---------------------------------------------------------------------------

def _make_chunk(
    text: str,
    doc: str = "test.pdf",
    page: int = 1,
    idx: int = 0,
    chat_id: str = CHAT_A,
) -> dict:
    return {
        "text": text,
        "metadata": {
            "chat_id": chat_id,
            "document_name": doc,
            "page": page,
            "chunk_index": idx,
        },
    }


def test_add_and_query_roundtrip(isolated_collection):
    chunk = _make_chunk("Paris is the capital of France.")
    vs.add_chunks([chunk], embed_texts([chunk["text"]]))

    results = vs.query_collection(embed_query("capital of France"), n_results=1, chat_id=CHAT_A)
    assert results["documents"][0][0] == "Paris is the capital of France."


def test_chat_has_no_document_before_insert(isolated_collection):
    assert vs.chat_has_document("nonexistent-session") is False


def test_chat_has_document_after_insert(isolated_collection):
    chunk = _make_chunk("Some text")
    vs.add_chunks([chunk], embed_texts([chunk["text"]]))
    assert vs.chat_has_document(CHAT_A) is True


def test_delete_by_chat_removes_chunks(isolated_collection):
    chunk = _make_chunk("Delete me", chat_id=CHAT_A)
    vs.add_chunks([chunk], embed_texts([chunk["text"]]))
    assert vs.chat_has_document(CHAT_A) is True

    vs.delete_by_chat(CHAT_A)
    assert vs.chat_has_document(CHAT_A) is False


def test_chunks_scoped_to_chat_id(isolated_collection):
    """Chunks from two different sessions never bleed into each other's results."""
    chunk_a = _make_chunk("Alpha document content.", doc="alpha.pdf", chat_id=CHAT_A)
    chunk_b = _make_chunk("Beta document content.", doc="beta.pdf", chat_id=CHAT_B)
    vs.add_chunks([chunk_a, chunk_b], embed_texts([chunk_a["text"], chunk_b["text"]]))

    results_a = vs.query_collection(embed_query("alpha"), n_results=5, chat_id=CHAT_A)
    results_b = vs.query_collection(embed_query("beta"), n_results=5, chat_id=CHAT_B)

    assert all(m["chat_id"] == CHAT_A for m in results_a["metadatas"][0])
    assert all(m["chat_id"] == CHAT_B for m in results_b["metadatas"][0])


# ---------------------------------------------------------------------------
# Retriever tests
# ---------------------------------------------------------------------------

def test_retrieve_returns_relevant_chunk(isolated_collection):
    chunk = _make_chunk("Machine learning is a subset of artificial intelligence.")
    vs.add_chunks([chunk], embed_texts([chunk["text"]]))

    results = retrieve("What is machine learning?", chat_id=CHAT_A, top_k=1)
    assert len(results) == 1
    assert results[0].document_name == "test.pdf"
    assert results[0].page == 1


def test_retrieve_distance_threshold_filters_irrelevant(isolated_collection, monkeypatch):
    """With a very tight threshold, a semantically unrelated query should return nothing."""
    monkeypatch.setattr(config, "SIMILARITY_THRESHOLD", 0.05)

    chunk = _make_chunk("The Eiffel Tower is located in Paris, France.")
    vs.add_chunks([chunk], embed_texts([chunk["text"]]))

    results = retrieve("quantum chromodynamics particle decay", chat_id=CHAT_A, top_k=1)
    assert results == []


def test_retrieve_empty_collection_returns_empty_list(isolated_collection):
    results = retrieve("anything", chat_id="empty-session", top_k=5)
    assert results == []


def test_retrieve_result_fields(isolated_collection):
    chunk = _make_chunk("FastAPI is a modern Python web framework.", doc="web.pdf", page=3)
    vs.add_chunks([chunk], embed_texts([chunk["text"]]))

    results = retrieve("Python web framework", chat_id=CHAT_A, top_k=1)
    assert len(results) == 1
    r = results[0]
    assert hasattr(r, "text")
    assert hasattr(r, "document_name")
    assert hasattr(r, "page")
    assert hasattr(r, "distance")
    assert r.document_name == "web.pdf"
    assert r.page == 3


def test_retrieve_does_not_return_other_chat_chunks(isolated_collection):
    """retrieve() scoped to CHAT_A must never surface a chunk that belongs to CHAT_B."""
    chunk_a = _make_chunk("Machine learning is a subset of AI.", doc="doc_a.pdf", chat_id=CHAT_A)
    chunk_b = _make_chunk("Machine learning is a subset of AI.", doc="doc_b.pdf", chat_id=CHAT_B)
    vs.add_chunks([chunk_a, chunk_b], embed_texts([chunk_a["text"], chunk_b["text"]]))

    results = retrieve("machine learning", chat_id=CHAT_A, top_k=5)
    assert len(results) == 1
    assert results[0].document_name == "doc_a.pdf"
