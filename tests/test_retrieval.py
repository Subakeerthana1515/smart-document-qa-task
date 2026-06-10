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

def _make_chunk(text: str, doc: str = "test.pdf", page: int = 1, idx: int = 0) -> dict:
    return {"text": text, "metadata": {"document_name": doc, "page": page, "chunk_index": idx}}


def test_add_and_query_roundtrip(isolated_collection):
    chunk = _make_chunk("Paris is the capital of France.")
    vs.add_chunks([chunk], embed_texts([chunk["text"]]))

    results = vs.query_collection(embed_query("capital of France"), n_results=1)
    assert results["documents"][0][0] == "Paris is the capital of France."


def test_document_exists_false_before_insert(isolated_collection):
    assert vs.document_exists("nonexistent.pdf") is False


def test_document_exists_true_after_insert(isolated_collection):
    chunk = _make_chunk("Some text")
    vs.add_chunks([chunk], embed_texts([chunk["text"]]))
    assert vs.document_exists("test.pdf") is True


def test_delete_document_removes_chunks(isolated_collection):
    chunk = _make_chunk("Delete me", doc="remove.pdf")
    vs.add_chunks([chunk], embed_texts([chunk["text"]]))
    assert vs.document_exists("remove.pdf") is True

    vs.delete_document("remove.pdf")
    assert vs.document_exists("remove.pdf") is False


def test_list_documents_returns_all_names(isolated_collection):
    chunks = [
        _make_chunk("Alpha content", doc="alpha.pdf"),
        _make_chunk("Beta content", doc="beta.pdf"),
    ]
    vs.add_chunks(chunks, embed_texts([c["text"] for c in chunks]))

    docs = vs.list_documents()
    names = {d["document_name"] for d in docs}
    assert "alpha.pdf" in names
    assert "beta.pdf" in names


def test_list_documents_chunk_count_accurate(isolated_collection):
    chunks = [
        _make_chunk("Chunk one", doc="counted.pdf", idx=0),
        _make_chunk("Chunk two", doc="counted.pdf", idx=1),
        _make_chunk("Chunk three", doc="counted.pdf", idx=2),
    ]
    vs.add_chunks(chunks, embed_texts([c["text"] for c in chunks]))

    docs = vs.list_documents()
    doc = next(d for d in docs if d["document_name"] == "counted.pdf")
    assert doc["chunk_count"] == 3


# ---------------------------------------------------------------------------
# Retriever tests
# ---------------------------------------------------------------------------

def test_retrieve_returns_relevant_chunk(isolated_collection):
    chunk = _make_chunk("Machine learning is a subset of artificial intelligence.")
    vs.add_chunks([chunk], embed_texts([chunk["text"]]))

    results = retrieve("What is machine learning?", top_k=1)
    assert len(results) == 1
    assert results[0].document_name == "test.pdf"
    assert results[0].page == 1


def test_retrieve_distance_threshold_filters_irrelevant(isolated_collection, monkeypatch):
    """With a very tight threshold, a semantically unrelated query should return nothing."""
    monkeypatch.setattr(config, "SIMILARITY_THRESHOLD", 0.05)

    chunk = _make_chunk("The Eiffel Tower is located in Paris, France.")
    vs.add_chunks([chunk], embed_texts([chunk["text"]]))

    results = retrieve("quantum chromodynamics particle decay", top_k=1)
    assert results == []


def test_retrieve_empty_collection_returns_empty_list(isolated_collection):
    results = retrieve("anything", top_k=5)
    assert results == []


def test_retrieve_result_fields(isolated_collection):
    chunk = _make_chunk("FastAPI is a modern Python web framework.", doc="web.pdf", page=3)
    vs.add_chunks([chunk], embed_texts([chunk["text"]]))

    results = retrieve("Python web framework", top_k=1)
    assert len(results) == 1
    r = results[0]
    assert hasattr(r, "text")
    assert hasattr(r, "document_name")
    assert hasattr(r, "page")
    assert hasattr(r, "distance")
    assert r.document_name == "web.pdf"
    assert r.page == 3
