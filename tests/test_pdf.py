"""Tests for src/pdf_loader.py and src/chunker.py.

PDFs are constructed in-memory with PyMuPDF — no binary test fixtures committed.
"""
import pytest
import fitz

import config
from src.pdf_loader import load_pdf
from src.chunker import chunk_pages

CHAT_ID = "test-session-01"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pdf_path(tmp_path):
    """Two-page PDF with extractable text."""
    doc = fitz.open()

    p1 = doc.new_page(width=612, height=792)
    p1.insert_text((72, 100), "The quick brown fox jumps over the lazy dog. " * 25, fontsize=11)

    p2 = doc.new_page(width=612, height=792)
    p2.insert_text((72, 100), "Python is a versatile programming language. " * 25, fontsize=11)

    path = tmp_path / "sample.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture
def blank_pdf_path(tmp_path):
    """Single blank page — no text inserted."""
    doc = fitz.open()
    doc.new_page()
    path = tmp_path / "blank.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


# ---------------------------------------------------------------------------
# pdf_loader tests
# ---------------------------------------------------------------------------

def test_load_pdf_returns_correct_page_count(sample_pdf_path):
    pages = load_pdf(sample_pdf_path)
    assert len(pages) == 2


def test_load_pdf_page_numbers_are_one_indexed(sample_pdf_path):
    pages = load_pdf(sample_pdf_path)
    assert pages[0]["page_number"] == 1
    assert pages[1]["page_number"] == 2


def test_load_pdf_text_is_non_empty(sample_pdf_path):
    pages = load_pdf(sample_pdf_path)
    for page in pages:
        assert isinstance(page["text"], str)
        assert len(page["text"]) > 0


def test_load_pdf_invalid_path_raises_value_error():
    with pytest.raises(ValueError, match="Cannot open PDF"):
        load_pdf("/nonexistent/path/file.pdf")


def test_load_pdf_blank_pdf_raises_user_friendly_error(blank_pdf_path):
    with pytest.raises(ValueError) as exc_info:
        load_pdf(blank_pdf_path)
    assert "Scanned PDFs are currently not supported" in str(exc_info.value)


# ---------------------------------------------------------------------------
# chunker tests
# ---------------------------------------------------------------------------

def test_chunk_pages_returns_non_empty_list(sample_pdf_path):
    pages = load_pdf(sample_pdf_path)
    chunks = chunk_pages(pages, "sample.pdf", CHAT_ID)
    assert len(chunks) > 0


def test_chunk_pages_metadata_fields_present(sample_pdf_path):
    pages = load_pdf(sample_pdf_path)
    chunks = chunk_pages(pages, "sample.pdf", CHAT_ID)
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        meta = chunk["metadata"]
        assert "chat_id" in meta
        assert "document_name" in meta
        assert "page" in meta
        assert "chunk_index" in meta


def test_chunk_pages_chat_id_propagated(sample_pdf_path):
    pages = load_pdf(sample_pdf_path)
    chunks = chunk_pages(pages, "sample.pdf", CHAT_ID)
    assert all(c["metadata"]["chat_id"] == CHAT_ID for c in chunks)


def test_chunk_pages_document_name_propagated(sample_pdf_path):
    pages = load_pdf(sample_pdf_path)
    chunks = chunk_pages(pages, "my_doc.pdf", CHAT_ID)
    assert all(c["metadata"]["document_name"] == "my_doc.pdf" for c in chunks)


def test_chunk_pages_text_length_within_bounds(sample_pdf_path):
    pages = load_pdf(sample_pdf_path)
    chunks = chunk_pages(pages, "sample.pdf", CHAT_ID)
    # Chunks may slightly exceed CHUNK_SIZE due to word boundaries, but not by much.
    for chunk in chunks:
        assert len(chunk["text"]) <= config.CHUNK_SIZE + config.CHUNK_OVERLAP


def test_chunk_pages_chunk_index_is_sequential_per_page(sample_pdf_path):
    pages = load_pdf(sample_pdf_path)
    chunks = chunk_pages(pages, "sample.pdf", CHAT_ID)
    for page_num in {c["metadata"]["page"] for c in chunks}:
        page_chunks = [c for c in chunks if c["metadata"]["page"] == page_num]
        indices = [c["metadata"]["chunk_index"] for c in page_chunks]
        assert indices == list(range(len(indices)))
