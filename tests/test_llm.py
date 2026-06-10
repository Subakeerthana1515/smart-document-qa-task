"""Tests for src/llm.py.

google.generativeai is never called for real — _model.generate_content is
patched via monkeypatch on the module-level instance in src.llm.
No API key consumption, no network traffic.
"""
from unittest.mock import MagicMock

import pytest

import src.llm as llm_module
from src.llm import LLMError, generate_answer
from src.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(
    text: str = "Sample document text.",
    document_name: str = "report.pdf",
    page: int = 1,
    distance: float = 0.25,
) -> RetrievedChunk:
    return RetrievedChunk(text=text, document_name=document_name, page=page, distance=distance)


def _mock_generate(monkeypatch, return_text: str = "Mocked answer."):
    """Patch _model.generate_content to return a fixed response."""
    response = MagicMock()
    response.text = return_text
    monkeypatch.setattr(
        llm_module._model, "generate_content", lambda *args, **kwargs: response
    )
    return response


# ---------------------------------------------------------------------------
# No-context behaviour
# ---------------------------------------------------------------------------

def test_no_context_returns_fallback_message():
    result = generate_answer("What is AI?", [])
    assert "not contain sufficient information" in result


def test_no_context_does_not_call_model(monkeypatch):
    called = {"flag": False}

    def should_not_be_called(*args, **kwargs):
        called["flag"] = True
        raise AssertionError("generate_content should not be called with no chunks.")

    monkeypatch.setattr(llm_module._model, "generate_content", should_not_be_called)
    generate_answer("test", [])
    assert not called["flag"]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def test_returns_model_response_text(monkeypatch):
    _mock_generate(monkeypatch, return_text="The answer is 42.")
    result = generate_answer("What is the answer?", [_chunk()])
    assert result == "The answer is 42."


def test_prompt_contains_question(monkeypatch):
    captured = {}

    def fake_generate(prompt, **kwargs):
        captured["prompt"] = prompt
        r = MagicMock()
        r.text = "ok"
        return r

    monkeypatch.setattr(llm_module._model, "generate_content", fake_generate)
    generate_answer("What is the boiling point of water?", [_chunk()])
    assert "What is the boiling point of water?" in captured["prompt"]


def test_prompt_contains_chunk_text(monkeypatch):
    captured = {}

    def fake_generate(prompt, **kwargs):
        captured["prompt"] = prompt
        r = MagicMock()
        r.text = "ok"
        return r

    monkeypatch.setattr(llm_module._model, "generate_content", fake_generate)
    generate_answer("question", [_chunk(text="Unique sentinel text XYZ123.")])
    assert "Unique sentinel text XYZ123." in captured["prompt"]


def test_prompt_contains_source_document_and_page(monkeypatch):
    captured = {}

    def fake_generate(prompt, **kwargs):
        captured["prompt"] = prompt
        r = MagicMock()
        r.text = "ok"
        return r

    monkeypatch.setattr(llm_module._model, "generate_content", fake_generate)
    generate_answer("question", [_chunk(document_name="financials.pdf", page=42)])
    assert "financials.pdf" in captured["prompt"]
    assert "Page 42" in captured["prompt"]


def test_prompt_numbers_multiple_excerpts(monkeypatch):
    captured = {}

    def fake_generate(prompt, **kwargs):
        captured["prompt"] = prompt
        r = MagicMock()
        r.text = "ok"
        return r

    monkeypatch.setattr(llm_module._model, "generate_content", fake_generate)
    chunks = [_chunk(text=f"Excerpt {i}") for i in range(3)]
    generate_answer("question", chunks)

    assert "Excerpt 1" in captured["prompt"]
    assert "Excerpt 2" in captured["prompt"]
    assert "Excerpt 3" in captured["prompt"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_api_error_raises_llm_error(monkeypatch):
    def failing_generate(*args, **kwargs):
        raise RuntimeError("503 Service Unavailable")

    monkeypatch.setattr(llm_module._model, "generate_content", failing_generate)

    with pytest.raises(LLMError, match="503 Service Unavailable"):
        generate_answer("test", [_chunk()])


def test_llm_error_wraps_original_exception(monkeypatch):
    original = ValueError("quota exceeded")

    def failing_generate(*args, **kwargs):
        raise original

    monkeypatch.setattr(llm_module._model, "generate_content", failing_generate)

    with pytest.raises(LLMError) as exc_info:
        generate_answer("test", [_chunk()])

    assert exc_info.value.__cause__ is original
