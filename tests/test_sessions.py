"""Tests for src/database.py.

Each test receives an isolated SQLite database via the `db` fixture,
which monkeypatches config.SQLITE_DB_PATH to a fresh per-test file.
No shared state between tests; the real database is never written to.
"""
import pytest

import config
import src.database as database


# ---------------------------------------------------------------------------
# Fixture — isolated SQLite database per test
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point all DB operations at a fresh per-test SQLite file."""
    monkeypatch.setattr(config, "SQLITE_DB_PATH", str(tmp_path / "test.db"))
    database.init_db()


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

def test_init_db_is_idempotent(db):
    """Calling init_db a second time must not raise (CREATE TABLE IF NOT EXISTS)."""
    database.init_db()


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------

def test_create_session_returns_all_fields(db):
    s = database.create_session("report.pdf")
    assert s["id"]
    assert s["title"] == "report.pdf"
    assert s["document_name"] == "report.pdf"
    assert s["created_at"]


def test_create_session_default_title_is_document_name(db):
    s = database.create_session("notes.pdf")
    assert s["title"] == "notes.pdf"


def test_create_session_custom_title_overrides_default(db):
    s = database.create_session("notes.pdf", title="Q3 Meeting Notes")
    assert s["title"] == "Q3 Meeting Notes"
    assert s["document_name"] == "notes.pdf"


def test_create_session_ids_are_unique(db):
    s1 = database.create_session("doc.pdf")
    s2 = database.create_session("doc.pdf")
    assert s1["id"] != s2["id"]


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------

def test_list_sessions_empty_initially(db):
    assert database.list_sessions() == []


def test_list_sessions_returns_all_sessions(db):
    database.create_session("alpha.pdf")
    database.create_session("beta.pdf")
    sessions = database.list_sessions()
    assert len(sessions) == 2
    names = {s["document_name"] for s in sessions}
    assert "alpha.pdf" in names
    assert "beta.pdf" in names


# ---------------------------------------------------------------------------
# get_session
# ---------------------------------------------------------------------------

def test_get_session_returns_none_for_unknown_id(db):
    assert database.get_session("nonexistent-uuid") is None


def test_get_session_returns_correct_data(db):
    created = database.create_session("slides.pdf", title="Q3 Slides")
    fetched = database.get_session(created["id"])
    assert fetched is not None
    assert fetched["id"] == created["id"]
    assert fetched["title"] == "Q3 Slides"
    assert fetched["document_name"] == "slides.pdf"


# ---------------------------------------------------------------------------
# delete_session
# ---------------------------------------------------------------------------

def test_delete_session_removes_the_row(db):
    s = database.create_session("temp.pdf")
    database.delete_session(s["id"])
    assert database.get_session(s["id"]) is None


def test_delete_session_also_removes_its_messages(db):
    s = database.create_session("temp.pdf")
    database.add_message(s["id"], role="user", content="Hello")
    database.delete_session(s["id"])
    assert database.get_messages(s["id"]) == []


def test_delete_nonexistent_session_does_not_raise(db):
    database.delete_session("ghost-id")  # must not raise


# ---------------------------------------------------------------------------
# add_message
# ---------------------------------------------------------------------------

def test_add_message_returns_expected_fields(db):
    s = database.create_session("doc.pdf")
    msg = database.add_message(s["id"], role="user", content="What is this about?")
    assert msg["chat_id"] == s["id"]
    assert msg["role"] == "user"
    assert msg["content"] == "What is this about?"
    assert msg["sources"] is None
    assert msg["timestamp"]


def test_add_message_with_sources_round_trips(db):
    s = database.create_session("doc.pdf")
    sources = [{"document_name": "doc.pdf", "page": 3}]
    msg = database.add_message(s["id"], role="assistant", content="Answer.", sources=sources)
    assert msg["sources"] == sources


# ---------------------------------------------------------------------------
# get_messages
# ---------------------------------------------------------------------------

def test_get_messages_returns_empty_for_new_session(db):
    s = database.create_session("empty.pdf")
    assert database.get_messages(s["id"]) == []


def test_get_messages_returns_in_insertion_order(db):
    s = database.create_session("doc.pdf")
    database.add_message(s["id"], role="user", content="First")
    database.add_message(s["id"], role="assistant", content="Second")
    database.add_message(s["id"], role="user", content="Third")
    msgs = database.get_messages(s["id"])
    assert len(msgs) == 3
    assert msgs[0]["content"] == "First"
    assert msgs[1]["content"] == "Second"
    assert msgs[2]["content"] == "Third"


def test_get_messages_sources_deserialised_from_json(db):
    s = database.create_session("doc.pdf")
    sources = [{"document_name": "doc.pdf", "page": 7}, {"document_name": "doc.pdf", "page": 12}]
    database.add_message(s["id"], role="assistant", content="Found it.", sources=sources)
    msgs = database.get_messages(s["id"])
    assert msgs[0]["sources"] == sources


def test_get_messages_null_sources_returned_as_none(db):
    s = database.create_session("doc.pdf")
    database.add_message(s["id"], role="user", content="A question.")
    msgs = database.get_messages(s["id"])
    assert msgs[0]["sources"] is None
