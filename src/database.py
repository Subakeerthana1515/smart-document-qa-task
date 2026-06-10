import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

import config


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create tables if they do not exist. Safe to call on every startup."""
    db_dir = os.path.dirname(config.SQLITE_DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id            TEXT PRIMARY KEY,
                title         TEXT NOT NULL,
                document_name TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id   TEXT    NOT NULL REFERENCES chat_sessions(id),
                role      TEXT    NOT NULL,
                content   TEXT    NOT NULL,
                sources   TEXT,
                timestamp TEXT    NOT NULL
            );
        """)


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

def create_session(document_name: str, title: str | None = None) -> dict:
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    title = title or document_name
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_sessions (id, title, document_name, created_at) VALUES (?, ?, ?, ?)",
            (session_id, title, document_name, now),
        )
    return {
        "id": session_id,
        "title": title,
        "document_name": document_name,
        "created_at": now,
    }


def list_sessions() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, document_name, created_at "
            "FROM chat_sessions ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_session(session_id: str) -> dict | None:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id, title, document_name, created_at "
            "FROM chat_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def delete_session(session_id: str) -> None:
    """Delete the session row and all its messages."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (session_id,))
        conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))


# ---------------------------------------------------------------------------
# Message CRUD
# ---------------------------------------------------------------------------

def add_message(
    chat_id: str,
    role: str,
    content: str,
    sources: list | None = None,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    sources_json = json.dumps(sources) if sources is not None else None
    with _get_conn() as conn:
        cursor = conn.execute(
            "INSERT INTO messages (chat_id, role, content, sources, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (chat_id, role, content, sources_json, now),
        )
        msg_id = cursor.lastrowid
    return {
        "id": msg_id,
        "chat_id": chat_id,
        "role": role,
        "content": content,
        "sources": sources,
        "timestamp": now,
    }


def get_messages(chat_id: str) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, chat_id, role, content, sources, timestamp "
            "FROM messages WHERE chat_id = ? ORDER BY id ASC",
            (chat_id,),
        ).fetchall()
    result = []
    for row in rows:
        msg = dict(row)
        msg["sources"] = json.loads(msg["sources"]) if msg["sources"] else None
        result.append(msg)
    return result
