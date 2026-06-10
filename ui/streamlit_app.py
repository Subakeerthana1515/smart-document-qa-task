"""Streamlit UI for Smart Document Q&A Assistant.

Pure HTTP client — no direct imports from src/ or api/.
All data comes from the FastAPI backend at API_BASE_URL.
"""
import os

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
_TIMEOUT = 120.0   # generous ceiling for Gemini calls

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Smart Document Q&A",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# API helpers — all return (result, error_string) so callers decide display
# ---------------------------------------------------------------------------

def _list_sessions() -> list[dict]:
    try:
        r = httpx.get(f"{API_BASE_URL}/sessions", timeout=10)
        r.raise_for_status()
        return r.json().get("sessions", [])
    except Exception:
        return []


def _load_messages(session_id: str) -> list[dict]:
    try:
        r = httpx.get(f"{API_BASE_URL}/sessions/{session_id}", timeout=10)
        r.raise_for_status()
        return r.json().get("messages", [])
    except Exception:
        return []


def _create_session(
    file_bytes: bytes, filename: str, title: str | None
) -> tuple[dict | None, str | None]:
    try:
        r = httpx.post(
            f"{API_BASE_URL}/sessions",
            files={"file": (filename, file_bytes, "application/pdf")},
            data={"title": title} if title else {},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json(), None
    except httpx.HTTPStatusError as e:
        return None, e.response.json().get("detail", str(e))
    except Exception as e:
        return None, str(e)


def _delete_session(session_id: str) -> tuple[bool, str | None]:
    try:
        r = httpx.delete(f"{API_BASE_URL}/sessions/{session_id}", timeout=15)
        r.raise_for_status()
        return True, None
    except httpx.HTTPStatusError as e:
        return False, e.response.json().get("detail", str(e))
    except Exception as e:
        return False, str(e)


def _ask(session_id: str, question: str) -> tuple[dict | None, str | None]:
    try:
        r = httpx.post(
            f"{API_BASE_URL}/sessions/{session_id}/ask",
            json={"question": question},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json(), None
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        detail = e.response.json().get("detail", str(e))
        if status == 503:
            return None, "The AI service is temporarily unavailable. Please try again."
        return None, detail
    except httpx.TimeoutException:
        return None, "Request timed out. Please try again."
    except Exception as e:
        return None, f"Connection error — is the API running? ({e})"


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.caption(f"📄 **{s['document_name']}** — Page {s['page']}")


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "sessions" not in st.session_state:
    st.session_state.sessions: list[dict] = []          # [{id, title, document_name, created_at}]

if "active_id" not in st.session_state:
    st.session_state.active_id: str | None = None       # UUID of the open chat

if "messages" not in st.session_state:
    st.session_state.messages: dict[str, list] = {}     # session_id -> [{role, content, sources}]

if "show_upload" not in st.session_state:
    st.session_state.show_upload: bool = False          # whether the new-chat panel is visible

if "loaded" not in st.session_state:
    st.session_state.loaded: bool = False               # sessions fetched at least once

# ---------------------------------------------------------------------------
# Bootstrap — load session list once per page load
# ---------------------------------------------------------------------------

if not st.session_state.loaded:
    st.session_state.sessions = _list_sessions()
    st.session_state.loaded = True

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📄 Smart Doc Q&A")
    st.divider()

    # ── New Chat button ────────────────────────────────────────────────────
    if st.button("＋ New Chat", use_container_width=True, type="primary"):
        st.session_state.show_upload = not st.session_state.show_upload

    # ── Upload panel (toggled by the button above) ─────────────────────────
    if st.session_state.show_upload:
        uploaded = st.file_uploader(
            "Choose a PDF",
            type=["pdf"],
            label_visibility="collapsed",
            key="new_chat_uploader",
        )
        custom_title = st.text_input(
            "Title (optional)",
            placeholder="Defaults to filename",
            key="new_chat_title",
        )

        if uploaded:
            if st.button("Create Chat", use_container_width=True):
                with st.spinner("Uploading and indexing…"):
                    session, err = _create_session(
                        file_bytes=uploaded.getvalue(),
                        filename=uploaded.name,
                        title=custom_title.strip() or None,
                    )
                if err:
                    st.error(f"Upload failed: {err}")
                else:
                    summary = {k: session[k] for k in ("id", "title", "document_name", "created_at")}
                    st.session_state.sessions.insert(0, summary)
                    st.session_state.active_id = session["id"]
                    st.session_state.messages[session["id"]] = []
                    st.session_state.show_upload = False
                    st.rerun()

        st.divider()

    # ── Chat list ──────────────────────────────────────────────────────────
    if st.session_state.sessions:
        for s in st.session_state.sessions:
            is_active = s["id"] == st.session_state.active_id
            col_btn, col_del = st.columns([5, 1])

            with col_btn:
                label = ("▶ " if is_active else "") + s["title"]
                if st.button(label, key=f"sel_{s['id']}", use_container_width=True):
                    if not is_active:
                        st.session_state.active_id = s["id"]
                        # Lazy-load history on first visit to this session
                        if s["id"] not in st.session_state.messages:
                            raw = _load_messages(s["id"])
                            st.session_state.messages[s["id"]] = [
                                {"role": m["role"], "content": m["content"], "sources": m["sources"]}
                                for m in raw
                            ]
                        st.rerun()

            with col_del:
                if st.button("🗑", key=f"del_{s['id']}", help="Delete this chat"):
                    with st.spinner("Deleting…"):
                        ok, err = _delete_session(s["id"])
                    if ok:
                        st.session_state.sessions = [x for x in st.session_state.sessions if x["id"] != s["id"]]
                        st.session_state.messages.pop(s["id"], None)
                        if st.session_state.active_id == s["id"]:
                            st.session_state.active_id = None
                        st.rerun()
                    else:
                        st.error(f"Delete failed: {err}")
    else:
        st.info("No chats yet. Click **＋ New Chat** to begin.")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

# ── No session selected ────────────────────────────────────────────────────
if st.session_state.active_id is None:
    st.title("Smart Document Q&A")
    st.markdown(
        "Welcome! Click **＋ New Chat** in the sidebar to upload a PDF and start chatting.\n\n"
        "**Each chat is scoped to one document** — questions only search the PDF "
        "attached to the current chat, so answers are always precise and traceable."
    )
    st.stop()

active_id = st.session_state.active_id

# Guard: session may have been deleted from the sidebar while it was active
active_session = next((s for s in st.session_state.sessions if s["id"] == active_id), None)
if not active_session:
    st.session_state.active_id = None
    st.rerun()

# Lazy-load messages if the cache entry is missing (e.g. after a hard refresh)
if active_id not in st.session_state.messages:
    raw = _load_messages(active_id)
    st.session_state.messages[active_id] = [
        {"role": m["role"], "content": m["content"], "sources": m["sources"]}
        for m in raw
    ]

# ── Header ─────────────────────────────────────────────────────────────────
st.title(active_session["title"])
st.caption(f"📄 {active_session['document_name']}")
st.divider()

# ── Message history ────────────────────────────────────────────────────────
for msg in st.session_state.messages[active_id]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            _render_sources(msg["sources"])

# ── Chat input ─────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about this document…"):
    # Immediately show the user turn
    st.session_state.messages[active_id].append(
        {"role": "user", "content": question, "sources": None}
    )
    with st.chat_message("user"):
        st.markdown(question)

    # Fetch and show the assistant turn
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result, err = _ask(active_id, question)

        if err:
            # Remove the pending user message so history stays consistent
            st.session_state.messages[active_id].pop()
            st.error(err)
        else:
            st.markdown(result["answer"])
            if result.get("sources"):
                _render_sources(result["sources"])
            st.session_state.messages[active_id].append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })
