import os

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Smart Document Q&A",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, sources}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fetch_documents() -> list[dict]:
    try:
        resp = httpx.get(f"{API_BASE_URL}/documents/list", timeout=10)
        resp.raise_for_status()
        return resp.json().get("documents", [])
    except Exception:
        return []


def api_upload(filename: str, data: bytes) -> dict:
    resp = httpx.post(
        f"{API_BASE_URL}/documents/upload",
        files={"file": (filename, data, "application/pdf")},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def api_delete(document_name: str) -> None:
    resp = httpx.delete(f"{API_BASE_URL}/documents/{document_name}", timeout=10)
    resp.raise_for_status()


def api_ask(question: str) -> dict:
    resp = httpx.post(
        f"{API_BASE_URL}/chat/ask",
        json={"question": question},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def render_sources(sources: list[dict]) -> None:
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for src in sources:
                st.caption(f"📄 **{src['document_name']}** — Page {src['page']}")


# ---------------------------------------------------------------------------
# Sidebar — document management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 Documents")
    st.divider()

    # --- Upload ---
    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF", type=["pdf"], label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if st.button("Upload & Index", type="primary", use_container_width=True):
            with st.spinner("Indexing…"):
                try:
                    result = api_upload(uploaded_file.name, uploaded_file.getvalue())
                    if result.get("status") == "already_exists":
                        st.warning(
                            f"**{uploaded_file.name}** is already indexed.\n\n"
                            "Delete it first to re-ingest."
                        )
                    else:
                        st.success(
                            f"**{result['document_name']}** indexed — "
                            f"{result['pages_processed']} pages, "
                            f"{result['chunks_added']} chunks."
                        )
                        st.rerun()
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.json().get("detail", str(exc))
                    st.error(f"Upload failed: {detail}")
                except httpx.TimeoutException:
                    st.error("Upload timed out. Try a smaller PDF.")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

    st.divider()

    # --- Document list ---
    st.subheader("Indexed Documents")
    docs = fetch_documents()

    if not docs:
        st.caption("No documents indexed yet.")
    else:
        for doc in docs:
            col_name, col_del = st.columns([4, 1])
            with col_name:
                st.markdown(f"**{doc['document_name']}**")
                st.caption(f"{doc['chunk_count']} chunks")
            with col_del:
                if st.button(
                    "🗑",
                    key=f"del_{doc['document_name']}",
                    help=f"Delete {doc['document_name']}",
                ):
                    try:
                        api_delete(doc["document_name"])
                        st.success(f"Deleted **{doc['document_name']}**.")
                        st.rerun()
                    except httpx.HTTPStatusError as exc:
                        detail = exc.response.json().get("detail", str(exc))
                        st.error(f"Delete failed: {detail}")
                    except Exception as exc:
                        st.error(f"Unexpected error: {exc}")

    st.divider()

    # --- Clear chat ---
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main area — chat
# ---------------------------------------------------------------------------
st.title("Smart Document Q&A")
st.caption(
    "Ask questions about your uploaded PDFs. "
    "Answers are grounded strictly in document content."
)

# Render existing message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))

# New question
if question := st.chat_input("Ask a question about your documents…"):
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    # Fetch and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                data = api_ask(question)
                answer = data["answer"]
                sources = data.get("sources", [])

                st.markdown(answer)
                render_sources(sources)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status == 503:
                    msg = "The AI service is temporarily unavailable. Please try again shortly."
                elif status == 400:
                    msg = exc.response.json().get("detail", "Bad request.")
                else:
                    msg = f"API error {status}: {exc.response.json().get('detail', str(exc))}"
                st.error(msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": msg, "sources": []}
                )

            except httpx.TimeoutException:
                msg = "Request timed out. Please try again."
                st.error(msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": msg, "sources": []}
                )

            except Exception as exc:
                msg = f"Connection error — is the API server running? ({exc})"
                st.error(msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": msg, "sources": []}
                )
