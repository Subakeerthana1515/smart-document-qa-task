from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def chunk_pages(pages: list[dict], document_name: str, chat_id: str) -> list[dict]:
    """Split page texts into overlapping chunks with source metadata.

    Returns a list of dicts with keys:
        text     (str)
        metadata (dict: chat_id, document_name, page, chunk_index)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for idx, text in enumerate(page_chunks):
            chunks.append({
                "text": text,
                "metadata": {
                    "chat_id": chat_id,
                    "document_name": document_name,
                    "page": page["page_number"],
                    "chunk_index": idx,
                },
            })

    return chunks
