import logging
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> list[dict]:
    """Extract text from each page of a PDF.

    Returns a list of dicts with keys:
        page_number (int, 1-indexed)
        text        (str)

    Skips pages with no extractable text (e.g. scanned images) with a warning.
    Raises ValueError for unreadable or password-protected files.
    """
    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{file_path}': {exc}") from exc

    if doc.needs_pass:
        doc.close()
        raise ValueError(f"PDF is password-protected: '{file_path}'")

    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text").strip()
        if not text:
            logger.warning("Page %d has no extractable text — skipping.", page_num + 1)
            continue
        pages.append({"page_number": page_num + 1, "text": text})

    doc.close()

    if not pages:
        raise ValueError(
            "PDF contains no extractable text. Scanned PDFs are currently not supported."
        )

    return pages
