import os
from dotenv import load_dotenv

load_dotenv()

# --- Secrets ---
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# --- Storage paths (env-overridable) ---
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./database/chroma_db")
SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "./database/chat_history.db")
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./documents/uploaded_pdfs")

# --- ChromaDB ---
COLLECTION_NAME: str = "document_chunks"

# --- Chunking ---
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Retrieval ---
TOP_K: int = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "1.0"))

# --- Upload ---
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# --- Models ---
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
GEMINI_MODEL: str = "gemini-2.5-flash"


def validate_config() -> None:
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to your .env file or environment."
        )
