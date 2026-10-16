"""Configuration and path definitions for Corpus Inquiry Lab."""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
CORPORA_DIR = BASE_DIR / "corpora"
PREPARED_CORPORA_DIR = CORPORA_DIR / "prepared"
SESSION_UPLOADS_DIR = CORPORA_DIR / "session_uploads"
STORAGE_DIR = BASE_DIR / "storage"

# Ingestion settings
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
MIN_CHARS_PER_PAGE_WARNING = 50

def ensure_directories() -> None:
    """Ensure all required project directories exist."""
    PREPARED_CORPORA_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
