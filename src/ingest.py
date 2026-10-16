"""Document ingestion, defensive text extraction, and provenance tracking for Corpus Inquiry Lab."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from src.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MIN_CHARS_PER_PAGE_WARNING,
    PREPARED_CORPORA_DIR,
    SESSION_UPLOADS_DIR,
    SUPPORTED_EXTENSIONS,
)


@dataclass
class DocumentChunk:
    """A single chunk of text with immutable provenance coordinates."""
    chunk_id: str
    doc_id: str
    doc_title: str
    page_number: int  # 1-indexed physical page number
    char_start: int   # Character start offset relative to the page
    char_end: int     # Character end offset relative to the page
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CorpusManifest:
    """Metadata describing a loaded research corpus and its evidentiary scope."""
    corpus_id: str
    corpus_name: str
    description: str = ""
    documents: list[dict[str, Any]] = field(default_factory=list)
    total_pages: int = 0
    total_chunks: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sanitize_id(name: str) -> str:
    """Convert filename or title into a clean, safe identifier."""
    sanitized = re.sub(r"[^\w\-_.]", "_", name.strip())
    return re.sub(r"_+", "_", sanitized).strip("_").lower()


def extract_text_from_plain(file_path: Path) -> list[tuple[int, str]]:
    """Extract text from .txt or .md files.

    Returns a list with a single entry: [(1, content)].
    Attempts UTF-8 first, then falls back to Latin-1.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = file_path.read_text(encoding="latin-1")
    return [(1, content)]


def extract_text_from_pdf(file_path: Path) -> tuple[list[tuple[int, str]], list[str]]:
    """Extract text page-by-page from a PDF file.

    Returns:
        tuple of (pages, warnings) where pages is [(page_number, text), ...] (1-indexed).
    """
    pages: list[tuple[int, str]] = []
    warnings: list[str] = []

    try:
        reader = PdfReader(str(file_path))
        num_pages = len(reader.pages)
        total_extracted_chars = 0

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            extracted = page.extract_text() or ""
            pages.append((page_num, extracted))
            total_extracted_chars += len(extracted.strip())

        # Defensive check for scanned PDFs or empty text layers
        if num_pages > 0 and total_extracted_chars < (MIN_CHARS_PER_PAGE_WARNING * num_pages):
            warnings.append(
                f"Defensive Notice: '{file_path.name}' yielded very low text density "
                f"({total_extracted_chars} characters across {num_pages} pages). "
                "This document may be a scanned image without an OCR text layer."
            )

    except Exception as exc:
        warnings.append(f"Error reading PDF '{file_path.name}': {exc}")

    return pages, warnings


def chunk_page_text(
    text: str,
    page_number: int,
    doc_id: str,
    doc_title: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    chunk_counter_start: int = 0,
) -> list[DocumentChunk]:
    """Split page text into overlapping chunks, anchored to exact page coordinates.

    Chunks never cross page boundaries to preserve strict page-level provenance.
    Splits prefer paragraph and sentence breaks when feasible.
    """
    chunks: list[DocumentChunk] = []
    clean_text = text.replace("\r\n", "\n")
    total_len = len(clean_text)

    if total_len == 0:
        return chunks

    if total_len <= chunk_size:
        chunk_id = f"{doc_id}_p{page_number}_c{chunk_counter_start}"
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_title=doc_title,
                page_number=page_number,
                char_start=0,
                char_end=total_len,
                text=clean_text.strip(),
            )
        )
        return chunks

    start = 0
    chunk_idx = chunk_counter_start

    while start < total_len:
        end = min(start + chunk_size, total_len)

        # If not at the end of the text, look for a natural break point (paragraph or sentence)
        if end < total_len:
            # Look for double newline first (paragraph break)
            para_break = clean_text.rfind("\n\n", start + chunk_overlap, end)
            if para_break != -1:
                end = para_break + 2
            else:
                # Look for sentence end
                sentence_break = -1
                for punct in [". ", "? ", "! ", ".\n"]:
                    pos = clean_text.rfind(punct, start + chunk_overlap, end)
                    if pos > sentence_break:
                        sentence_break = pos + len(punct)
                if sentence_break != -1:
                    end = sentence_break
                else:
                    # Fallback to single newline or space
                    space_break = clean_text.rfind(" ", start + chunk_overlap, end)
                    if space_break != -1:
                        end = space_break + 1

        chunk_text_slice = clean_text[start:end].strip()
        if chunk_text_slice:
            chunk_id = f"{doc_id}_p{page_number}_c{chunk_idx}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    page_number=page_number,
                    char_start=start,
                    char_end=end,
                    text=chunk_text_slice,
                )
            )
            chunk_idx += 1

        if end >= total_len:
            break

        # Advance with overlap
        start = max(start + 1, end - chunk_overlap)

    return chunks


def extract_text_from_docx(file_path: Path) -> list[tuple[int, str]]:
    """Extract text from Microsoft Word (.docx) documents, including paragraphs and tables."""
    try:
        import docx
        doc = docx.Document(str(file_path))
        blocks: list[str] = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    blocks.append(" | ".join(row_cells))
        full_text = "\n\n".join(blocks)
        return [(1, full_text)]
    except Exception:
        return [(1, "")]


def ingest_document(
    file_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    doc_id: str | None = None,
) -> tuple[list[DocumentChunk], list[str], int]:
    """Ingest a single document, extracting text page-by-page and chunking it.

    Returns:
        (chunks, warnings, total_pages)
    """
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return [], [f"Unsupported file format '{suffix}' for file '{file_path.name}'."], 0

    doc_title = file_path.name
    resolved_doc_id = doc_id or sanitize_id(file_path.stem)
    all_chunks: list[DocumentChunk] = []
    warnings: list[str] = []

    if suffix == ".pdf":
        pages, pdf_warnings = extract_text_from_pdf(file_path)
        warnings.extend(pdf_warnings)
    elif suffix == ".docx":
        pages = extract_text_from_docx(file_path)
        if not any(text.strip() for _, text in pages):
            warnings.append(
                f"Defensive Notice: '{file_path.name}' yielded no extractable text."
            )
    else:
        pages = extract_text_from_plain(file_path)

    chunk_counter = 0
    for page_number, page_text in pages:
        page_chunks = chunk_page_text(
            text=page_text,
            page_number=page_number,
            doc_id=resolved_doc_id,
            doc_title=doc_title,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_counter_start=chunk_counter,
        )
        all_chunks.extend(page_chunks)
        chunk_counter += len(page_chunks)

    return all_chunks, warnings, len(pages)


def ingest_corpus_directory(
    corpus_dir: Path,
    corpus_id: str | None = None,
    corpus_name: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[list[DocumentChunk], CorpusManifest]:
    """Ingest an entire directory of documents into an isolated corpus.

    Discovers all supported files, parses them, and builds a CorpusManifest.
    """
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    cid = corpus_id or sanitize_id(corpus_dir.name)
    cname = corpus_name or corpus_dir.name.replace("_", " ").title()

    manifest_file = corpus_dir / "manifest.json"
    description = ""
    if manifest_file.exists():
        try:
            meta = json.loads(manifest_file.read_text(encoding="utf-8"))
            cid = meta.get("id", cid)
            cname = meta.get("name", cname)
            description = meta.get("description", "")
        except Exception:
            pass

    all_chunks: list[DocumentChunk] = []
    all_warnings: list[str] = []
    doc_summaries: list[dict[str, Any]] = []
    total_pages = 0

    # Locate documents in the directory or docs/ subfolder
    docs_dir = corpus_dir / "docs" if (corpus_dir / "docs").is_dir() else corpus_dir
    files = sorted(
        [p for p in docs_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )

    # Detect stem collisions so files like report.pdf and report.txt get distinct doc_ids
    stem_counts: dict[str, int] = {}
    for file_path in files:
        base_id = sanitize_id(file_path.stem)
        stem_counts[base_id] = stem_counts.get(base_id, 0) + 1

    used_doc_ids: set[str] = set()
    for file_path in files:
        base_id = sanitize_id(file_path.stem)
        if stem_counts[base_id] > 1 or base_id in used_doc_ids:
            ext_tag = sanitize_id(file_path.suffix.lstrip("."))
            resolved_doc_id = f"{base_id}_{ext_tag}" if ext_tag else f"{base_id}_{len(used_doc_ids)}"
        else:
            resolved_doc_id = base_id
        used_doc_ids.add(resolved_doc_id)

        chunks, doc_warnings, pages_count = ingest_document(
            file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            doc_id=resolved_doc_id,
        )
        all_chunks.extend(chunks)
        all_warnings.extend(doc_warnings)
        total_pages += pages_count
        doc_summaries.append(
            {
                "file_name": file_path.name,
                "doc_id": resolved_doc_id,
                "pages": pages_count,
                "chunks": len(chunks),
                "char_count": sum(len(c.text) for c in chunks),
            }
        )

    manifest = CorpusManifest(
        corpus_id=cid,
        corpus_name=cname,
        description=description,
        documents=doc_summaries,
        total_pages=total_pages,
        total_chunks=len(all_chunks),
        warnings=all_warnings,
    )

    return all_chunks, manifest


def list_prepared_corpora() -> list[dict[str, Any]]:
    """Scan corpora/prepared/ dynamically and return list of available prepared corpora."""
    PREPARED_CORPORA_DIR.mkdir(parents=True, exist_ok=True)
    corpora: list[dict[str, Any]] = []

    for item in sorted(PREPARED_CORPORA_DIR.iterdir()):
        if item.is_dir():
            manifest_file = item / "manifest.json"
            cid = sanitize_id(item.name)
            cname = item.name.replace("_", " ").title()
            description = ""
            if manifest_file.exists():
                try:
                    meta = json.loads(manifest_file.read_text(encoding="utf-8"))
                    cid = meta.get("id", cid)
                    cname = meta.get("name", cname)
                    description = meta.get("description", "")
                except Exception:
                    pass

            # Count supported files
            docs_dir = item / "docs" if (item / "docs").is_dir() else item
            file_count = sum(
                1 for p in docs_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )

            corpora.append(
                {
                    "id": cid,
                    "name": cname,
                    "path": str(item),
                    "description": description,
                    "doc_count": file_count,
                }
            )

    return corpora


def purge_session_uploads() -> None:
    """Safely wipe participant-uploaded temporary files for a clean reset on Windows/OneDrive.

    Deletes all files and subdirectories inside SESSION_UPLOADS_DIR rather than
    deleting the root directory itself, preventing Windows [WinError 5] Access is denied.
    """
    if not SESSION_UPLOADS_DIR.exists():
        SESSION_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        return

    for item in SESSION_UPLOADS_DIR.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
        except Exception:
            # Handle potential temporary locks gracefully
            pass
