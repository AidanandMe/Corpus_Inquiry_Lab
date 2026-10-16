"""Unit tests for document ingestion, page-level provenance, and corpus isolation."""

import json
from io import BytesIO
from pathlib import Path
import pytest

from src.config import PREPARED_CORPORA_DIR, SESSION_UPLOADS_DIR
from src.ingest import (
    DocumentChunk,
    extract_text_from_pdf,
    extract_text_from_plain,
    ingest_corpus_directory,
    ingest_document,
    list_prepared_corpora,
    purge_session_uploads,
    sanitize_id,
)


def create_minimal_pdf(pages_text: list[str]) -> bytes:
    """Create a minimal valid multi-page PDF in-memory for testing."""
    objs = []
    num_pages = len(pages_text)
    font_id = 3 + 2 * num_pages
    kids = " ".join(f"{3 + 2*i} 0 R" for i in range(num_pages))

    objs.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objs.append(f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {num_pages} >>\nendobj\n".encode("latin1"))

    for i, text in enumerate(pages_text):
        content = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode("latin1")
        stream_len = len(content)
        page_id = 3 + 2 * i
        content_id = 4 + 2 * i
        objs.append(
            f"{page_id} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>\nendobj\n".encode("latin1")
        )
        objs.append(
            f"{content_id} 0 obj\n<< /Length {stream_len} >>\nstream\n".encode("latin1")
            + content
            + b"\nendstream\nendobj\n"
        )

    objs.append(f"{font_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode("latin1"))

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for obj in objs:
        offsets.append(len(out))
        out.extend(obj)

    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode("latin1"))
    for off in offsets:
        out.extend(f"{off:010d} 00000 n \n".encode("latin1"))
    out.extend(f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin1"))
    return bytes(out)


def test_sanitize_id():
    assert sanitize_id("Report on Baetulo (1924).pdf") == "report_on_baetulo_1924_.pdf"
    assert sanitize_id("  Corpus Alpha #1  ") == "corpus_alpha_1"


def test_txt_ingestion(tmp_path: Path):
    txt_file = tmp_path / "survey_notes.txt"
    sample_text = "The Roman forum at Baetulo was excavated between 1924 and 1928.\nSeveral ceramic fragments were identified."
    txt_file.write_text(sample_text, encoding="utf-8")

    chunks, warnings, pages = ingest_document(txt_file)

    assert len(warnings) == 0
    assert pages == 1
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.doc_title == "survey_notes.txt"
    assert chunk.doc_id == "survey_notes"
    assert chunk.page_number == 1
    assert chunk.char_start == 0
    assert chunk.char_end == len(sample_text)
    assert chunk.text == sample_text.strip()


def test_markdown_ingestion(tmp_path: Path):
    md_file = tmp_path / "epigraphy.md"
    md_content = "# Inscription 402\n\nFound near the western gate. Dedication to Apollo."
    md_file.write_text(md_content, encoding="utf-8")

    chunks, warnings, pages = ingest_document(md_file)

    assert len(warnings) == 0
    assert pages == 1
    assert len(chunks) == 1
    assert chunks[0].doc_title == "epigraphy.md"
    assert "Inscription 402" in chunks[0].text


def test_pdf_page_provenance(tmp_path: Path):
    page1_text = "Page One: Initial excavations in Sector Alpha revealed Hellenistic walls."
    page2_text = "Page Two: Sector Beta contained imperial coinage from the reign of Trajan."
    page3_text = "Page Three: The necropolis yielded bronze fibulae and imported amphorae."

    pdf_bytes = create_minimal_pdf([page1_text, page2_text, page3_text])
    pdf_file = tmp_path / "excavation_report.pdf"
    pdf_file.write_bytes(pdf_bytes)

    chunks, warnings, pages = ingest_document(pdf_file)

    assert pages == 3
    assert len(chunks) == 3
    assert len(warnings) == 0

    # Strict page provenance checks
    c1, c2, c3 = chunks
    assert c1.page_number == 1
    assert "Page One" in c1.text
    assert "Page Two" not in c1.text

    assert c2.page_number == 2
    assert "Page Two" in c2.text
    assert "Page One" not in c2.text

    assert c3.page_number == 3
    assert "Page Three" in c3.text
    assert "Page Two" not in c3.text


def test_defensive_scanned_pdf_warning(tmp_path: Path):
    # Create empty PDF pages (simulating scanned raster without OCR)
    pdf_bytes = create_minimal_pdf(["", ""])
    pdf_file = tmp_path / "scanned_archive.pdf"
    pdf_file.write_bytes(pdf_bytes)

    chunks, warnings, pages = ingest_document(pdf_file)

    assert pages == 2
    assert len(warnings) > 0
    assert "scanned image without an OCR text layer" in warnings[0]


def test_chunk_natural_break_and_offsets(tmp_path: Path):
    para1 = "First scholarly observation regarding the ceramic typography in this site."
    para2 = "Second paragraph elaborating on stratigraphic layers and carbon dating."
    full_text = f"{para1}\n\n{para2}"

    txt_file = tmp_path / "stratigraphy.txt"
    txt_file.write_text(full_text, encoding="utf-8")

    # Force small chunk size to trigger multi-chunking with paragraph boundary respect
    chunks, warnings, pages = ingest_document(txt_file, chunk_size=85, chunk_overlap=20)

    assert len(chunks) >= 2
    for chunk in chunks:
        # Verify text slice strictly matches offsets
        expected = full_text[chunk.char_start:chunk.char_end].strip()
        assert chunk.text == expected


def test_corpus_directory_ingestion_with_manifest(tmp_path: Path):
    corpus_dir = tmp_path / "prepared_test_corpus"
    docs_dir = corpus_dir / "docs"
    docs_dir.mkdir(parents=True)

    # Document 1
    (docs_dir / "doc_a.txt").write_text("Text from document A.", encoding="utf-8")
    # Document 2
    pdf_bytes = create_minimal_pdf(["PDF page 1 text", "PDF page 2 text"])
    (docs_dir / "doc_b.pdf").write_bytes(pdf_bytes)

    # Manifest
    manifest_data = {
        "id": "test_corpus_01",
        "name": "Test Archaeological Corpus",
        "description": "Curated collection for testing provenance and isolation."
    }
    (corpus_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    chunks, manifest = ingest_corpus_directory(corpus_dir)

    assert manifest.corpus_id == "test_corpus_01"
    assert manifest.corpus_name == "Test Archaeological Corpus"
    assert manifest.total_pages == 3  # 1 from txt + 2 from pdf
    assert manifest.total_chunks == 3
    assert len(manifest.documents) == 2


def test_list_prepared_corpora_and_purge_uploads(tmp_path: Path, monkeypatch):
    # Set up mock prepared dir
    mock_prepared = tmp_path / "prepared"
    mock_uploads = tmp_path / "session_uploads"
    mock_prepared.mkdir()
    mock_uploads.mkdir()

    monkeypatch.setattr("src.ingest.PREPARED_CORPORA_DIR", mock_prepared)
    monkeypatch.setattr("src.ingest.SESSION_UPLOADS_DIR", mock_uploads)

    # Create 2 prepared corpora
    c1 = mock_prepared / "corpus_alpha"
    c1.mkdir()
    (c1 / "doc1.txt").write_text("Doc 1", encoding="utf-8")

    c2 = mock_prepared / "corpus_beta"
    c2.mkdir()
    (c2 / "manifest.json").write_text(json.dumps({"name": "Custom Beta Name"}), encoding="utf-8")
    (c2 / "doc2.txt").write_text("Doc 2", encoding="utf-8")

    available = list_prepared_corpora()
    assert len(available) == 2
    names = [c["name"] for c in available]
    assert "Corpus Alpha" in names
    assert "Custom Beta Name" in names

    # Test purge session uploads
    temp_file = mock_uploads / "uploaded_research.pdf"
    temp_file.write_text("temporary upload", encoding="utf-8")
    assert temp_file.exists()

    purge_session_uploads()
    assert not temp_file.exists()
    assert mock_uploads.exists()


def test_stem_collision_disambiguation(tmp_path: Path):
    """Ensure two files sharing a stem (e.g., report.txt and report.md) produce distinct doc_ids and chunk_ids."""
    corpus_dir = tmp_path / "collision_corpus"
    corpus_dir.mkdir()
    (corpus_dir / "report.txt").write_text("First report text.", encoding="utf-8")
    (corpus_dir / "report.md").write_text("Second report markdown.", encoding="utf-8")

    chunks, manifest = ingest_corpus_directory(corpus_dir)
    assert len(chunks) == 2
    chunk_ids = {c.chunk_id for c in chunks}
    doc_ids = {c.doc_id for c in chunks}
    assert len(chunk_ids) == 2
    assert len(doc_ids) == 2
