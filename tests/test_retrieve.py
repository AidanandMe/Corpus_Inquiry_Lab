"""Unit tests for Phase 2: Dual Retrieval, Per-Corpus Vector Storage, and Methodological Isolation."""

import shutil
from pathlib import Path
import pytest

from src.config import STORAGE_DIR
from src.ingest import DocumentChunk
from src.lexical import (
    DOCUMENTED_STOPWORDS,
    extract_deterministic_terms,
    scan_corpus_lexical,
)
from src.retrieve import execute_dual_retrieval
from src.store import CorpusVectorStore, SemanticResult


def create_sample_chunk(
    chunk_id: str,
    doc_id: str,
    doc_title: str,
    page_number: int,
    text: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_title=doc_title,
        page_number=page_number,
        char_start=0,
        char_end=len(text),
        text=text,
    )


# ---------------------------------------------------------------------------
# Lexical Scanner Tests
# ---------------------------------------------------------------------------

def test_deterministic_term_extraction():
    # Quoted phrase + content words + stop words
    query = 'What was found at "Baetulo forum" during the 1924 excavation of ceramics?'
    terms = extract_deterministic_terms(query)

    # "Baetulo forum" should be preserved as an exact phrase
    assert "Baetulo forum" in terms

    # Stop words like 'what', 'was', 'at', 'during', 'the', 'of' must be excluded
    for stop in ["what", "was", "at", "during", "the", "of"]:
        assert stop not in terms

    # Content words and 4-digit year should be present
    lower_terms = [t.lower() for t in terms]
    assert "ceramics" in lower_terms
    assert "excavation" in lower_terms
    assert "found" in lower_terms
    assert "1924" in lower_terms

    # Explicit terms override
    explicit = ["Emporion", "coinage"]
    override_terms = extract_deterministic_terms(query, explicit_terms=explicit)
    assert override_terms == ["Emporion", "coinage"]


def test_lexical_punctuation_bounded_terms():
    chunk = create_sample_chunk(
        "c1", "doc_alpha", "doc_alpha.txt", 1,
        "Field work led by Dr. E. Martinez uncovered [Inscription 101] in Sector Alpha."
    )
    report = scan_corpus_lexical(
        [chunk],
        query='Check "Dr. E. Martinez" and "[Inscription 101]"',
    )
    assert report.term_results["Dr. E. Martinez"].present_in_corpus is True
    assert report.term_results["[Inscription 101]"].present_in_corpus is True


def test_lexical_presence_and_miss():
    chunk1 = create_sample_chunk(
        "c1", "doc_alpha", "doc_alpha.txt", 1,
        "Excavations at Baetulo revealed extensive Roman domestic architecture and amphorae."
    )
    chunk2 = create_sample_chunk(
        "c2", "doc_beta", "doc_beta.txt", 2,
        "Coinage from the reign of Trajan was catalogued alongside ceramic shards."
    )
    chunks = [chunk1, chunk2]

    # Query with one term present ("Baetulo") and one term absent ("Emporion")
    query = 'Were there coins in "Baetulo" and what about Emporion?'
    report = scan_corpus_lexical(chunks, query)

    # 1. Present term checks
    assert "Baetulo" in report.term_results
    baetulo_res = report.term_results["Baetulo"]
    assert baetulo_res.present_in_corpus is True
    assert baetulo_res.total_occurrences == 1
    assert baetulo_res.documents_count == 1
    assert "doc_alpha" in [m.doc_id for m in baetulo_res.matches]
    assert "found 1 time(s)" in baetulo_res.epistemic_statement

    # 2. Absent term checks (strictly framed as lexical miss, not conceptual absence)
    assert "Emporion" in report.term_results
    emporion_res = report.term_results["Emporion"]
    assert emporion_res.present_in_corpus is False
    assert emporion_res.total_occurrences == 0
    assert len(emporion_res.matches) == 0
    assert (
        "The searched term or phrase 'Emporion' was not found in the corpus using this lexical check."
        == emporion_res.epistemic_statement
    )

    # 3. Methodological notice presence
    assert "not that the concept is absent" in report.methodological_notice


# ---------------------------------------------------------------------------
# Vector Store and Semantic Retrieval Tests
# ---------------------------------------------------------------------------

def test_vector_store_indexing_and_similarity_metrics(tmp_path: Path, monkeypatch):
    test_storage = tmp_path / "storage"
    monkeypatch.setattr("src.store.STORAGE_DIR", test_storage)

    store = CorpusVectorStore("test_corpus_metrics")

    c1 = create_sample_chunk(
        "c1", "ceramics_doc", "ceramics_doc.txt", 1,
        "Stratified ceramic typologies and terra sigillata fragments from the western forum."
    )
    c2 = create_sample_chunk(
        "c2", "metallurgy_doc", "metallurgy_doc.txt", 1,
        "Bronze smelting furnaces and medieval iron slag discovered outside the city wall."
    )

    meta = store.index_chunks([c1, c2])
    assert meta["chunks_indexed"] == 2
    assert meta["embedding_model"] == "all-MiniLM-L6-v2 (ONNX)"
    assert meta["embedding_dimension"] == 384
    assert meta["distance_metric"] == "cosine"
    assert store.is_indexed() is True

    # Query for pottery/ceramics
    results = store.search("pottery and Roman ceramics", top_k=2)
    assert len(results) == 2

    top_result = results[0]
    assert top_result.chunk.doc_id == "ceramics_doc"
    assert top_result.rank == 1

    # Verify score labeling: similarity is mathematical metric, not confidence
    assert 0.0 <= top_result.similarity_score <= 1.0
    assert top_result.distance >= 0.0
    metric_dict = top_result.to_dict()
    assert "mathematical vector proximity metric" in metric_dict["metric_note"]
    assert "confidence" not in metric_dict["metric_type"]


def test_strict_corpus_isolation(tmp_path: Path, monkeypatch):
    """Verify that retrieval against Corpus A CANNOT return chunks from Corpus B."""
    test_storage = tmp_path / "storage"
    monkeypatch.setattr("src.store.STORAGE_DIR", test_storage)

    # Corpus A
    store_a = CorpusVectorStore("corpus_isolated_a")
    chunk_a = create_sample_chunk(
        "a_01", "lisbon_doc", "lisbon_treaty.txt", 1,
        "Confidential diplomatic dispatch regarding maritime provisions in Lisbon."
    )
    store_a.index_chunks([chunk_a])

    # Corpus B
    store_b = CorpusVectorStore("corpus_isolated_b")
    chunk_b = create_sample_chunk(
        "b_01", "porto_doc", "porto_expedition.txt", 1,
        "Logbook of naval vessels departing from the Douro estuary in Porto."
    )
    store_b.index_chunks([chunk_b])

    # Assert separate physical directories
    assert store_a.storage_path != store_b.storage_path
    assert (store_a.storage_path / "index_metadata.json").exists()
    assert (store_b.storage_path / "index_metadata.json").exists()

    # Query Corpus A with terms prominent in Corpus B
    results_a = store_a.search("naval vessels departing Douro estuary Porto", top_k=5)
    for res in results_a:
        assert res.chunk.doc_id != "porto_doc"
        assert res.chunk.chunk_id != "b_01"
        assert "Porto" not in res.chunk.text

    # Query Corpus B with terms prominent in Corpus A
    results_b = store_b.search("diplomatic dispatch maritime Lisbon", top_k=5)
    for res in results_b:
        assert res.chunk.doc_id != "lisbon_doc"
        assert res.chunk.chunk_id != "a_01"
        assert "Lisbon" not in res.chunk.text


# ---------------------------------------------------------------------------
# Coordinated Retrieval Test
# ---------------------------------------------------------------------------

def test_coordinated_dual_retrieval_unmerged(tmp_path: Path, monkeypatch):
    test_storage = tmp_path / "storage"
    monkeypatch.setattr("src.store.STORAGE_DIR", test_storage)

    c1 = create_sample_chunk(
        "chunk_1", "doc1", "doc1.txt", 1,
        "Epigraphic survey documenting imperial Latin inscriptions at Conimbriga."
    )
    chunks = [c1]

    result = execute_dual_retrieval(
        corpus_id="test_dual",
        chunks=chunks,
        query="Latin inscriptions at Conimbriga",
        top_k=1,
    )

    # Check that signals are preserved side-by-side
    assert result.query == "Latin inscriptions at Conimbriga"
    assert result.corpus_id == "test_dual"

    # Lexical signal
    assert result.lexical_report is not None
    assert "Conimbriga" in result.lexical_report.term_results
    assert result.lexical_report.term_results["Conimbriga"].present_in_corpus is True

    # Semantic signal
    assert len(result.semantic_results) == 1
    assert result.semantic_results[0].chunk.doc_id == "doc1"
    assert result.semantic_results[0].rank == 1

    # Verify no combined or fused epistemic score exists
    result_dict = result.to_dict()
    assert "blended_score" not in result_dict
    assert "combined_score" not in result_dict
    assert "confidence_score" not in result_dict
