"""Unit tests for Phase 3: Epistemic Prompt Engine, Citation Auditing, and Synthesis Formatting."""

from pathlib import Path
import pytest

from src.ingest import DocumentChunk
from src.lexical import LexicalMatch, LexicalScanReport, LexicalTermResult
from src.retrieve import CoordinatedRetrievalResult
from src.store import SemanticResult
from src.synthesize import (
    audit_citations,
    build_evidence_context,
    construct_synthesis_prompt,
    parse_epistemic_sections,
    synthesize_epistemic_response,
)


class MockLLMClient:
    """Mock client returning deterministic responses for testing."""
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_prompt = ""

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response_text


def create_test_retrieval_result() -> CoordinatedRetrievalResult:
    chunk1 = DocumentChunk(
        chunk_id="doc_a_p1_c0",
        doc_id="doc_a",
        doc_title="excavation_alpha.pdf",
        page_number=1,
        char_start=0,
        char_end=120,
        text="Excavation of Sector Alpha began in 1924 under Professor Martinez.",
    )
    chunk2 = DocumentChunk(
        chunk_id="doc_b_p4_c1",
        doc_id="doc_b",
        doc_title="reinterpretation_beta.txt",
        page_number=4,
        char_start=0,
        char_end=135,
        text="Archival records indicate that field work in Sector Alpha did not commence until 1931.",
    )

    sem1 = SemanticResult(chunk=chunk1, similarity_score=0.88, distance=0.24, rank=1)
    sem2 = SemanticResult(chunk=chunk2, similarity_score=0.82, distance=0.36, rank=2)

    lex_results = {
        "Sector Alpha": LexicalTermResult(
            term="Sector Alpha",
            total_occurrences=2,
            documents_count=2,
            present_in_corpus=True,
            matches=[],
            epistemic_statement="Term 'Sector Alpha' found 2 time(s) across 2 document(s) in this lexical scan.",
        ),
        "Emporion": LexicalTermResult(
            term="Emporion",
            total_occurrences=0,
            documents_count=0,
            present_in_corpus=False,
            matches=[],
            epistemic_statement="The searched term or phrase 'Emporion' was not found in the corpus using this lexical check.",
        ),
    }

    lex_report = LexicalScanReport(
        searched_terms=["Sector Alpha", "Emporion"],
        term_results=lex_results,
        any_present=True,
        all_present=False,
        methodological_notice="Notice: Lexical search reports literal term presence or absence only.",
    )

    return CoordinatedRetrievalResult(
        query="When did excavation of Sector Alpha begin, and was Emporion involved?",
        corpus_id="archaeology_test",
        lexical_report=lex_report,
        semantic_results=[sem1, sem2],
        timestamp="2026-09-05T00:00:00Z",
    )


def test_build_evidence_context_and_prompt():
    result = create_test_retrieval_result()
    prompt = construct_synthesis_prompt(result.query, result)

    # Verify presence of mandatory instructions
    assert "### 1. Directly Attributed Claims" in prompt
    assert "### 2. Inferred Connections" in prompt
    assert "### 3. Divergent or Contrasting Perspectives" in prompt
    assert "### 4. Epistemic Limitations & Unevidenced Aspects" in prompt

    # Verify lexical report inclusion
    assert "Term 'Sector Alpha' found 2 time(s)" in prompt
    assert "The searched term or phrase 'Emporion' was not found" in prompt

    # Verify evidence card formatting
    assert "[CARD 1]" in prompt
    assert "Source: excavation_alpha.pdf (Page: 1" in prompt
    assert "[CARD 2]" in prompt
    assert "Source: reinterpretation_beta.txt (Page: 4" in prompt


def test_citation_auditing():
    # Valid citations (cards 1 and 2 exist)
    text_valid = "Sector Alpha was excavated in 1924 [Card 1] or possibly 1931 [Card 2]."
    audit1 = audit_citations(text_valid, available_card_count=2)
    assert audit1.all_citations_valid is True
    assert audit1.cited_cards == [1, 2]
    assert audit1.invalid_cards == []

    # Invalid hallucinated citation (card 5 does not exist)
    text_invalid = "Sector Alpha was founded by Greeks [Card 5]."
    audit2 = audit_citations(text_invalid, available_card_count=2)
    assert audit2.all_citations_valid is False
    assert 5 in audit2.invalid_cards


def test_section_parsing():
    sample_response = (
        "### 1. Directly Attributed Claims\n"
        "- Sector Alpha excavation is reported in 1924 [Card 1].\n"
        "- Another document reports it started in 1931 [Card 2].\n\n"
        "### 2. Inferred Connections\n"
        "- The chronological discrepancy may reflect preparatory surveys versus formal excavation.\n\n"
        "### 3. Divergent or Contrasting Perspectives\n"
        "- Card 1 (1924) directly contradicts Card 2 (1931) regarding the start date.\n\n"
        "### 4. Epistemic Limitations & Unevidenced Aspects\n"
        "- The searched term 'Emporion' was not found in the corpus; no evidence addresses Emporion's involvement."
    )

    sec1, sec2, sec3, sec4 = parse_epistemic_sections(sample_response)
    assert "Sector Alpha excavation is reported in 1924 [Card 1]" in sec1
    assert "preparatory surveys versus formal excavation" in sec2
    assert "Card 1 (1924) directly contradicts Card 2 (1931)" in sec3
    assert "Emporion' was not found in the corpus" in sec4


def test_synthesize_with_mock_client():
    result = create_test_retrieval_result()
    mock_response = (
        "### 1. Directly Attributed Claims\n"
        "- Sector Alpha excavation is documented as beginning in 1924 [Card 1].\n"
        "- Field work is alternatively dated to 1931 [Card 2].\n\n"
        "### 2. Inferred Connections\n"
        "- Different authors may define 'field work' and 'excavation' differently.\n\n"
        "### 3. Divergent or Contrasting Perspectives\n"
        "- Competing dates: Card 1 claims 1924, whereas Card 2 records 1931.\n\n"
        "### 4. Epistemic Limitations & Unevidenced Aspects\n"
        "- Emporion: The searched term was not found; the corpus provides no evidence on this question through the searches performed."
    )
    mock_client = MockLLMClient(mock_response)

    synthesis = synthesize_epistemic_response(
        query=result.query,
        retrieval_result=result,
        client=mock_client,
    )

    assert synthesis.query == result.query
    assert synthesis.corpus_id == "archaeology_test"
    assert synthesis.citation_audit.all_citations_valid is True
    assert synthesis.citation_audit.cited_cards == [1, 2]
    assert "Competing dates: Card 1 claims 1924, whereas Card 2 records 1931." in synthesis.divergent_perspectives
    assert "the corpus provides no evidence on this question through the searches performed" in synthesis.epistemic_limits


def test_missing_api_key_uses_built_in_synthesis():
    result = create_test_retrieval_result()
    synthesis = synthesize_epistemic_response(
        query=result.query,
        retrieval_result=result,
        api_key=None,
    )
    assert synthesis.model_name == "Built-in Grounded Evaluator (Offline)"
    assert synthesis.citation_audit.all_citations_valid is True
    assert "[Card 1]" in synthesis.direct_claims
    assert "Emporion" in synthesis.epistemic_limits
