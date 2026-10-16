"""Coordinated dual retrieval engine for Corpus Inquiry Lab.

Methodological Safeguard:
Lexical concordance and semantic vector retrieval are executed alongside each other
but NEVER combined into a blended epistemic score. They represent categorically
different evidentiary signals and remain independently inspectable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from src.ingest import DocumentChunk
from src.lexical import LexicalScanReport, scan_corpus_lexical
from src.store import CorpusVectorStore, SemanticResult


@dataclass
class CoordinatedRetrievalResult:
    """Result container preserving lexical and semantic findings as independent signals."""
    query: str
    corpus_id: str
    lexical_report: LexicalScanReport
    semantic_results: list[SemanticResult]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "corpus_id": self.corpus_id,
            "lexical_report": self.lexical_report.to_dict(),
            "semantic_results": [r.to_dict() for r in self.semantic_results],
            "timestamp": self.timestamp,
        }


def execute_dual_retrieval(
    corpus_id: str,
    chunks: list[DocumentChunk],
    query: str,
    top_k: int = 5,
    explicit_lexical_terms: list[str] | None = None,
    vector_store: CorpusVectorStore | None = None,
) -> CoordinatedRetrievalResult:
    """Execute both deterministic lexical search and dense semantic retrieval.

    Both signals are computed and returned side-by-side without fusion.
    """
    # 1. Deterministic full-corpus lexical scan
    lexical_report = scan_corpus_lexical(
        chunks=chunks,
        query=query,
        explicit_terms=explicit_lexical_terms,
    )

    # 2. Dense semantic retrieval via isolated per-corpus store
    store = vector_store or CorpusVectorStore(corpus_id)
    if not store.is_indexed() and chunks:
        store.index_chunks(chunks)

    semantic_results = store.search(query=query, top_k=top_k)

    return CoordinatedRetrievalResult(
        query=query,
        corpus_id=corpus_id,
        lexical_report=lexical_report,
        semantic_results=semantic_results,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
