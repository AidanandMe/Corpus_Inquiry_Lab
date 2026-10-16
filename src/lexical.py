"""Deterministic full-corpus lexical concordance scanner for Corpus Inquiry Lab.

Methodological Principles:
1. Lexical presence/absence is strictly deterministic and term-based.
2. A zero lexical match establishes ONLY:
   "The searched term or phrase was not found in the corpus using this lexical check."
   It does NOT establish conceptual or historical absence.
3. No LLM or heuristic model is used to generate or alter search terms.
   Search terms are either supplied directly or extracted using a transparent,
   documented tokenization rule.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.ingest import DocumentChunk

# Transparent, documented stopword list to filter high-frequency function words
# when extracting default search terms from a natural language question.
DOCUMENTED_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
        "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
        "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
        "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
        "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
        "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
        "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
        "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
        "yourself", "yourselves",
        # Common Latin/Romance scholarly particles
        "et", "de", "in", "non", "ad", "ex", "cum", "ab", "per", "sub", "pro",
    }
)


@dataclass
class LexicalMatch:
    """A single occurrence of a searched term in a document chunk."""
    doc_id: str
    doc_title: str
    page_number: int
    chunk_id: str
    char_start: int  # Offset within chunk
    char_end: int    # Offset within chunk
    snippet: str     # Context snippet surrounding match

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LexicalTermResult:
    """Findings for a single explicit search term or phrase."""
    term: str
    total_occurrences: int
    documents_count: int
    present_in_corpus: bool
    matches: list[LexicalMatch]
    epistemic_statement: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "total_occurrences": self.total_occurrences,
            "documents_count": self.documents_count,
            "present_in_corpus": self.present_in_corpus,
            "matches": [m.to_dict() for m in self.matches],
            "epistemic_statement": self.epistemic_statement,
        }


@dataclass
class LexicalScanReport:
    """Aggregated report of deterministic lexical concordance across the corpus."""
    searched_terms: list[str]
    term_results: dict[str, LexicalTermResult]
    any_present: bool
    all_present: bool
    methodological_notice: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "searched_terms": self.searched_terms,
            "term_results": {t: res.to_dict() for t, res in self.term_results.items()},
            "any_present": self.any_present,
            "all_present": self.all_present,
            "methodological_notice": self.methodological_notice,
        }


def extract_deterministic_terms(
    query: str,
    explicit_terms: list[str] | None = None,
) -> list[str]:
    """Deterministically extract search terms from a query string.

    Rule:
    1. If explicit_terms is provided, return cleaned unique explicit terms.
    2. Otherwise:
       a. Extract all quoted phrases (e.g. "Bronze Age", 'Roman forum') as exact units.
       b. Split unquoted text into words.
       c. Normalize to lowercase, strip punctuation.
       d. Filter out documented stopwords and tokens shorter than 3 characters.
       e. Preserve original phrase order, deduplicating terms.
    """
    if explicit_terms is not None:
        terms: list[str] = []
        for t in explicit_terms:
            cleaned = t.strip()
            if cleaned and cleaned not in terms:
                terms.append(cleaned)
        return terms

    terms = []

    # Step 1: Extract quoted phrases
    quoted_phrases = re.findall(r'["\'](.*?)["\']', query)
    for phrase in quoted_phrases:
        cleaned_phrase = phrase.strip()
        if cleaned_phrase and cleaned_phrase.lower() not in [t.lower() for t in terms]:
            terms.append(cleaned_phrase)

    # Step 2: Remove quoted portions from query to process remaining tokens
    unquoted = re.sub(r'["\'].*?["\']', ' ', query)

    # Step 3: Extract content words and numeric identifiers (e.g., years, inscription numbers)
    tokens = re.findall(r'\b[^\W_]+(?:-[^\W_]+)*\b', unquoted, flags=re.UNICODE)
    for token in tokens:
        normalized = token.lower().strip()
        if len(normalized) >= 3 and normalized not in DOCUMENTED_STOPWORDS:
            if normalized not in [t.lower() for t in terms]:
                terms.append(token)

    return terms


def scan_corpus_lexical(
    chunks: list[DocumentChunk],
    query: str,
    explicit_terms: list[str] | None = None,
    context_window: int = 50,
) -> LexicalScanReport:
    """Scan all chunks in the corpus for exact occurrences of terms.

    This operation is strictly deterministic and case-insensitive.
    """
    terms = extract_deterministic_terms(query, explicit_terms)
    term_results: dict[str, LexicalTermResult] = {}

    for term in terms:
        matches: list[LexicalMatch] = []
        matching_docs: set[str] = set()

        # Build regex pattern for exact substring/word-boundary match
        # Uses lookarounds so terms starting/ending with punctuation still match
        escaped_term = re.escape(term)
        pattern = re.compile(rf'(?<!\w){escaped_term}(?!\w)', re.IGNORECASE)

        for chunk in chunks:
            for m in pattern.finditer(chunk.text):
                start = m.start()
                end = m.end()

                # Build surrounding snippet for concordance context
                snip_start = max(0, start - context_window)
                snip_end = min(len(chunk.text), end + context_window)
                prefix = "..." if snip_start > 0 else ""
                suffix = "..." if snip_end < len(chunk.text) else ""
                snippet = f"{prefix}{chunk.text[snip_start:snip_end].strip()}{suffix}"

                matches.append(
                    LexicalMatch(
                        doc_id=chunk.doc_id,
                        doc_title=chunk.doc_title,
                        page_number=chunk.page_number,
                        chunk_id=chunk.chunk_id,
                        char_start=start,
                        char_end=end,
                        snippet=snippet,
                    )
                )
                matching_docs.add(chunk.doc_id)

        count = len(matches)
        doc_count = len(matching_docs)
        present = count > 0

        if present:
            statement = (
                f"Term '{term}' found {count} time(s) across {doc_count} document(s) "
                "in this lexical scan."
            )
        else:
            statement = (
                f"The searched term or phrase '{term}' was not found in the corpus "
                "using this lexical check."
            )

        term_results[term] = LexicalTermResult(
            term=term,
            total_occurrences=count,
            documents_count=doc_count,
            present_in_corpus=present,
            matches=matches,
            epistemic_statement=statement,
        )

    any_present = any(res.present_in_corpus for res in term_results.values()) if term_results else False
    all_present = all(res.present_in_corpus for res in term_results.values()) if term_results else False

    methodological_notice = (
        "Methodological Note: Lexical search reports literal term presence or absence only. "
        "A zero lexical match establishes that the term was not found, not that the concept "
        "is absent. Concepts may appear under synonyms, historical variants, translations, "
        "or paraphrase."
    )

    return LexicalScanReport(
        searched_terms=terms,
        term_results=term_results,
        any_present=any_present,
        all_present=all_present,
        methodological_notice=methodological_notice,
    )
