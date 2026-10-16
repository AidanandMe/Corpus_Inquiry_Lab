"""Epistemic synthesis engine and negative-constraint prompting for Corpus Inquiry Lab.

Methodological Safeguards:
1. Structured 4-Zone Output: Direct Claims, Inferences, Divergences, Epistemic Limits.
2. Anti-Harmonization (Artificial Coherence Prevention): Prohibits smoothing over
   conflicts between sources. Competing claims must be juxtaposed.
3. Absence of Evidence as Epistemic Limitation: Absence in retrieved passages is
   reported as a limitation of the search, never as proof of historical silence.
4. Citation Auditability: All direct claims must cite [Card X] identifiers, which
   are verified against the retrieved evidence set.
5. Dual Synthesis Modes: Live Gemini LLM when an API key is provided, or a built-in
   deterministic grounded evaluator when offline or running without external keys.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.retrieve import CoordinatedRetrievalResult
from src.store import SemanticResult

DEFAULT_MODEL_NAME = "gemini-2.5-flash"


@dataclass
class CitationAudit:
    """Audit report verifying citations in generated text against retrieved cards."""
    available_cards: list[int]
    cited_cards: list[int]
    invalid_cards: list[int]
    all_citations_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpistemicSynthesis:
    """Structured synthesis partitioned into epistemic boundary categories."""
    query: str
    corpus_id: str
    direct_claims: str
    inferred_connections: str
    divergent_perspectives: str
    epistemic_limits: str
    raw_markdown: str
    citation_audit: CitationAudit
    model_name: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "corpus_id": self.corpus_id,
            "direct_claims": self.direct_claims,
            "inferred_connections": self.inferred_connections,
            "divergent_perspectives": self.divergent_perspectives,
            "epistemic_limits": self.epistemic_limits,
            "raw_markdown": self.raw_markdown,
            "citation_audit": self.citation_audit.to_dict(),
            "model_name": self.model_name,
            "timestamp": self.timestamp,
        }


EPISTEMIC_SYSTEM_INSTRUCTION = """You are an AI research assistant in the Corpus Inquiry Lab, a digital humanities research environment.
Your purpose is to examine the boundaries of available evidence for interpretive scholarship.

CRITICAL METHODOLOGICAL RULES:
1. GROUNDING & CITATION: Every factual claim in Section 1 MUST cite one or more source cards using [Card X] notation. Do NOT make claims unsupported by the cards.
2. NO ARTIFICIAL COHERENCE: If retrieved cards contain conflicting dates, locations, identifications, or interpretations, DO NOT harmonize or smooth over them. You must juxtapose them in Section 3 as competing or divergent perspectives.
3. INFERENCE ISOLATION: Clearly distinguish what the text explicitly states from what may be inferred. Place all interpretive hypotheses, contextual extrapolations, or deductive bridges in Section 2.
4. ABSENCE OF EVIDENCE: If an aspect of the query is not addressed in the cards, or if the lexical check reports zero matches, explicitly declare in Section 4 that the corpus provides no supporting evidence through the searches performed. Do NOT claim absolute historical non-existence; acknowledge that unretrieved formulations or alternative terminology may exist.

You MUST structure your response into EXACTLY these four markdown headings:
### 1. Directly Attributed Claims
(Direct observations backed by [Card X] citations)

### 2. Inferred Connections
(Interpretive leaps, deductive bridges, or contextual hypotheses, referencing the specific cards that inspire them)

### 3. Divergent or Contrasting Perspectives
(Conflicting information, competing accounts, or chronological/interpretive tensions across sources. If none exist, state 'No conflicting evidence was observed among the retrieved cards.')

### 4. Epistemic Limitations & Unevidenced Aspects
(Aspects of the question that the performed searches did not substantiate, acknowledging the boundaries of the corpus and search)
"""


def build_evidence_context(retrieval_result: CoordinatedRetrievalResult) -> str:
    """Format lexical and semantic retrieval results into transparent prompt context."""
    sections: list[str] = []

    # 1. Deterministic Lexical Findings
    lex = retrieval_result.lexical_report
    lex_lines = ["--- DETERMINISTIC LEXICAL CONCORDANCE FINDINGS ---"]
    if lex.searched_terms:
        for term, res in lex.term_results.items():
            lex_lines.append(f"- {res.epistemic_statement}")
    else:
        lex_lines.append("- No explicit search terms extracted.")
    lex_lines.append(f"Notice: {lex.methodological_notice}")
    sections.append("\n".join(lex_lines))

    # 2. Retrieved Semantic Evidence Cards
    card_lines = ["--- RETRIEVED EVIDENCE CARDS ---"]
    if retrieval_result.semantic_results:
        for res in retrieval_result.semantic_results:
            card_lines.append(
                f"[CARD {res.rank}]\n"
                f"Source: {res.chunk.doc_title} (Page: {res.chunk.page_number}, Chunk: {res.chunk.chunk_id})\n"
                f"Proximity Metric: {res.similarity_score:.4f} (cosine similarity)\n"
                f'Excerpt: "{res.chunk.text}"\n'
            )
    else:
        card_lines.append("No semantic evidence cards were retrieved for this query.")
    sections.append("\n".join(card_lines))

    return "\n\n".join(sections)


def construct_synthesis_prompt(query: str, retrieval_result: CoordinatedRetrievalResult) -> str:
    """Assemble the complete prompt for the LLM."""
    evidence_context = build_evidence_context(retrieval_result)
    return (
        f"{EPISTEMIC_SYSTEM_INSTRUCTION}\n\n"
        f"RESEARCH QUERY:\n\"{query}\"\n\n"
        f"{evidence_context}\n\n"
        f"Now produce your structured four-section epistemic evaluation based strictly on the above context."
    )


def audit_citations(markdown_text: str, available_card_count: int) -> CitationAudit:
    """Audit citations in generated markdown against available card indices."""
    available_cards = list(range(1, available_card_count + 1))
    raw_citations = re.findall(r'\[Card\s+(\d+)\]', markdown_text, flags=re.IGNORECASE)
    cited_cards = sorted(list(set(int(c) for c in raw_citations)))

    invalid_cards = [c for c in cited_cards if c not in available_cards]
    all_valid = len(invalid_cards) == 0

    return CitationAudit(
        available_cards=available_cards,
        cited_cards=cited_cards,
        invalid_cards=invalid_cards,
        all_citations_valid=all_valid,
    )


def parse_epistemic_sections(markdown_text: str) -> tuple[str, str, str, str]:
    """Extract the four mandatory epistemic sections from markdown."""
    patterns = [
        r'###\s*1\.\s*Directly Attributed Claims\s*\n(.*?)(?=###\s*2\.|\Z)',
        r'###\s*2\.\s*Inferred Connections\s*\n(.*?)(?=###\s*3\.|\Z)',
        r'###\s*3\.\s*Divergent or Contrasting Perspectives\s*\n(.*?)(?=###\s*4\.|\Z)',
        r'###\s*4\.\s*Epistemic Limitations & Unevidenced Aspects\s*\n(.*?)(?=\Z)',
    ]

    extracted = []
    for pattern in patterns:
        match = re.search(pattern, markdown_text, flags=re.DOTALL | re.IGNORECASE)
        extracted.append(match.group(1).strip() if match else "")

    return extracted[0], extracted[1], extracted[2], extracted[3]


def generate_built_in_grounded_synthesis(
    query: str,
    retrieval_result: CoordinatedRetrievalResult,
) -> str:
    """Generate a deterministic, evidence-grounded synthesis without requiring an external API key.

    This ensures the research instrument is 100% operational out of the box.
    """
    cards = retrieval_result.semantic_results
    lex = retrieval_result.lexical_report

    # Section 1: Directly Attributed Claims
    claims: list[str] = []
    if not cards:
        claims.append(f"- No passages in the active corpus matched the query '{query}' under the current retrieval parameters.")
    else:
        for res in cards[:3]:
            doc_name = res.chunk.doc_title
            page = res.chunk.page_number
            # Take the first 1-2 sentences of the excerpt
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', res.chunk.text) if s.strip()]
            snippet = " ".join(sentences[:2]) if sentences else res.chunk.text[:150]
            claims.append(f"- In **{doc_name}** (p. {page}): \"{snippet}\" [Card {res.rank}]")

    sec1_text = "\n".join(claims)

    # Section 2: Inferred Connections
    inferences: list[str] = []
    if cards:
        doc_titles = list(dict.fromkeys(r.chunk.doc_title for r in cards))
        if len(doc_titles) > 1:
            inferences.append(
                f"- The retrieved evidence bridges multiple accounts ({', '.join(doc_titles)}). "
                "Synthesizing these sources into a single continuity assumes comparable reporting criteria, "
                "which should remain a matter of critical interpretation rather than computational certainty."
            )
        else:
            inferences.append(
                f"- The evidence is concentrated within a single source ({doc_titles[0]}). "
                "Extrapolating broader historical conclusions relies on the singular perspective of this author."
            )
    else:
        inferences.append("- No inferential connections can be established in the absence of retrieved textual evidence.")

    sec2_text = "\n".join(inferences)

    # Section 3: Divergent Perspectives
    divergences: list[str] = []
    # Check for obvious chronological or textual tensions
    years_found = {}
    for res in cards:
        years = re.findall(r'\b(1[89]\d\d|20\d\d)\b', res.chunk.text)
        if years:
            years_found[res.rank] = (res.chunk.doc_title, years)

    if len(years_found) >= 2:
        items = list(years_found.items())
        rank_a, (doc_a, yrs_a) = items[0]
        rank_b, (doc_b, yrs_b) = items[1]
        if set(yrs_a) != set(yrs_b):
            divergences.append(
                f"- **Chronological / Source Divergence**: [Card {rank_a}] ({doc_a}) references dates ({', '.join(yrs_a)}), "
                f"whereas [Card {rank_b}] ({doc_b}) references ({', '.join(yrs_b)}). "
                "The system explicitly juxtaposes these competing dates rather than harmonizing them."
            )
    if not divergences:
        divergences.append("- No overt contradictory statements were observed across the specific retrieved cards.")

    sec3_text = "\n".join(divergences)

    # Section 4: Epistemic Limitations & Unevidenced Aspects
    limits: list[str] = []
    absent_terms = [t for t, r in lex.term_results.items() if not r.present_in_corpus]
    if absent_terms:
        for t in absent_terms:
            limits.append(
                f"- **Lexical Miss for '{t}'**: This term was not located in the corpus using the deterministic lexical scan. "
                "The corpus provides no evidence for this exact formulation through the searches performed. "
                "(Note: Alternative historical terms, synonyms, or translations may still exist)."
            )
    if not cards:
        limits.append(f"- Complete absence of semantic retrieval matches for '{query}'.")
    else:
        limits.append(
            f"- Evidence is bounded strictly to the {len(cards)} excerpt(s) retrieved. "
            "Context outside these passages was not evaluated and cannot be presumed to corroborate the query."
        )

    sec4_text = "\n".join(limits)

    return (
        f"### 1. Directly Attributed Claims\n{sec1_text}\n\n"
        f"### 2. Inferred Connections\n{sec2_text}\n\n"
        f"### 3. Divergent or Contrasting Perspectives\n{sec3_text}\n\n"
        f"### 4. Epistemic Limitations & Unevidenced Aspects\n{sec4_text}"
    )


def synthesize_epistemic_response(
    query: str,
    retrieval_result: CoordinatedRetrievalResult,
    api_key: str | None = None,
    client: Any = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> EpistemicSynthesis:
    """Generate an epistemic synthesis from coordinated retrieval results.

    If an API key or custom client is present, calls the generative LLM.
    If no API key is present, falls back automatically to the built-in deterministic
    grounded synthesis engine, guaranteeing a seamless, crash-free experience.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    prompt = construct_synthesis_prompt(query, retrieval_result)
    available_card_count = len(retrieval_result.semantic_results)
    used_model = model_name

    if client is not None:
        # Injected client (e.g. for testing)
        raw_response = client.generate(prompt)
    elif key:
        # Live Gemini API call
        from google import genai

        genai_client = genai.Client(api_key=key)
        response = genai_client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        raw_response = response.text or ""
    else:
        # Built-in local grounded evaluator (100% turnkey offline)
        raw_response = generate_built_in_grounded_synthesis(query, retrieval_result)
        used_model = "Built-in Grounded Evaluator (Offline)"

    # Audit citations
    citation_audit = audit_citations(raw_response, available_card_count)

    # Parse sections
    sec1, sec2, sec3, sec4 = parse_epistemic_sections(raw_response)

    return EpistemicSynthesis(
        query=query,
        corpus_id=retrieval_result.corpus_id,
        direct_claims=sec1,
        inferred_connections=sec2,
        divergent_perspectives=sec3,
        epistemic_limits=sec4,
        raw_markdown=raw_response,
        citation_audit=citation_audit,
        model_name=used_model,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
