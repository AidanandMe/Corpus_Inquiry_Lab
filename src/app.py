"""Corpus Inquiry Lab - Dark Aesthetic Epistemic Workbench for Humanities Scholarship.

Matches the exact dark midnight blue theme, layout, and visual identity from the user's mockup:
- Top Nav: Corpus Inquiry Lab | Texts, Questions, Evidence | About, Help, LT
- Hero: Corpus Inquiry Lab | Explore what a corpus supports — and what it cannot establish. | Reset experiment
- Corpus Bar: [Corpus Select] OR [Upload documents: PDF, DOCX, TXT] | 🟢 Corpus ready
- Inquiry Bar: [🔍 What would you like to investigate in this corpus?] [Run Inquiry →]
- Three Core Columns:
    1. THE ANSWER
    2. EVIDENCE
    3. WHAT THE CORPUS CANNOT ESTABLISH
- Footer: CORPUS INQUIRY LAB | A TOOL FOR CRITICAL ENGAGEMENT WITH TEXTS ... EXPLORE • QUESTION • REFLECT
"""

from __future__ import annotations

import html
import os
import re
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import (
    PREPARED_CORPORA_DIR,
    SESSION_UPLOADS_DIR,
    STORAGE_DIR,
    ensure_directories,
)
from src.ingest import (
    CorpusManifest,
    DocumentChunk,
    ingest_corpus_directory,
    list_prepared_corpora,
    purge_session_uploads,
)
from src.lexical import DOCUMENTED_STOPWORDS
from src.retrieve import CoordinatedRetrievalResult, execute_dual_retrieval
from src.store import CorpusVectorStore
from src.synthesize import (
    DEFAULT_MODEL_NAME,
    EpistemicSynthesis,
    synthesize_epistemic_response,
)


def format_epistemic_html(text: str) -> str:
    """Safely escape text for HTML while preserving Markdown bold/italics and styling [Card X] citations.

    Ensures epigraphic angle brackets (e.g., <Augustus>) are never stripped as HTML tags.
    """
    escaped = html.escape(text)
    # Convert **bold** and *italic*
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^\*\n]+?)\*(?!\*)", r"<i>\1</i>", escaped)
    # Highlight [Card X] citations
    escaped = re.sub(
        r"(\[Card\s+\d+\])",
        r'<span class="citation-badge">\1</span>',
        escaped,
        flags=re.IGNORECASE,
    )
    return escaped.replace("\n", "<br>")


# Set page config with dark theme preference
st.set_page_config(
    page_title="Corpus Inquiry Lab",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ensure_directories()

# Initialize Session State
if "active_corpus_id" not in st.session_state:
    st.session_state.active_corpus_id = "sample_archaeological_survey"
if "last_selected_prepared_id" not in st.session_state:
    st.session_state.last_selected_prepared_id = "sample_archaeological_survey"
if "active_chunks" not in st.session_state:
    st.session_state.active_chunks = []
if "active_manifest" not in st.session_state:
    st.session_state.active_manifest = None
if "retrieval_result" not in st.session_state:
    st.session_state.retrieval_result = None
if "synthesis_result" not in st.session_state:
    st.session_state.synthesis_result = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "last_uploaded_hash" not in st.session_state:
    st.session_state.last_uploaded_hash = ""


def reset_workspace() -> None:
    """Purge temporary uploads, clear custom vector index, and reset session state to pristine baseline."""
    purge_session_uploads()
    try:
        CorpusVectorStore("session_custom_corpus").clear()
    except Exception:
        pass
    st.session_state.active_corpus_id = "sample_archaeological_survey"
    st.session_state.last_selected_prepared_id = "sample_archaeological_survey"
    st.session_state.active_chunks = []
    st.session_state.active_manifest = None
    st.session_state.retrieval_result = None
    st.session_state.synthesis_result = None
    st.session_state.uploader_key += 1
    st.session_state.last_uploaded_hash = ""
    st.rerun()


reset_experiment = reset_workspace

# Apply Dark Modern Academic Theme CSS
st.markdown(
    """
    <style>
    /* Dark Theme Core Background & Text */
    .stApp {
        background-color: #070d1e;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Top Nav Bar */
    .nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0.5rem 1.2rem 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.5rem;
    }
    .nav-left {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .nav-logo-text {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.01em;
    }
    .nav-divider {
        color: rgba(255, 255, 255, 0.2);
        font-weight: 300;
    }
    .nav-tagline {
        color: #94a3b8;
        font-size: 0.88rem;
    }
    .nav-right {
        display: flex;
        align-items: center;
        gap: 1.4rem;
        font-size: 0.9rem;
        color: #94a3b8;
    }
    .nav-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.8rem;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 28, 63, 0.9) 0%, rgba(11, 19, 43, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 12px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.4rem;
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        color: #38bdf8;
        font-size: 1.25rem;
        margin-top: 0.4rem;
        font-weight: 400;
    }
    .hero-taglines {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        line-height: 1.5;
        text-align: right;
        margin-top: 0.6rem;
    }

    /* Reset Button */
    div[data-testid="stButton"] > button:has-text("Reset experiment") {
        background-color: rgba(30, 41, 59, 0.8) !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        font-size: 0.85rem !important;
        padding: 0.35rem 0.8rem !important;
    }

    /* Bordered Container for Corpus Bar */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(15, 23, 42, 0.75) !important;
        border: 1px solid rgba(56, 189, 248, 0.18) !important;
        border-radius: 10px !important;
        padding: 0.8rem 1.2rem !important;
        margin-bottom: 1rem !important;
    }

    /* Green status ready pill */
    .status-ready {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #34d399;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .status-dot {
        width: 10px;
        height: 10px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
    }

    /* Three Columns Panel Styling */
    .column-panel {
        background-color: rgba(13, 22, 45, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.14);
        border-radius: 10px;
        padding: 1.4rem 1.4rem;
        min-height: 480px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .column-header {
        font-size: 1.05rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }
    .column-divider {
        height: 1px;
        background-color: rgba(255, 255, 255, 0.12);
        margin-bottom: 1.2rem;
    }

    /* Citation Badge */
    .citation-badge {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 4px;
        padding: 0.05rem 0.35rem;
        font-size: 0.82em;
        font-weight: 700;
    }

    /* Evidence Excerpt Cards */
    .evidence-block {
        background-color: rgba(18, 30, 60, 0.65);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-bottom: 1rem;
    }
    .evidence-meta {
        font-size: 0.82rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 0.35rem;
    }
    .evidence-quote {
        color: #e2e8f0;
        font-size: 0.92rem;
        line-height: 1.5;
        font-style: italic;
    }

    /* Footer */
    .footer-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 2rem 0.5rem 1rem 0.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        color: #475569;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        margin-top: 2rem;
    }

    /* Streamlit Input Customization for Dark Look */
    div[data-baseweb="input"] {
        background-color: rgba(15, 23, 42, 0.85) !important;
        border-color: rgba(56, 189, 248, 0.25) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    input {
        color: #ffffff !important;
    }
    div[data-baseweb="select"] > div {
        background-color: rgba(15, 23, 42, 0.85) !important;
        border-color: rgba(56, 189, 248, 0.25) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    /* Primary Run Inquiry Button */
    div[data-testid="stButton"] > button:has-text("Run Inquiry") {
        background: #0284c7 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.4rem !important;
        font-size: 0.98rem !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4) !important;
    }
    div[data-testid="stButton"] > button:has-text("Run Inquiry"):hover {
        background: #0369a1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# TOP NAVIGATION BAR
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="nav-bar">
        <div class="nav-left">
            <span style="font-size: 1.4rem; color: #38bdf8;">📖</span>
            <span class="nav-logo-text">Corpus Inquiry Lab</span>
            <span class="nav-divider">|</span>
            <span class="nav-tagline">Texts, Questions, Evidence.</span>
        </div>
        <div class="nav-right">
            <span>About</span>
            <span>Help</span>
            <div class="nav-avatar">LT</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------------------------
hero_col1, hero_col2 = st.columns([8, 2.5])

with hero_col1:
    st.markdown(
        """
        <div style="padding-left: 0.5rem;">
            <h1 class="hero-title">Corpus Inquiry Lab</h1>
            <div class="hero-subtitle">Explore what a corpus supports — and what it cannot establish.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_col2:
    if st.button("🔄 Reset experiment", use_container_width=True, help="Purge active query, uploads, and session memory."):
        reset_workspace()
    st.markdown(
        """
        <div class="hero-taglines">
            SAME TEXTS.<br>
            SHARPER QUESTIONS.<br>
            RICHER PERSPECTIVES.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CORPUS CONTROL BAR (PREPARED OR UPLOAD)
# ---------------------------------------------------------------------------
prepared_corpora = list_prepared_corpora()
corpus_lookup = {c["name"]: c["id"] for c in prepared_corpora}

with st.container(border=True):
    c_col1, c_col_or, c_col2, c_col3 = st.columns([4, 0.4, 4.5, 2.6])

    with c_col1:
        st.markdown("<span style='font-size: 0.8rem; color: #94a3b8; font-weight: 600;'>Corpus</span>", unsafe_allow_html=True)
        selected_corpus_name = st.selectbox(
            "Corpus",
            options=list(corpus_lookup.keys()),
            index=0,
            label_visibility="collapsed",
        )
        selected_prepared_id = corpus_lookup[selected_corpus_name]

    # If the user explicitly changed the prepared corpus dropdown while custom uploads were active,
    # switch back to the newly selected prepared corpus cleanly.
    if selected_prepared_id != st.session_state.last_selected_prepared_id:
        st.session_state.last_selected_prepared_id = selected_prepared_id
        if st.session_state.active_corpus_id == "session_custom_corpus":
            purge_session_uploads()
            st.session_state.uploader_key += 1
            st.session_state.last_uploaded_hash = ""
            st.session_state.active_corpus_id = selected_prepared_id
            st.session_state.active_chunks = []
            st.session_state.retrieval_result = None
            st.session_state.synthesis_result = None
            st.rerun()

    with c_col_or:
        st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.82rem; font-weight: 700; margin-top: 28px;'>OR</div>", unsafe_allow_html=True)

    with c_col2:
        st.markdown("<span style='font-size: 0.8rem; color: #94a3b8; font-weight: 600;'>Upload documents (PDF, DOCX, or TXT)</span>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
            label_visibility="collapsed",
        )

    # Automatic Ingestion & State Binding (Zero extra clicks required!)
    if uploaded_files:
        current_hash = "-".join(sorted(f.name + str(f.size) for f in uploaded_files))
        if st.session_state.last_uploaded_hash != current_hash:
            with st.spinner("Indexing uploaded documents..."):
                purge_session_uploads()
                for up in uploaded_files:
                    target = SESSION_UPLOADS_DIR / up.name
                    target.write_bytes(up.getbuffer())

                session_id = "session_custom_corpus"
                chunks, manifest = ingest_corpus_directory(
                    SESSION_UPLOADS_DIR,
                    corpus_id=session_id,
                    corpus_name="Participant Uploaded Documents",
                )
                store = CorpusVectorStore(session_id)
                store.index_chunks(chunks)

                st.session_state.active_corpus_id = session_id
                st.session_state.active_chunks = chunks
                st.session_state.active_manifest = manifest
                st.session_state.last_uploaded_hash = current_hash
                st.session_state.retrieval_result = None
                st.session_state.synthesis_result = None
    else:
        # Load prepared corpus if no custom files are uploaded
        if st.session_state.active_corpus_id != selected_prepared_id or not st.session_state.active_chunks:
            st.session_state.active_corpus_id = selected_prepared_id
            corpus_dir = PREPARED_CORPORA_DIR / selected_prepared_id
            chunks, manifest = ingest_corpus_directory(corpus_dir)
            st.session_state.active_chunks = chunks
            st.session_state.active_manifest = manifest
            store = CorpusVectorStore(selected_prepared_id)
            if not store.is_indexed():
                store.index_chunks(chunks)
            st.session_state.last_uploaded_hash = ""
            st.session_state.retrieval_result = None
            st.session_state.synthesis_result = None

    with c_col3:
        manifest = st.session_state.active_manifest
        doc_count = len(manifest.documents) if manifest else 0
        chunk_count = manifest.total_chunks if manifest else 0
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="status-ready">
                <div class="status-dot"></div>
                <span>Corpus ready: {doc_count} doc(s), {chunk_count} chunk(s)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Surface any defensive ingestion warnings (e.g., scanned PDFs lacking an OCR layer)
    if st.session_state.active_manifest and st.session_state.active_manifest.warnings:
        for warn_msg in st.session_state.active_manifest.warnings:
            st.warning(warn_msg, icon="⚠️")


# ---------------------------------------------------------------------------
# INQUIRY SEARCH BAR
# ---------------------------------------------------------------------------
with st.container():
    col_input, col_run = st.columns([9.5, 2.5])

    with col_input:
        query_text = st.text_input(
            "What would you like to investigate in this corpus?",
            value=st.session_state.get("inquiry_query", ""),
            placeholder="🔍  What would you like to investigate in this corpus?",
            label_visibility="collapsed",
        )

    with col_run:
        run_clicked = st.button("Run Inquiry →", use_container_width=True)

    # Clickable suggestions for fast testing
    if st.session_state.active_corpus_id == "sample_archaeological_survey":
        st.markdown(
            "<div style='margin: -0.4rem 0 0.8rem 0.4rem; font-size: 0.8rem; color: #64748b;'>"
            "Try: <i>'When did excavation begin in Sector Alpha?'</i> or <i>'What ceramic and numismatic evidence was catalogued?'</i>"
            "</div>",
            unsafe_allow_html=True,
        )
    elif st.session_state.active_corpus_id == "sample_epigraphic_register":
        st.markdown(
            "<div style='margin: -0.4rem 0 0.8rem 0.4rem; font-size: 0.8rem; color: #64748b;'>"
            "Try: <i>'Which deities are honored across coastal and inland sanctuaries?'</i> or <i>'What inscriptions mention Bandua or Neptune?'</i>"
            "</div>",
            unsafe_allow_html=True,
        )

    # Single-Click Inquiry Execution
    if run_clicked and query_text:
        if not st.session_state.active_chunks:
            st.error("Corpus is empty. Please select or upload documents.")
        else:
            with st.spinner("Searching evidence across corpus..."):
                ret_result = execute_dual_retrieval(
                    corpus_id=st.session_state.active_corpus_id,
                    chunks=st.session_state.active_chunks,
                    query=query_text,
                    top_k=4,
                )
                st.session_state.retrieval_result = ret_result

            with st.spinner("Synthesizing answer and evidence boundaries..."):
                synth_result = synthesize_epistemic_response(
                    query=query_text,
                    retrieval_result=ret_result,
                )
                st.session_state.synthesis_result = synth_result


# ---------------------------------------------------------------------------
# THREE CORE COLUMNS: THE ANSWER | EVIDENCE | WHAT THE CORPUS CANNOT ESTABLISH
# ---------------------------------------------------------------------------
ret_res: CoordinatedRetrievalResult | None = st.session_state.retrieval_result
synth_res: EpistemicSynthesis | None = st.session_state.synthesis_result

col_ans, col_evid, col_limits = st.columns(3)

# ===========================================================================
# COLUMN 1: THE ANSWER
# ===========================================================================
with col_ans:
    ans_body_parts: list[str] = []
    ans_footer_html = ""

    if not synth_res:
        ans_body_parts.append(
            """
            <div style="color: #64748b; font-size: 0.9rem; line-height: 1.6;">
                The grounded synthesis of your inquiry will appear here, strictly derived from verified source passages.
            </div>
            """
        )
    else:
        # Directly Attributed Claims
        if synth_res.direct_claims:
            ans_body_parts.append(
                f"""
                <div style="color: #f1f5f9; font-size: 0.95rem; line-height: 1.6; margin-bottom: 1.2rem;">
                    <b>Directly Supported Findings:</b><br>
                    {format_epistemic_html(synth_res.direct_claims)}
                </div>
                """
            )

        # Inferred Connections
        if synth_res.inferred_connections:
            ans_body_parts.append(
                f"""
                <div style="color: #94a3b8; font-size: 0.88rem; line-height: 1.5; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 0.8rem; margin-bottom: 1rem;">
                    <b style="color: #38bdf8;">Derived Interpretation / Inferences:</b><br>
                    {format_epistemic_html(synth_res.inferred_connections)}
                </div>
                """
            )

        # Citation Audit & Synthesis Provenance Footer
        audit = synth_res.citation_audit
        if audit.all_citations_valid:
            cited_str = (
                ", ".join(f"[Card {c}]" for c in audit.cited_cards)
                if audit.cited_cards
                else "None cited"
            )
            audit_badge = (
                f'<span style="color: #34d399; font-weight: 600;">&#10003; Citation Audit Passed</span> '
                f'<span style="color: #64748b;">({html.escape(cited_str)})</span>'
            )
        else:
            invalid_str = ", ".join(f"[Card {c}]" for c in audit.invalid_cards)
            audit_badge = (
                f'<span style="color: #f87171; font-weight: 600;">&#9888; Unverified Citations Detected: {html.escape(invalid_str)}</span>'
            )

        ans_footer_html = f"""
        <div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 0.75rem; margin-top: 1rem; font-size: 0.76rem; line-height: 1.5;">
            <div>{audit_badge}</div>
            <div style="color: #64748b;">Synthesis Engine: {html.escape(synth_res.model_name)}</div>
        </div>
        """

    st.markdown(
        f"""
        <div class="column-panel">
            <div>
                <div class="column-header">THE ANSWER</div>
                <div class="column-divider"></div>
                {"".join(ans_body_parts)}
            </div>
            {ans_footer_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# COLUMN 2: EVIDENCE
# ===========================================================================
with col_evid:
    evid_parts: list[str] = []

    if not ret_res or (not ret_res.semantic_results and not ret_res.lexical_report.any_present):
        evid_parts.append(
            """
            <div style="color: #64748b; font-size: 0.9rem; line-height: 1.6;">
                Passages retrieved directly from the source texts, with physical page numbers and provenance coordinates, will be displayed here.
            </div>
            """
        )
    else:
        for res in ret_res.semantic_results:
            c = res.chunk
            evid_parts.append(
                f"""
                <div class="evidence-block">
                    <div class="evidence-meta">
                        [Card {res.rank}] {html.escape(c.doc_title)} &bull; Page {c.page_number}
                    </div>
                    <div class="evidence-quote">
                        "{html.escape(c.text)}"
                    </div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.4rem;">
                        Offset: chars {c.char_start}&ndash;{c.char_end} &bull; Cosine Proximity: {res.similarity_score:.4f}
                    </div>
                </div>
                """
            )

        # Positive Deterministic Lexical Concordance Hits
        lex = ret_res.lexical_report
        present_terms = [r for r in lex.term_results.values() if r.present_in_corpus]
        if present_terms:
            lex_items: list[str] = []
            for tr in present_terms:
                sample_snip = (
                    f'<div style="color: #94a3b8; font-style: italic; font-size: 0.76rem; margin-top: 0.15rem;">'
                    f'e.g. ({html.escape(tr.matches[0].doc_title)}, p. {tr.matches[0].page_number}): "{html.escape(tr.matches[0].snippet)}"'
                    f"</div>"
                    if tr.matches
                    else ""
                )
                lex_items.append(
                    f"<li style='margin-bottom: 0.35rem;'>"
                    f"<b>'{html.escape(tr.term)}'</b>: {tr.total_occurrences} hit(s) across {tr.documents_count} doc(s)"
                    f"{sample_snip}</li>"
                )
            evid_parts.append(
                f"""
                <div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 0.8rem; margin-top: 0.4rem; font-size: 0.8rem; color: #cbd5e1;">
                    <b style="color: #38bdf8;">Deterministic Lexical Concordance (Exact Hits):</b>
                    <ul style="margin: 0.4rem 0 0 0; padding-left: 1.2rem;">
                        {"".join(lex_items)}
                    </ul>
                </div>
                """
            )

    st.markdown(
        f"""
        <div class="column-panel">
            <div>
                <div class="column-header">EVIDENCE</div>
                <div class="column-divider"></div>
                {"".join(evid_parts)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# COLUMN 3: WHAT THE CORPUS CANNOT ESTABLISH
# ===========================================================================
with col_limits:
    limits_parts: list[str] = []

    if not synth_res or not ret_res:
        limits_parts.append(
            """
            <div style="color: #64748b; font-size: 0.9rem; line-height: 1.6;">
                Epistemic boundaries, conflicting source accounts, and absent information will appear here to expose the limits of the available evidence.
            </div>
            """
        )
    else:
        # Divergent Perspectives / Conflicts
        if (
            synth_res.divergent_perspectives
            and "No conflicting" not in synth_res.divergent_perspectives
            and "No overt contradictory" not in synth_res.divergent_perspectives
        ):
            limits_parts.append(
                f"""
                <div style="background-color: rgba(239, 68, 68, 0.1); border-left: 3px solid #ef4444; padding: 0.7rem 0.9rem; border-radius: 6px; margin-bottom: 1rem;">
                    <b style="color: #f87171; font-size: 0.88rem;">&#9888;&#65039; Source Tensions / Competing Accounts:</b><br>
                    <div style="color: #fca5a5; font-size: 0.85rem; line-height: 1.5; margin-top: 0.3rem;">
                        {format_epistemic_html(synth_res.divergent_perspectives)}
                    </div>
                </div>
                """
            )

        # Epistemic Limitations & Gaps
        if synth_res.epistemic_limits:
            limits_parts.append(
                f"""
                <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5; margin-bottom: 1rem;">
                    <b style="color: #94a3b8;">Evidentiary Boundaries:</b><br>
                    {format_epistemic_html(synth_res.epistemic_limits)}
                </div>
                """
            )

        # Lexical Miss Report
        lex = ret_res.lexical_report
        misses = [t for t, r in lex.term_results.items() if not r.present_in_corpus]
        if misses:
            miss_items = "".join(
                f"<li><b>'{html.escape(m)}'</b>: Not found in corpus using lexical scan</li>"
                for m in misses
            )
            limits_parts.append(
                f"""
                <div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 0.8rem; font-size: 0.82rem; color: #94a3b8;">
                    <b style="color: #f59e0b;">Deterministic Search Limitations:</b>
                    <ul style="margin: 0.3rem 0; padding-left: 1.2rem;">
                        {miss_items}
                    </ul>
                    <span style="font-size: 0.75rem; color: #64748b;">
                        (Note: Word misses indicate search boundaries, not definitive conceptual absence).
                    </span>
                </div>
                """
            )

    st.markdown(
        f"""
        <div class="column-panel">
            <div>
                <div class="column-header">WHAT THE CORPUS CANNOT ESTABLISH</div>
                <div class="column-divider"></div>
                {"".join(limits_parts)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# FOOTER BAR
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-bar">
        <span>CORPUS INQUIRY LAB &nbsp;|&nbsp; A TOOL FOR CRITICAL ENGAGEMENT WITH TEXTS</span>
        <span>EXPLORE &bull; QUESTION &bull; REFLECT</span>
    </div>
    """,
    unsafe_allow_html=True,
)
