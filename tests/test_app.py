"""Unit and smoke tests for the Phase 4 Streamlit application interface."""

import pytest
from src.config import PREPARED_CORPORA_DIR
from src.ingest import list_prepared_corpora, purge_session_uploads


def test_prepared_corpora_discovery():
    corpora = list_prepared_corpora()
    assert len(corpora) >= 2
    ids = [c["id"] for c in corpora]
    assert "sample_archaeological_survey" in ids
    assert "sample_epigraphic_register" in ids


def test_app_imports_cleanly():
    # Verify that app module and its dependencies import without error
    import src.app
    assert hasattr(src.app, "reset_experiment")


def test_format_epistemic_html_escapes_epigraphic_brackets_and_formats_markdown():
    from src.app import format_epistemic_html

    raw = 'In **coastal_inscriptions.txt** (p. 1): "To <Neptune Augustus>, sacred dedication" [Card 1]'
    rendered = format_epistemic_html(raw)

    # Epigraphic angle brackets must be escaped so browser doesn't strip them
    assert "&lt;Neptune Augustus&gt;" in rendered
    assert "<Neptune Augustus>" not in rendered
    # Markdown bold must be converted to HTML <b> tags
    assert "<b>coastal_inscriptions.txt</b>" in rendered
    assert "**coastal_inscriptions.txt**" not in rendered
    # Citations must be styled
    assert '<span class="citation-badge">[Card 1]</span>' in rendered
