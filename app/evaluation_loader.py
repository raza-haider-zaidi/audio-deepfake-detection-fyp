"""Loads the committed, curated evaluation-data summary for the Evaluation
and Model Development pages.

This intentionally reads ONLY app/data/evaluation_summary.json -- a small,
git-committed file whose every value was copied verbatim from this
project's actual (gitignored) results/metrics/*.json outputs, with
provenance recorded in the file itself (see its "_provenance" key). It
never fabricates a metric: if a key is missing, callers must treat that
section as unavailable and omit the corresponding UI element rather than
inventing a placeholder value.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SUMMARY_PATH = Path(__file__).resolve().parent / "data" / "evaluation_summary.json"


def load_evaluation_summary() -> dict[str, Any] | None:
    """Returns the parsed summary dict, or None if the file is missing or
    unreadable -- callers MUST handle None by omitting the relevant
    section, never by fabricating placeholder metrics."""
    try:
        with _SUMMARY_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def get_section(summary: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
    """Safe accessor: returns summary[key] if both summary and the key
    exist, else None."""
    if summary is None:
        return None
    return summary.get(key)
