"""User-facing error type for the Streamlit layer.

Lets the UI show a short, friendly message while technical details (the
original exception) stay available for internal logging only.
"""

from __future__ import annotations


class UserFacingError(Exception):
    """An error with a message safe to show directly to an end user."""

    def __init__(self, friendly_message: str, technical_detail: str | None = None):
        super().__init__(friendly_message)
        self.friendly_message = friendly_message
        self.technical_detail = technical_detail
