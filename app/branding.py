"""Public-facing project branding and identifier display rules.

The frozen model artifact is hosted under a Hugging Face repository id
that includes a personal account name (see `configs/models.yaml`). That
identifier is required, unmodified, for the application to actually
download and verify the model -- renaming it here would not migrate the
artifact and would break deployment, so it is intentionally left alone in
configuration and in any code path that downloads/verifies the model.

This module only controls what is *displayed* to a user: the personal
hosting-account identifier is never shown in the UI, PDF/HTML/JSON
reports, or documentation -- a neutral technical label is shown instead.
"""

from __future__ import annotations

PROJECT_AUTHOR = "Syed Raza Haider Zaidi"

NEUTRAL_MODEL_ARTIFACT_LABEL = "Spectra-AASIST3 INT8 deployment artifact"


def display_model_artifact(repository: str | None) -> str:
    """Neutral, user-facing label for the model artifact. `repository`
    (the real Hugging Face repo id) continues to be used unmodified
    everywhere the app actually needs it to function -- only the
    displayed text is neutralized here."""
    if not repository:
        return "Not available"
    return NEUTRAL_MODEL_ARTIFACT_LABEL
