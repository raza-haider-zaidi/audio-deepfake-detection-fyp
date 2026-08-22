"""Device resolution.

"auto" always safely falls back to CPU-only in this project. Phase 2 treats
the production deployment target (Streamlit Cloud) as CPU-only regardless
of what hardware the development machine has.
"""

from __future__ import annotations


def resolve_device(device: str = "cpu") -> str:
    """Resolve a requested device string to an actual torch device string.

    Supported inputs: "cpu", "auto". Anything else (e.g. "cuda") is
    rejected — this project's production path has no GPU dependency.
    """
    if device not in ("cpu", "auto"):
        raise ValueError(
            f"Unsupported device '{device}'. Only 'cpu' and 'auto' are supported; "
            "this project's deployment target has no GPU dependency."
        )
    # "auto" resolves to CPU by design in this phase — see CLAUDE.md /
    # docs/architecture.md. We deliberately do not probe torch.cuda here to
    # keep behavior identical regardless of the development machine's GPU.
    return "cpu"
