# CLAUDE.md — Development Rules for This Repository

Permanent rules for any Claude Code session working in this repository.
These rules apply across all phases of the project (setup, MVP, research
extensions) unless the user explicitly overrides one for a specific task.

## Environment

- Always use the project-local `.venv` (`.venv\Scripts\python.exe`). Never
  use a global `python`/`py` command without first confirming it resolves to
  this project's virtual environment — the global `python` command may
  resolve to an unrelated project's environment on this machine.
- Do not create, modify, or depend on any virtual environment outside this
  project directory.
- Target Python 3.12 for compatibility with PyTorch/torchaudio/Transformers.

## Data and Models

- **Never commit datasets or large model checkpoints.** Anything under
  `data/raw/`, `data/processed/`, `models/checkpoints/`, `models/cache/`, or
  matching audio/weight file extensions is gitignored — keep it that way.
- Do not fabricate evaluation results, metrics, or benchmark numbers. If a
  model hasn't been run or a dataset hasn't been evaluated, say so explicitly
  rather than inventing plausible-looking numbers.
- Always record the **actual** model name, model revision/commit hash, and
  dataset version used for any experiment or result — never a generic or
  assumed identifier.
- Preserve reproducibility: pin dependency versions, record random seeds,
  and document exact commands used to produce any result in `results/`.

## Code Organisation

- Keep ML inference logic separate from UI code. Inference/model code lives
  under `src/audio_deepfake_detector/inference/` and `.../models/`; UI code
  (e.g. Streamlit) lives under `app/`. The UI layer should call into the
  library, not contain model logic itself.
- Use modular model adapters under `src/audio_deepfake_detector/models/` so
  new detection approaches can be added and swapped without changing
  inference or evaluation code.
- Add tests under `tests/` for every meaningful implementation phase (not
  just at the end of the project). Run the test suite before committing.

## Git Workflow

- Inspect `git status` before making changes and again after, to confirm
  only intended files were modified/created.
- Use logical, scoped commits with clear messages describing *why* a change
  was made, not just what changed.
- Avoid destructive commands (`git reset --hard`, `git clean -f`, force
  pushes, deleting branches) unless explicitly requested by the user.
- Do not commit `.env`, credentials, API keys, or other secrets.

## Scope Discipline

- Do not skip ahead of the current phase (e.g. do not start model
  integration, dataset downloads, or Streamlit implementation while still in
  environment-setup phase) unless explicitly instructed.
