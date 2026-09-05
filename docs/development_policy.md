# Development Policy

Engineering rules that apply across all phases of this project (setup,
MVP, research extensions). Referenced throughout the codebase (comments,
docstrings, `configs/models.yaml`) as the source of the project's
no-fabrication / reproducibility / scope conventions.

## Environment

- Always use the project-local `.venv` (`.venv\Scripts\python.exe`). Never
  use a global `python`/`py` command without first confirming it resolves to
  this project's virtual environment.
- Do not create, modify, or depend on any virtual environment outside this
  project directory.
- Target Python 3.12 for compatibility with PyTorch/torchaudio/Transformers.
- **The production/deployment target is CPU-only.** Install `torch`/
  `torchaudio` from the official CPU wheel index
  (`https://download.pytorch.org/whl/cpu`), never a CUDA build. Do not
  optimize for or depend on a development machine's GPU. All code must
  support `device="cpu"` and `device="auto"`, and `"auto"` must always
  safely fall back to CPU — there is no code path in this project that
  requires CUDA.

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
- **Never trust a model card's prose alone for architecture or label
  mapping.** Verify against the actual cited source repository/code where
  one exists. If a cited source repository cannot be found or does not
  exist, record that as a documented finding — do not invent an
  architecture merely to make a checkpoint's `state_dict` load.
- Clearly separate author-reported/model-card metrics from metrics actually
  measured by this project. Never present a model card's EER as if this
  project reproduced it, and never present a prediction on synthetic smoke
  audio as accuracy evidence.

## Code Organisation

- Keep ML inference logic separate from UI code. Inference/model code lives
  under `src/audio_deepfake_detector/inference/` and `.../models/`; UI code
  (e.g. Streamlit) lives under `app/`. The UI layer should call into the
  library, not contain model logic itself.
- Use modular model adapters under `src/audio_deepfake_detector/models/` so
  new detection approaches can be added and swapped without changing
  inference or evaluation code. Adapters implement the
  `BaseDeepfakeDetector` interface (`models/base.py`) and are resolved via
  `models/registry.py`, which reads `configs/models.yaml`. No repository
  name or checkpoint filename should be hardcoded anywhere else.
- Shared audio preprocessing (`preprocessing/audio_loader.py`) must not
  crop/pad audio to a fixed length — that is model-specific and belongs in
  each adapter (using the shared `preprocessing/windowing.py` primitive).
- Add tests under `tests/` for every meaningful implementation phase (not
  just at the end of the project). Run the test suite before committing.
  Mark tests that download real model checkpoints or run real inference
  with `@pytest.mark.integration` / `@pytest.mark.slow`; the default suite
  (`pytest -m "not integration and not slow"`) must never trigger a large
  download.

## Git Workflow

- Inspect `git status` before making changes and again after, to confirm
  only intended files were modified/created.
- Use logical, scoped commits with clear messages describing *why* a change
  was made, not just what changed.
- Avoid destructive commands (`git reset --hard`, `git clean -f`, force
  pushes, deleting branches) unless explicitly required.
- Do not commit `.env`, credentials, API keys, or other secrets.

## Scope Discipline

- Do not skip ahead of the current phase (e.g. do not start model
  integration, dataset downloads, or Streamlit implementation while still in
  environment-setup phase) unless explicitly required by the task at hand.
