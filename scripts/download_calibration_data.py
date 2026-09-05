"""Download a small, labeled calibration dataset for Phase 4 inference
calibration (docs/inference_calibration.md).

Downloads TWO calibration sets:

1. In-domain: a balanced bonafide/spoof subset of the ASVspoof 2019 LA
   *development* partition (never the training partition — see
   docs/inference_calibration.md "Data Separation"), sourced from the
   public Hugging Face mirror `Bisher/ASVspoof_2019_LA` (split="validation",
   which corresponds to the official ASVspoof2019 LA "dev" protocol).

2. Out-of-domain / real-world sanity set:
   - genuine speech: `openslr/librispeech_asr`, split="test.clean"
   - synthetic/spoof speech: `ajaykarthick/wavefake-audio` (WaveFake,
     LJSpeech-based multi-vocoder synthetic speech), split="train"
     (WaveFake ships no train/test split of its own — "train" is simply the
     HF repo's only split name)

This is a RESEARCH/CALIBRATION-ONLY script. It requires the `calibration`
extra (`pip install -e ".[calibration]"`) and network access. It never
writes into a path tracked by Git — everything lands under
`data/raw/calibration/`, which `.gitignore` already excludes
(`data/raw/*`).

Usage:
    .venv\\Scripts\\python.exe scripts\\download_calibration_data.py \
        --in-domain-per-class 100 --ood-per-class 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = REPO_ROOT / "data" / "raw" / "calibration"

MAX_OOD_GENUINE_DURATION_S = 20.0  # bound window count for CPU inference


def _write_wav(waveform: np.ndarray, sample_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, sample_rate, subtype="PCM_16")


def _dataset_revision(repo_id: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id)
    return info.sha


def download_in_domain(per_class: int) -> list[dict]:
    """Bisher/ASVspoof_2019_LA, split=validation (== ASVspoof2019 LA dev).
    key: 0 = bonafide, 1 = spoof."""
    from datasets import load_dataset

    repo_id = "Bisher/ASVspoof_2019_LA"
    split = "validation"
    revision = _dataset_revision(repo_id)
    out_dir = CALIBRATION_DIR / "in_domain"

    print(f"[in-domain] streaming {repo_id}@{revision} split={split} ...")
    ds = load_dataset(repo_id, split=split, streaming=True)

    counts = {"bonafide": 0, "spoof": 0}
    manifest: list[dict] = []
    label_names = {0: "bonafide", 1: "spoof"}

    t0 = time.time()
    for ex in ds:
        if counts["bonafide"] >= per_class and counts["spoof"] >= per_class:
            break
        label = label_names[int(ex["key"])]
        if counts[label] >= per_class:
            continue
        audio = ex["audio"]
        rel_path = Path("in_domain") / label / f"{ex['audio_file_name']}.wav"
        _write_wav(np.asarray(audio["array"], dtype=np.float32), audio["sampling_rate"], CALIBRATION_DIR / rel_path)
        manifest.append(
            {
                "path": str(rel_path).replace("\\", "/"),
                "label": label,
                "dataset": repo_id,
                "dataset_revision": revision,
                "split": split,
                "source_id": ex["audio_file_name"],
                "domain": "in_domain",
            }
        )
        counts[label] += 1

    print(f"[in-domain] done: {counts} in {time.time() - t0:.1f}s")
    return manifest


def download_ood_genuine(n: int) -> list[dict]:
    """openslr/librispeech_asr, split=test.clean — known genuine human speech,
    a different corpus/domain than ASVspoof (real-world sanity check)."""
    from datasets import load_dataset

    repo_id = "openslr/librispeech_asr"
    split = "test.clean"
    revision = _dataset_revision(repo_id)
    out_dir = CALIBRATION_DIR / "out_of_domain" / "genuine"

    print(f"[ood-genuine] streaming {repo_id}@{revision} split={split} ...")
    ds = load_dataset(repo_id, split=split, streaming=True)

    manifest: list[dict] = []
    t0 = time.time()
    for ex in ds:
        if len(manifest) >= n:
            break
        audio = ex["audio"]
        duration = len(audio["array"]) / audio["sampling_rate"]
        if duration > MAX_OOD_GENUINE_DURATION_S:
            continue
        rel_path = Path("out_of_domain") / "genuine" / f"{ex['id']}.wav"
        _write_wav(np.asarray(audio["array"], dtype=np.float32), audio["sampling_rate"], CALIBRATION_DIR / rel_path)
        manifest.append(
            {
                "path": str(rel_path).replace("\\", "/"),
                "label": "bonafide",
                "dataset": repo_id,
                "dataset_revision": revision,
                "split": split,
                "source_id": ex["id"],
                "domain": "out_of_domain",
            }
        )

    print(f"[ood-genuine] done: {len(manifest)} in {time.time() - t0:.1f}s")
    return manifest


def download_ood_spoof(n: int) -> list[dict]:
    """ajaykarthick/wavefake-audio, split=train — WaveFake synthetic speech
    (multiple neural vocoders over LJSpeech utterances), a different corpus
    and synthesis family than ASVspoof2019 LA."""
    from datasets import load_dataset

    repo_id = "ajaykarthick/wavefake-audio"
    split = "train"
    revision = _dataset_revision(repo_id)

    print(f"[ood-spoof] streaming {repo_id}@{revision} split={split} ...")
    ds = load_dataset(repo_id, split=split, streaming=True)

    manifest: list[dict] = []
    seen_source_utterances: set[str] = set()
    t0 = time.time()
    for ex in ds:
        if len(manifest) >= n:
            break
        # Avoid many near-duplicate vocoder renderings of the same source
        # utterance dominating a small calibration set.
        if ex["audio_id"] in seen_source_utterances:
            continue
        audio = ex["audio"]
        rel_path = Path("out_of_domain") / "spoof" / f"{ex['audio_id']}_{ex['real_or_fake']}.wav"
        _write_wav(np.asarray(audio["array"], dtype=np.float32), audio["sampling_rate"], CALIBRATION_DIR / rel_path)
        manifest.append(
            {
                "path": str(rel_path).replace("\\", "/"),
                "label": "spoof",
                "dataset": repo_id,
                "dataset_revision": revision,
                "split": split,
                "source_id": f"{ex['audio_id']}_{ex['real_or_fake']}",
                "domain": "out_of_domain",
            }
        )
        seen_source_utterances.add(ex["audio_id"])

    print(f"[ood-spoof] done: {len(manifest)} in {time.time() - t0:.1f}s")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-domain-per-class", type=int, default=100)
    parser.add_argument("--ood-per-class", type=int, default=60)
    args = parser.parse_args()

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    manifest += download_in_domain(args.in_domain_per_class)
    manifest += download_ood_genuine(args.ood_per_class)
    manifest += download_ood_spoof(args.ood_per_class)

    manifest_path = CALIBRATION_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote manifest with {len(manifest)} entries to {manifest_path}")
    print("This directory is gitignored (data/raw/*) — no audio is committed.")


if __name__ == "__main__":
    main()
