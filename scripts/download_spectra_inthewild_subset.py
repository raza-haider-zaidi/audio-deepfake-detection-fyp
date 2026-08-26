"""Download a deterministic, balanced subset of SpeechAntiSpoofingBenchmarks/InTheWild
for the Spectra-AASIST3 evaluation (docs/spectra_aasist3_evaluation.md).

This is the SAME canonical "In-the-Wild" corpus (58 speakers, real-world
audio) already used as a valid OOD domain for `antideepfake_wav2vec2_small`
(Phase 5), sourced here from the Arena's own parquet mirror, pinned to the
exact revision (`a957f2582802cdb5964e118818c2e46b3d61aa35`) their own
InTheWild result.yaml for Spectra-AASIST3 was computed against, so our
subset is drawn from literally the same corpus snapshot the model-card
number cites (though our small subset's own EER is NOT expected to match
that number -- see the evaluation docs for why).

Label convention (verified directly from the dataset's own `notes` field,
not assumed): `label == 1` => spoof, `label == 0` => bonafide.

Deterministic sampling: numpy Generator(seed=1337) shuffles the full
utterance_id list per class, then takes the first 200 bonafide + 200 spoof
IDs. Exact sample IDs are recorded in the manifest for reproducibility.

RESEARCH-ONLY script. Writes audio under `data/raw/spectra_inthewild/`,
which is gitignored (`data/raw/*`) -- no audio is committed.

Usage:
    .venv\\Scripts\\python.exe scripts\\download_spectra_inthewild_subset.py
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "raw" / "spectra_inthewild"
CACHE_DIR = REPO_ROOT / "models" / "cache"

DATASET_REPO = "SpeechAntiSpoofingBenchmarks/InTheWild"
DATASET_REVISION = "a957f2582802cdb5964e118818c2e46b3d61aa35"
N_SHARDS = 8
PER_CLASS_TARGET = 200
SEED = 1337


def _load_labels() -> dict[str, dict]:
    path = hf_hub_download(
        repo_id=DATASET_REPO, filename="data/labels.parquet", repo_type="dataset",
        revision=DATASET_REVISION, cache_dir=str(CACHE_DIR),
    )
    table = pq.read_table(path)
    by_utt = {}
    for row in table.to_pylist():
        by_utt[row["utterance_id"]] = row
    return by_utt


def _select_ids(rng: np.random.Generator) -> tuple[set[str], set[str]]:
    """Deterministically select PER_CLASS_TARGET utterance_ids per class
    from the label parquet alone (cheap: no audio download needed here)."""
    labels = _load_labels()
    bonafide_ids = sorted(uid for uid, r in labels.items() if r["label"] == 0)
    spoof_ids = sorted(uid for uid, r in labels.items() if r["label"] == 1)
    rng.shuffle(bonafide_ids)
    rng.shuffle(spoof_ids)
    return set(bonafide_ids[:PER_CLASS_TARGET]), set(spoof_ids[:PER_CLASS_TARGET])


def main() -> None:
    rng = np.random.default_rng(SEED)
    wanted_bonafide, wanted_spoof = _select_ids(rng)
    wanted = wanted_bonafide | wanted_spoof
    print(f"Target: {len(wanted_bonafide)} bonafide + {len(wanted_spoof)} spoof = {len(wanted)} utterance_ids")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    found_bonafide = 0
    found_spoof = 0
    t0 = time.time()

    for shard in range(N_SHARDS):
        if len(manifest) >= len(wanted):
            break
        fname = f"data/test-{shard:05d}-of-{N_SHARDS:05d}.parquet"
        path = hf_hub_download(
            repo_id=DATASET_REPO, filename=fname, repo_type="dataset",
            revision=DATASET_REVISION, cache_dir=str(CACHE_DIR),
        )
        table = pq.read_table(path)
        print(f"[shard {shard}] {table.num_rows} rows, {time.time()-t0:.1f}s elapsed")
        for row in table.to_pylist():
            notes = json.loads(row["notes"])
            utt_id = notes["utterance_id"]
            if utt_id not in wanted:
                continue
            label = "spoof" if row["label"] == 1 else "bonafide"
            audio_bytes = row["audio"]["bytes"]
            waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            rel_path = Path(label) / f"{utt_id}.wav"
            out_path = OUT_DIR / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), waveform, sr, subtype="PCM_16")
            manifest.append(
                {
                    "utterance_id": utt_id,
                    "path": str(rel_path).replace("\\", "/"),
                    "label": label,
                    "speaker": notes.get("speaker"),
                    "duration_seconds": len(waveform) / sr,
                    "sample_rate": sr,
                    "dataset": DATASET_REPO,
                    "dataset_revision": DATASET_REVISION,
                }
            )
            if label == "bonafide":
                found_bonafide += 1
            else:
                found_spoof += 1

    print(f"Found {found_bonafide} bonafide, {found_spoof} spoof in {time.time()-t0:.1f}s")

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest with {len(manifest)} entries to {manifest_path}")
    print("data/raw/spectra_inthewild/ is gitignored -- no audio is committed.")


if __name__ == "__main__":
    main()
