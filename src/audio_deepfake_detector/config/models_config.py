"""Loader for configs/models.yaml.

Centralizes model repository IDs, checkpoint filenames, and label mappings
so they are never hardcoded inside adapters or the inference service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "models.yaml"


@dataclass
class ModelConfig:
    id: str
    repository: str
    revision: str
    enabled: bool
    architecture: str
    base_model: str
    sample_rate: int
    window_seconds: float | None
    checkpoint_filename: str
    checkpoint_size_bytes: int | None
    label_mapping: dict[int, str]
    license: str
    training_dataset: str
    source_repository: str | None
    source_repository_status: str | None
    deployment_role: str
    notes: str = ""
    window_hop_samples: int | None = None
    window_samples: int | None = None
    expected_sha256: str | None = None


@dataclass
class ModelsRegistryConfig:
    models: dict[str, ModelConfig] = field(default_factory=dict)

    def enabled_models(self) -> dict[str, ModelConfig]:
        return {k: v for k, v in self.models.items() if v.enabled}

    def get(self, model_id: str) -> ModelConfig:
        if model_id not in self.models:
            raise KeyError(
                f"Unknown model id '{model_id}'. Known ids: {sorted(self.models)}"
            )
        return self.models[model_id]


def load_models_config(path: Path | str = DEFAULT_CONFIG_PATH) -> ModelsRegistryConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or "models" not in raw:
        raise ValueError(f"Model config at {path} is missing a top-level 'models' key")

    models: dict[str, ModelConfig] = {}
    for model_id, entry in raw["models"].items():
        label_mapping = {int(k): v for k, v in (entry.get("label_mapping") or {}).items()}
        models[model_id] = ModelConfig(
            id=entry["id"],
            repository=entry["repository"],
            revision=entry["revision"],
            enabled=bool(entry.get("enabled", False)),
            architecture=entry.get("architecture", "Not documented"),
            base_model=entry.get("base_model", "Not documented"),
            sample_rate=int(entry.get("sample_rate", 16000)),
            window_seconds=entry.get("window_seconds"),
            checkpoint_filename=entry.get("checkpoint_filename", "Not documented"),
            checkpoint_size_bytes=entry.get("checkpoint_size_bytes"),
            label_mapping=label_mapping,
            license=entry.get("license", "Not documented"),
            training_dataset=entry.get("training_dataset", "Not documented"),
            source_repository=entry.get("source_repository"),
            source_repository_status=entry.get("source_repository_status"),
            deployment_role=entry.get("deployment_role", "Not documented"),
            notes=(entry.get("notes") or "").strip(),
            window_hop_samples=entry.get("window_hop_samples"),
            window_samples=entry.get("window_samples"),
            expected_sha256=entry.get("expected_sha256"),
        )

    return ModelsRegistryConfig(models=models)
