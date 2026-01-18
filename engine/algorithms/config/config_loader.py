"""Runtime JSON config loader with strict validation."""
from __future__ import annotations

import json
from pathlib import Path

from engine.algorithms.config.runtime_schema import RuntimeConfig, validate_runtime_config


class ConfigLoadError(ValueError):
    """Raised when loading runtime config fails."""


class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config: RuntimeConfig = self._load_config()

    def _load_config(self) -> RuntimeConfig:
        path = Path(self.config_path)
        if not path.exists():
            raise ConfigLoadError(f"Runtime config not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ConfigLoadError(f"Runtime config root must be an object: {path}")
        overrides = self._load_external_inference_settings(path)
        if overrides:
            raw.setdefault("inference", {})
            if not isinstance(raw["inference"], dict):
                raise ConfigLoadError("inference must be an object when using inference_settings.json")
            raw["inference"] = _deep_merge(raw["inference"], overrides)
        return validate_runtime_config(raw)

    def _load_external_inference_settings(self, path: Path) -> dict:
        external_path = path.parent / "inference_settings.json"
        if not external_path.is_file():
            return {}
        with external_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ConfigLoadError("inference_settings.json must be an object")
        return data

    def get_config(self) -> dict:
        return self.config.model_dump(by_alias=True)

    def get_config_path(self) -> str:
        return self.config_path

    def get_inference_settings(self, algorithm: str | None = None) -> dict:
        base = {
            "max_delta": self.config.inference.max_delta,
            "image_obs_every": self.config.inference.image_obs_every,
        }
        if algorithm == "sequential":
            seq = self.config.inference.sequential
            if seq is None:
                raise ConfigLoadError("inference.sequential is required for sequential inference")
            base.update(
                {
                    "max_timesteps": seq.max_timesteps,
                    "temporal_ensemble": seq.temporal_ensemble,
                    "esb_k": seq.esb_k,
                    "policy_update_period": seq.policy_update_period,
                }
            )
        return base

    def get_camera_names(self) -> list[str]:
        return list(self.config.camera_names)

    def get_image_resize(self) -> tuple[int, int]:
        return (self.config.image_resize.width, self.config.image_resize.height)

    def get_observation_keys(self) -> list[str]:
        obs_keys: list[str] = []
        for topic in self.config.topics.values():
            for data_key in topic.fields.keys():
                if data_key.startswith("/observation/") and not data_key.startswith("/observation/images/"):
                    obs_keys.append(data_key)
        return obs_keys

    def get_observation_field_config(self, data_key: str):
        for topic in self.config.topics.values():
            if data_key in topic.fields:
                rule = topic.fields[data_key].root
                if isinstance(rule, str):
                    return rule
                return {
                    "slice": list(rule.slice),
                    "attr": rule.attr,
                }
        return None


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
