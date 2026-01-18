"""Strict schemas for inference YAML configuration."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from engine.config.errors import ConfigError, ConfigValidationIssue


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component_config_paths: dict[str, str]
    component_checkpoint_paths: dict[str, str] | None = None

    @field_validator("component_config_paths")
    @classmethod
    def _component_paths_valid(cls, v: dict[str, str]) -> dict[str, str]:
        if not v:
            raise ValueError("component_config_paths must contain at least one entry")
        for key, path in v.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("component_config_paths keys must be non-empty strings")
            if not isinstance(path, str) or not path.strip():
                raise ValueError("component_config_paths values must be non-empty strings")
        return v

    @field_validator("component_checkpoint_paths")
    @classmethod
    def _checkpoint_paths_valid(cls, v: dict[str, str] | None, info) -> dict[str, str] | None:
        if v is None:
            return v
        model_cfg = info.data.get("component_config_paths", {})
        for key, path in v.items():
            if key not in model_cfg:
                raise ValueError(f"component_checkpoint_paths has unknown key: {key}")
            if not isinstance(path, str) or not path.strip():
                raise ValueError("component_checkpoint_paths values must be non-empty strings")
        return v


class PolicyParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    state_dim: int
    action_dim: int
    num_queries: int
    num_robot_observations: int
    num_image_observations: int
    image_observation_skip: int = 1
    camera_names: list[str]
    stats_path: str
    stats_eps: float = 1.0e-2

    @field_validator(
        "state_dim",
        "action_dim",
        "num_queries",
        "num_robot_observations",
        "num_image_observations",
        "image_observation_skip",
    )
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    @field_validator("camera_names")
    @classmethod
    def _camera_names_valid(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("camera_names must be a non-empty list")
        cleaned = [name.strip() for name in v]
        if any(not name for name in cleaned):
            raise ValueError("camera_names entries must be non-empty strings")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("camera_names must be unique")
        return cleaned

    @field_validator("stats_path")
    @classmethod
    def _stats_path_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("stats_path must be a non-empty string")
        return v

    @field_validator("stats_eps")
    @classmethod
    def _stats_eps_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("stats_eps must be > 0")
        return v


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    params: PolicyParams

    @field_validator("type")
    @classmethod
    def _type_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("type must be a non-empty string")
        return v


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["sequential", "real_time_action_chunking"]
    runtime_config_path: str
    checkpoint_path: str
    model: ModelConfig
    policy: PolicyConfig
    plugins: list[str] = Field(default_factory=list)
    hz: int | None = None

    @field_validator("runtime_config_path", "checkpoint_path")
    @classmethod
    def _path_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be a non-empty string")
        return v

    @field_validator("hz")
    @classmethod
    def _hz_positive(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("hz must be > 0")
        return v

    @field_validator("plugins")
    @classmethod
    def _plugins_valid(cls, v: list[str]) -> list[str]:
        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("plugins entries must be non-empty strings")
        return v


def validate_inference_config(raw: dict) -> InferenceConfig:
    try:
        return InferenceConfig.model_validate(raw)
    except ValidationError as exc:
        issues = []
        for err in exc.errors():
            path = _loc_to_path(err.get("loc", []))
            issues.append(
                ConfigValidationIssue(
                    error_path=path or "<root>",
                    error_message=err.get("msg", "Invalid value"),
                )
            )
        raise ConfigError(issues) from exc


def _loc_to_path(loc: tuple | list) -> str:
    path = ""
    for item in loc:
        if isinstance(item, int):
            path += f"[{item}]"
        else:
            if path:
                path += "."
            path += str(item)
    return path
