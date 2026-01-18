"""Strict runtime JSON schema for inference pipelines."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError, field_validator, model_validator

from engine.config.errors import ConfigError, ConfigValidationIssue


AllowedMsgType = Literal[
    "PoseStamped",
    "JointState",
    "Float32MultiArray",
    "Bool",
    "Int32",
    "Float32",
    "Int64",
    "Float64",
    "String",
]


class ImageResize(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int
    height: int

    @field_validator("width", "height")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v


class SliceRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slice: tuple[int, int]
    attr: Literal["position", "effort", "velocity", "data"] = "data"

    @field_validator("slice")
    @classmethod
    def _validate_slice(cls, v: tuple[int, int]) -> tuple[int, int]:
        if len(v) != 2:
            raise ValueError("slice must have exactly two entries")
        start, end = v
        if start < 0 or end <= start:
            raise ValueError("slice must be [start, end] with 0 <= start < end")
        return v


class FieldRule(
    RootModel[SliceRule | Literal["pose.position", "pose.orientation", "data"]]
):
    """Field rule is either a string alias or a slice rule."""

    @model_validator(mode="after")
    def _validate_rule(self) -> "FieldRule":
        if isinstance(self.root, str) and self.root not in {
            "pose.position",
            "pose.orientation",
            "data",
        }:
            raise ValueError(f"Unsupported field rule: {self.root}")
        return self


class TopicConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str
    msg_type: AllowedMsgType
    fields: dict[str, FieldRule]

    @field_validator("topic")
    @classmethod
    def _topic_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("topic must be a non-empty string")
        return v

    @model_validator(mode="after")
    def _validate_field_keys(self) -> "TopicConfig":
        for key in self.fields.keys():
            if not key.startswith("/observation/"):
                raise ValueError(f"field key must start with /observation/: {key}")
            if key.startswith("/observation/images/"):
                raise ValueError(f"image observations are not allowed in topics: {key}")
        return self


class SequentialInferenceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_timesteps: int
    temporal_ensemble: bool = False
    esb_k: float = 0.01
    policy_update_period: int = 1

    @field_validator("max_timesteps", "policy_update_period")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v


class InferenceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_delta: float = 10.0
    image_obs_every: int = 1
    sequential: SequentialInferenceSettings | None = None

    @field_validator("image_obs_every")
    @classmethod
    def _image_obs_every_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("image_obs_every must be > 0")
        return v


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robot_id: str
    HZ: int = Field(alias="HZ")
    image_resize: ImageResize
    camera_names: list[str]
    topics: dict[str, TopicConfig]
    inference: InferenceSettings = Field(default_factory=InferenceSettings)

    @field_validator("robot_id")
    @classmethod
    def _robot_id_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("robot_id must be a non-empty string")
        return v

    @field_validator("HZ")
    @classmethod
    def _hz_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("HZ must be > 0")
        return v

    @field_validator("camera_names")
    @classmethod
    def _camera_names_valid(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("camera_names must contain at least one entry")
        for name in v:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("camera_names entries must be non-empty strings")
        return v


def validate_runtime_config(raw: dict) -> RuntimeConfig:
    try:
        return RuntimeConfig.model_validate(raw)
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
