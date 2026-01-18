"""Policy interfaces for inference."""
from __future__ import annotations

from typing import Protocol

import torch

from engine.config.inference_schemas import PolicyParams


class InferencePolicy(Protocol):
    state_dim: int
    action_dim: int
    num_queries: int
    num_robot_observations: int
    num_image_observations: int
    image_observation_skip: int
    camera_names: list[str]
    stats_eps: float

    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor

    def __call__(self, robot_history: torch.Tensor, cam_images: torch.Tensor) -> torch.Tensor: ...

    def encode_vision(self, cam_images: torch.Tensor) -> torch.Tensor: ...

    @property
    def body(self) -> torch.nn.Module: ...

    def freeze_all_model_params(self) -> None: ...

    @property
    def normalization_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]: ...


class PolicyBuilder(Protocol):
    def build_policy(
        self,
        components: dict[str, torch.nn.Module],
        params: PolicyParams,
        device: torch.device | str | None = None,
    ) -> InferencePolicy: ...
