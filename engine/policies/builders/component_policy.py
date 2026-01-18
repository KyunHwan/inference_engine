"""Default policy builder that wraps a single policy component."""
from __future__ import annotations

import torch

from engine.config.inference_schemas import PolicyParams
from engine.policies.interfaces import InferencePolicy, PolicyBuilder
from engine.policies.normalization import load_stats


def _assert_no_extra_params(params: PolicyParams, allowed_extras: set[str]) -> None:
    extras = set(params.model_extra.keys()) if params.model_extra else set()
    unknown = extras - allowed_extras
    if unknown:
        raise ValueError(f"Unsupported policy params: {sorted(unknown)}")


class ComponentPolicy(torch.nn.Module):
    def __init__(
        self,
        policy_module: torch.nn.Module,
        params: PolicyParams,
        normalization_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float],
    ) -> None:
        super().__init__()
        self._policy = policy_module
        self.state_dim = params.state_dim
        self.action_dim = params.action_dim
        self.num_queries = params.num_queries
        self.num_robot_observations = params.num_robot_observations
        self.num_image_observations = params.num_image_observations
        self.image_observation_skip = params.image_observation_skip
        self.camera_names = list(params.camera_names)
        self._sm, self._ss, self._am, self._asd, self.stats_eps = normalization_tensors

    def forward(self, robot_history: torch.Tensor, cam_images: torch.Tensor) -> torch.Tensor:
        return self._policy(robot_history, cam_images)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._policy, name)

    def encode_vision(self, cam_images: torch.Tensor) -> torch.Tensor:
        if not hasattr(self._policy, "encode_vision"):
            raise AttributeError("Policy does not implement encode_vision")
        return self._policy.encode_vision(cam_images)

    @property
    def body(self) -> torch.nn.Module:
        if not hasattr(self._policy, "body"):
            raise AttributeError("Policy does not expose body")
        return self._policy.body

    def freeze_all_model_params(self) -> None:
        if hasattr(self._policy, "freeze_all_model_params"):
            self._policy.freeze_all_model_params()
            return
        for param in self._policy.parameters():
            param.requires_grad_(False)
        self._policy.eval()

    @property
    def normalization_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        return self._sm, self._ss, self._am, self._asd, self.stats_eps


class ComponentPolicyBuilder(PolicyBuilder):
    def build_policy(
        self,
        components: dict[str, torch.nn.Module],
        params: PolicyParams,
        device: torch.device | str | None = None,
    ) -> InferencePolicy:
        _assert_no_extra_params(params, allowed_extras=set())

        policy_module = _resolve_policy_component(components)
        normalization = load_stats(
            params.stats_path,
            params.state_dim,
            params.action_dim,
            device,
            params.stats_eps,
        )
        policy = ComponentPolicy(policy_module, params, normalization)
        if device is not None:
            policy.to(device)
        return policy


def _resolve_policy_component(components: dict[str, torch.nn.Module]) -> torch.nn.Module:
    if not components:
        raise ValueError("No components provided to policy builder")
    if "policy" in components:
        policy = components["policy"]
        if not isinstance(policy, torch.nn.Module):
            raise TypeError("Policy component must be a torch.nn.Module")
        return policy
    if len(components) == 1:
        policy = next(iter(components.values()))
        if not isinstance(policy, torch.nn.Module):
            raise TypeError("Policy component must be a torch.nn.Module")
        return policy
    raise ValueError("ComponentPolicyBuilder requires a single 'policy' component")
