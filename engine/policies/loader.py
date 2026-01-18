"""Two-stage policy loader (models -> policy)."""
from __future__ import annotations

import torch

from engine.config.inference_schemas import ModelConfig, PolicyConfig
from engine.modeling.build_models import ModelBuildRequest, build_models
from engine.policies.interfaces import InferencePolicy
from engine.policies.registry import POLICY_REGISTRY


def build_policy(
    model_cfg: ModelConfig,
    policy_cfg: PolicyConfig,
    checkpoint_path: str,
    device: torch.device | str | None = None,
) -> InferencePolicy:
    builder = POLICY_REGISTRY.get(policy_cfg.type)
    components = build_models(
        ModelBuildRequest(
            component_config_paths=model_cfg.component_config_paths,
            component_checkpoint_paths=model_cfg.component_checkpoint_paths,
            checkpoint_path=checkpoint_path,
            device=device,
        )
    )
    return builder.build_policy(components, policy_cfg.params, device=device)
