"""Build model components from policy_constructor configs and load checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from engine.modeling.factories import PolicyConstructorModelFactory


@dataclass(frozen=True)
class ModelBuildRequest:
    component_config_paths: Mapping[str, str]
    checkpoint_path: str
    component_checkpoint_paths: Mapping[str, str] | None = None
    device: torch.device | str | None = None
    strict: bool = True
    eval_mode: bool = True


def build_models(req: ModelBuildRequest) -> dict[str, torch.nn.Module]:
    component_paths = _resolve_component_paths(req.component_config_paths)
    factory = PolicyConstructorModelFactory()
    models = factory.build(component_paths)
    if not isinstance(models, dict):
        models = {"policy": models}

    _load_checkpoints(
        models,
        req.checkpoint_path,
        req.component_checkpoint_paths,
        strict=req.strict,
    )

    if req.device is not None:
        device = torch.device(req.device)
        for key, model in models.items():
            models[key] = model.to(device)

    if req.eval_mode:
        for model in models.values():
            model.eval()

    return models


def _resolve_component_paths(paths: Mapping[str, str]) -> dict[str, str]:
    resolved = {}
    for name, path in paths.items():
        path_str = str(path)
        resolved[name] = str(Path(path_str).expanduser().resolve())
    return resolved


def _load_checkpoints(
    models: dict[str, torch.nn.Module],
    checkpoint_path: str,
    component_checkpoint_paths: Mapping[str, str] | None,
    strict: bool,
) -> None:
    ckpt_root = Path(checkpoint_path).expanduser()
    if component_checkpoint_paths:
        for name, rel_path in component_checkpoint_paths.items():
            if name not in models:
                raise KeyError(f"Checkpoint mapping provided for unknown component: {name}")
            resolved = _resolve_checkpoint_path(ckpt_root, rel_path)
            _load_state(models[name], resolved, strict)
        return

    if len(models) == 1 and ckpt_root.is_file():
        only_model = next(iter(models.values()))
        _load_state(only_model, ckpt_root, strict)
        return

    if ckpt_root.is_dir():
        for name, model in models.items():
            candidate = ckpt_root / f"{name}.pt"
            _load_state(model, candidate, strict)
        return

    raise ValueError("checkpoint_path must be a directory or a file for single-component policies")


def _resolve_checkpoint_path(root: Path, path: str) -> Path:
    cand = Path(path).expanduser()
    if cand.is_absolute():
        return cand
    if root.is_dir():
        return (root / cand).resolve()
    return (root.parent / cand).resolve()


def _load_state(model: torch.nn.Module, path: Path, strict: bool) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=strict)
