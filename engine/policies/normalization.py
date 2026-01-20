"""Normalization loading utilities for inference policies."""
from __future__ import annotations

import os
import pickle
from typing import Any

import torch


def load_stats(
    stats_path: str,
    state_dim: int,
    action_dim: int,
    device: torch.device | str | None,
    stats_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if not stats_path:
        raise ValueError("stats_path must be provided for normalization")

    resolved = os.path.abspath(stats_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Normalization stats not found: {resolved}")

    with open(resolved, "rb") as f:
        stats = pickle.load(f)
    if not isinstance(stats, dict):
        raise ValueError("Normalization stats file must contain a dict")
    try:
        sm = _as_tensor(stats, "state_mean", state_dim, device)
        ss = _as_tensor(stats, "state_std", state_dim, device)
        am = _as_tensor(stats, "action_mean", action_dim, device)
        asd = _as_tensor(stats, "action_std", action_dim, device)
    except:
        sm = torch.cat([_as_tensor(stats['observation.state'], "mean", 24, device), _as_tensor(stats['observation.current'], "mean", 24, device)], dim=0)
        ss = torch.cat([_as_tensor(stats['observation.state'], "std", 24, device), _as_tensor(stats['observation.current'], "std", 24, device)], dim=0)
        am = _as_tensor(stats['action'], "mean", 24, device)
        asd = _as_tensor(stats['action'], "std", 24, device)
        print(sm.shape)

    return sm, ss, am, asd, float(stats_eps)


def _as_tensor(
    stats: dict[str, Any],
    key: str,
    expected_dim: int,
    device: torch.device | str | None,
) -> torch.Tensor:
    if key not in stats:
        raise KeyError(f"Normalization stats missing key: {key}")
    tensor = torch.as_tensor(stats[key], dtype=torch.float32, device=device).view(-1)
    if tensor.numel() != expected_dim:
        raise ValueError(f"{key} length {tensor.numel()} does not match expected {expected_dim}")
    return tensor
