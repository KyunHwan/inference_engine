# policy/utils

**Parent:** [policy](../README.md)

Utilities for policy loading and weight management.

## Files

| File | What it provides |
|---|---|
| [`loader.py`](loader.py) | `build_policy(yaml_path, map_location="cpu") -> Policy` — the single API the actors call to instantiate a policy. |
| [`weight_transfer.py`](weight_transfer.py) | `load_state_dict_cpu_into_module(module, sd_cpu, strict=True)` — moves a CPU-loaded state dict into a module that may already be on GPU. Unwraps DDP-wrapped modules; matches each tensor to the target's device + dtype. |

## `build_policy` resolution

```
YAML file
  │
  ▼
load_policy_config()          ← delegate to trainer's load_config (defaults composition)
  │
  ▼
resolve component_config_paths  (relative-to-YAML or absolute)
  │
  ▼
PolicyConstructorModelFactory().build(paths) → dict[str, nn.Module]
  │
  ▼
(optional) torch.load(checkpoint_path/<name>.pt) per component
  │
  ▼
POLICY_REGISTRY.get(policy.type)  ← auto-import on miss
  │
  ▼
policy_cls(components=components, **policy.params)
```

Notes:

- `map_location` is `"cpu"` in both actors at call time; they then move the policy to `cuda` via `.to(self.device)`. Loading on CPU first avoids transient GPU memory pressure during construction.
- `model.component_config_paths` must be a non-empty mapping; the loader raises `ValueError` otherwise.
- `policy.type` is required; missing it raises `ValueError`.

## `load_state_dict_cpu_into_module`

Used when a policy class explicitly manages its own checkpoint loading. The function:

1. Unwraps DDP: `target = module.module if hasattr(module, "module") else module`.
2. Reads the target's current `state_dict()` to learn each tensor's device + dtype.
3. For each input tensor, casts to that device + dtype (using `non_blocking=True`).
4. Calls `target.load_state_dict(...)`.

This is not called by `build_policy` itself — `build_policy` uses plain `module.load_state_dict(torch.load(...))`. It exists for policies that load weights post-construction (e.g., shared weights pulled from Ray's object store).

## Related docs

- [docs/api.md § build_policy](../../../docs/api.md#build_policy) — public reference.
- [docs/walkthroughs/03_add_a_new_policy.md](../../../docs/walkthroughs/03_add_a_new_policy.md) — the worked example.
