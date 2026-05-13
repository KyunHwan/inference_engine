# policy/registry

**Parent:** [policy](../README.md)

The string → class lookup that lets `build_policy` instantiate a policy named in YAML.

## Table of contents

- [Files](#files)
- [POLICY_REGISTRY](#policy_registry)
- [Registry semantics](#registry-semantics)
- [Plugins](#plugins)
- [Related docs](#related-docs)

## Files

| File | What it provides |
|---|---|
| [`__init__.py`](__init__.py) | `POLICY_REGISTRY` — the global `Registry[type[Policy]]` instance used by [`build_policy`](../utils/loader.py). |
| [`core.py`](core.py) | Local `Registry` class with optional base-class enforcement. **Note:** `POLICY_REGISTRY` in `__init__.py` actually imports `Registry` from the trainer submodule (`from trainer.trainer.registry.core import Registry`), not from this local file. The local file is kept as a structurally-equivalent reference but is not currently used by `POLICY_REGISTRY`. If you import `Registry` here, you'll be using the local class — the trainer's version is otherwise identical in API. |
| [`plugins.py`](plugins.py) | `load_plugins(modules: Iterable[str])` — `importlib.import_module` each path; dedups via `_LOADED_MODULES`. Not currently wired into any startup path in this repo. |

## POLICY_REGISTRY

The single instance you interact with from policy code:

```python
from env_actor.policy.registry import POLICY_REGISTRY

@POLICY_REGISTRY.register("openpi_policy")
class OpenPiPolicy:
    ...
```

`POLICY_REGISTRY` is declared with `expected_base=Policy`, so `Registry.add` will reject a class that isn't structurally a `Policy`. With Protocol-based base detection this means "has the right methods at registration time."

## Registry semantics

The `Registry` API (from either `core.py` or the trainer's equivalent):

| Method | Purpose |
|---|---|
| `register(key=None)` | Decorator factory. Without a key, uses `cls.__name__`. |
| `add(key, obj)` | Add explicitly. Raises `KeyError` on duplicate. |
| `get(key)` | Look up. Raises `KeyError` with available-key suggestions. |
| `has(key)` | Boolean check. |
| `keys()` | Sorted list. |

Registrations happen at **module import time** via the decorator. [`build_policy`](../utils/loader.py) handles the case where a policy hasn't been imported yet by doing `importlib.import_module(f"env_actor.policy.policies.{type}.{type}")` on a registry miss.

## Plugins

The trainer-side codebase uses `plugins.py` to import extension modules listed in a YAML config so their `@register` decorators fire. The inference engine doesn't currently use it — `build_policy` does its own targeted auto-import on miss. If you ever need to load a third-party policy that doesn't follow the `env_actor.policy.policies.X.X` naming convention, you could call `load_plugins([...])` from your launch script before invoking `build_policy`.

> **What this README replaces.** An earlier version of this file enumerated trainer registries (`TRAINER_REGISTRY`, `DATAMODULE_REGISTRY`, `OPTIMIZER_REGISTRY`, `SCHEDULER_REGISTRY`, etc.) and a `builtins.py` file. None of those exist in this directory — they live in [trainer/trainer/registry/](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/trainer/registry/) and are not used by the inference engine.

## Related docs

- [docs/api.md § build_policy](../../../docs/api.md#build_policy) — how the registry is consulted at load time.
- [docs/walkthroughs/03_add_a_new_policy.md § Where build_policy finds your class](../../../docs/walkthroughs/03_add_a_new_policy.md#where-build_policy-finds-your-class)
- [docs/glossary.md § Registry](../../../docs/glossary.md#registry)
