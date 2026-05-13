# policies

**Parent:** [policy](../README.md)

Concrete policy implementations. Each lives in its own subdirectory.

## Currently shipped

| Directory | Class | Registry key | Notes |
|---|---|---|---|
| [`openpi_policy/`](openpi_policy/README.md) | `OpenPiPolicy` | `openpi_policy` | Single-component wrapper for OpenPI/Pi0.5. |
| [`dsrl_openpi_policy/`](dsrl_openpi_policy/README.md) | `DsrlOpenpiPolicy` | `dsrl_openpi_policy` | Four-component pipeline: backbone → noise processor → noise actor → OpenPI. The default policy for [run_inference.py](../../../run_inference.py). |

## Directory convention

Adding a new policy:

```
policies/
└── your_policy/
    ├── __init__.py           # Can be empty
    ├── your_policy.py        # @POLICY_REGISTRY.register("your_policy")
    ├── your_policy.yaml      # policy.type: your_policy
    └── components/
        └── your_model.yaml   # Architecture config consumed by PolicyConstructorModelFactory
```

The string in `@POLICY_REGISTRY.register("...")` must match `policy.type` in the YAML.

## How `build_policy` finds your code

If `your_policy` isn't yet in `POLICY_REGISTRY`, [`build_policy`](../utils/loader.py) auto-imports `env_actor.policy.policies.your_policy.your_policy`. So the file naming must follow the convention exactly.

## Related docs

- [docs/walkthroughs/03_add_a_new_policy.md](../../../docs/walkthroughs/03_add_a_new_policy.md) — the worked copy-and-modify walkthrough.
- [docs/api.md § build_policy](../../../docs/api.md#build_policy)
- [policy_constructor MENTAL_MODEL.md](https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/docs/MENTAL_MODEL.md) — how each component YAML becomes an `nn.Module`.
