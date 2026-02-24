# policies

Concrete policy implementations. Each policy lives in its own subdirectory following the convention `{name}/{name}.py` + `{name}.yaml`.

## Current Policies

### `openpi_policy`

Wraps the OpenPI (Pi0.5) vision-language-action model for inference on IGRIS robots.

**Files:**
- `openpi_policy/openpi_policy.py` — Policy class registered as `"openpi_policy"` in `POLICY_REGISTRY`
- `openpi_policy/openpi_policy.yaml` — Top-level policy config pointing to the component YAML
- `openpi_policy/components/openpi_batched.yaml` — Model architecture config (checkpoint path, action dimensions, inference steps)

**How it works:**
- Receives an `OpenPiBatchedWrapper` built by the model factory as a component
- Converts single-sample observations to batched format (adds batch dimension of 1)
- Delegates inference to the wrapper's `predict()` method
- For `guided_inference()`, computes action inpainting weights via `compute_guided_prefix_weights()` and blends the previous action chunk with the new prediction

**Key parameters** (from `openpi_batched.yaml`):
- `action_dim: 24` — 24-DOF action space (12 arm joints + 12 hand joints)
- `action_horizon: 50` — 50-step action chunk
- `num_inference_steps: 10` — flow-matching denoising steps

## Directory Convention

To add a new policy, create:

```
policies/
└── your_policy/
    ├── your_policy.py           # @POLICY_REGISTRY.register("your_policy")
    ├── your_policy.yaml         # policy.type: your_policy
    └── components/
        └── your_model.yaml      # Model architecture config
```

The `policy.type` field in the YAML must match the registry key used in the `@POLICY_REGISTRY.register()` decorator.
