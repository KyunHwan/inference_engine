# Policies

Policy construction layer that turns built model components into an `InferencePolicy` used by inference pipelines.

Key modules
- `loader.py`: two-stage loader; builds model components then calls the registered policy builder.
- `interfaces.py`: Protocols for `InferencePolicy` and `PolicyBuilder`.
- `normalization.py`: loads dataset stats and returns normalization tensors.
- `registry.py`: `POLICY_REGISTRY` maps builder names to implementations.

Flow
1) `build_policy` selects a builder by `policy.type` from the inference YAML.
2) `build_models` loads components and checkpoints.
3) The builder wraps components into an `InferencePolicy` with metadata and normalization stats.
