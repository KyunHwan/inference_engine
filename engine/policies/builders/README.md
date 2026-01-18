# Policy Builders

Implementations of `PolicyBuilder` that assemble `InferencePolicy` objects from model components.

Included builder
- `component_policy.py`: `ComponentPolicyBuilder` wraps a single policy component.
  - Loads normalization stats from `stats_path`.
  - Exposes metadata (dims, camera names, normalization tensors).
  - Delegates `encode_vision`, `body`, and forward pass to the underlying module.

Extending
- Implement a new `PolicyBuilder`.
- Register it in a plugin module and add that plugin to the inference YAML.
