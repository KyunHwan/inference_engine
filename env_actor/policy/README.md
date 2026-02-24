# policy

Policy loading, registration, and protocol definition. This package defines how inference policies are structured, discovered, and instantiated from YAML configuration.

## Key Files

| Path | Purpose |
|---|---|
| `templates/policy.py` | `Policy` protocol — the interface all policies must satisfy |
| `registry/core.py` | Generic `Registry` class with optional base-class enforcement |
| `registry/__init__.py` | `POLICY_REGISTRY` global instance |
| `utils/loader.py` | `build_policy()` — YAML config → instantiated policy |
| `utils/weight_transfer.py` | CPU → GPU state dict transfer utility |
| `policies/` | Concrete policy implementations |

## The Policy Protocol

Defined in `templates/policy.py` using Python's `Protocol` (structural subtyping — no inheritance required):

```python
@runtime_checkable
class Policy(Protocol):
    def __init__(self, components: dict[str, nn.Module], **kwargs) -> None: ...
    def predict(self, input_data: dict, data_normalization_interface) -> np.ndarray: ...
    def guided_inference(self, input_data: dict, data_normalization_interface,
                         min_num_actions_executed, action_chunk_size) -> np.ndarray: ...
    def warmup(self) -> None: ...
    def freeze_all_model_params(self) -> None: ...
```

| Method | Used By | Purpose |
|---|---|---|
| `predict()` | Sequential algorithm | Standard single-shot inference |
| `guided_inference()` | RTC algorithm | Inference with action inpainting (blends previous chunk with new prediction) |
| `warmup()` | Both | CUDA kernel warmup via `torch.compile` / `cudnn.benchmark` |
| `freeze_all_model_params()` | Both | Freeze parameters for inference-only mode |

### Input/Output Shapes

**`predict()` input (`input_data` dict):**
- `"proprio"`: `(proprio_history_size, state_dim)` float32
- `"head"`, `"left"`, `"right"`: `(num_img_obs, 3, H, W)` uint8
- `"prompt"` (optional): string

**`guided_inference()` additional inputs:**
- `"est_delay"`: int — estimated inference latency in control steps
- `"prev_action"`: `(action_chunk_size, action_dim)` float32 — previous un-executed actions

**Return:** `np.ndarray` of shape `(action_chunk_size, action_dim)` float32

## How `build_policy()` Works

`utils/loader.py` → `build_policy(policy_yaml_path, map_location="cpu")`:

1. **Load YAML** — calls `load_policy_config()` (delegating to the trainer's config loader)
2. **Resolve component paths** — relative paths are resolved against the YAML file's directory
3. **Build model components** — uses `PolicyConstructorModelFactory` from the trainer submodule to build `nn.Module` components from component YAML configs
4. **Load checkpoints** — if `checkpoint_path` is set, loads `{component_name}.pt` state dicts
5. **Instantiate policy** — looks up `policy.type` in `POLICY_REGISTRY`; if not registered, auto-imports `env_actor.policy.policies.{type}.{type}`

## Adding a New Policy

1. Create `policies/your_policy/your_policy.py`
2. Register with the decorator:

```python
from env_actor.policy.registry import POLICY_REGISTRY

@POLICY_REGISTRY.register("your_policy")
class YourPolicy:
    def __init__(self, components, **kwargs):
        self.model = next(iter(components.values()))

    def predict(self, input_data, data_normalization_interface):
        ...  # Return (action_chunk_size, action_dim) np.ndarray

    def guided_inference(self, input_data, data_normalization_interface,
                         min_num_actions_executed, action_chunk_size):
        ...  # Return (action_chunk_size, action_dim) np.ndarray

    def warmup(self):
        pass

    def freeze_all_model_params(self):
        for p in self.model.parameters():
            p.requires_grad = False
```

3. Create `policies/your_policy/your_policy.yaml`:

```yaml
model:
  component_config_paths:
    main: components/your_model.yaml

policy:
  type: your_policy
```

4. Run with: `python run_inference.py --robot igris_b -P env_actor/policy/policies/your_policy/your_policy.yaml`
