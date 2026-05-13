# policy

**Parent:** [env_actor](../README.md)

Policy loading, registration, and protocol definition. Everything inference-time about a policy lives here.

## Table of contents

- [Subdirectories](#subdirectories)
- [The Policy protocol](#the-policy-protocol)
- [How build_policy works](#how-build_policy-works)
- [Adding a new policy](#adding-a-new-policy)
- [Related docs](#related-docs)

## Subdirectories

| Directory | Purpose |
|---|---|
| [`templates/`](templates/README.md) | The `Policy` protocol — the structural-subtyping interface every policy must satisfy. |
| [`registry/`](registry/README.md) | `POLICY_REGISTRY` — the string-keyed lookup used by `build_policy`. |
| [`utils/`](utils/README.md) | `build_policy` (YAML → policy instance), `load_state_dict_cpu_into_module`. |
| [`policies/`](policies/README.md) | Concrete implementations: `openpi_policy`, `dsrl_openpi_policy`. |

## The Policy protocol

Defined in [`templates/policy.py`](templates/policy.py) using Python's `Protocol` (structural subtyping — no inheritance required):

```python
@runtime_checkable
class Policy(Protocol):
    def __init__(self, components: dict[str, nn.Module], **kwargs) -> None: ...
    def predict(self, input_data: dict, data_normalization_interface) -> np.ndarray: ...
    def guided_inference(self, input_data: dict, data_normalization_interface,
                         min_num_actions_executed: int, action_chunk_size: int) -> np.ndarray: ...
    def warmup(self) -> None: ...
    def freeze_all_model_params(self) -> None: ...
```

| Method | Used by | Purpose |
|---|---|---|
| `predict()` | [Sequential](../auto/inference_algorithms/sequential/README.md) | Standard single-shot inference |
| `guided_inference()` | [RTC](../auto/inference_algorithms/rtc/README.md) | Inference with action inpainting |
| `warmup()` | Both | CUDA kernel warmup; triggers `torch.compile` |
| `freeze_all_model_params()` | Both (optional) | Disable `requires_grad` to skip gradient memory |

Full I/O shapes: [docs/api.md § Policy protocol](../../docs/api.md#policy-protocol).

## How build_policy works

[`utils/loader.py`](utils/loader.py) → `build_policy(policy_yaml_path, map_location="cpu")`:

1. **Load YAML** via `load_policy_config()` (delegating to the trainer's config loader, which handles `defaults` composition).
2. **Resolve component paths** — entries in `model.component_config_paths` are resolved against the policy YAML's directory.
3. **Build model components** — `PolicyConstructorModelFactory.build(resolved_paths)` returns a `dict[str, nn.Module]`.
4. **Load checkpoints** (optional) — if `config["checkpoint_path"]` is set, `torch.load(checkpoint_path/<name>.pt, map_location=...)` for each component.
5. **Resolve policy class** — looks up `policy.type` in `POLICY_REGISTRY`. If missing, auto-imports `env_actor.policy.policies.<type>.<type>` (which will register on import).
6. **Instantiate** — `policy_cls(components=components, **policy_params)`.

## Adding a new policy

1. Create `policies/your_policy/your_policy.py`.
2. Decorate the class:

   ```python
   from env_actor.policy.registry import POLICY_REGISTRY

   @POLICY_REGISTRY.register("your_policy")
   class YourPolicy:
       def __init__(self, components, **kwargs):
           self.model = next(iter(components.values()))
       def predict(self, input_data, data_normalization_interface): ...
       def guided_inference(self, input_data, data_normalization_interface,
                            min_num_actions_executed, action_chunk_size): ...
       def warmup(self): pass
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

4. Run: `python run_inference.py --robot igris_b -P env_actor/policy/policies/your_policy/your_policy.yaml`.

Full walkthrough: [docs/walkthroughs/03_add_a_new_policy.md](../../docs/walkthroughs/03_add_a_new_policy.md).

## Related docs

- [docs/api.md § Policy protocol](../../docs/api.md#policy-protocol) and [§ build_policy](../../docs/api.md#build_policy)
- [docs/walkthroughs/03_add_a_new_policy.md](../../docs/walkthroughs/03_add_a_new_policy.md)
- [docs/concepts.md § What is a VLA policy?](../../docs/concepts.md#what-is-a-vla-policy)
- [policy_constructor MENTAL_MODEL.md](https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/docs/MENTAL_MODEL.md) — how YAML becomes `nn.Module`
