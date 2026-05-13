# Walkthrough 03 — Add a new policy

Copy `openpi_policy`, rename to `my_policy`, change one method, register it, and run. ~90 minutes.

## Table of contents

- [Goal](#goal)
- [Step 1: Make a copy](#step-1-make-a-copy)
- [Step 2: Implement the Policy protocol](#step-2-implement-the-policy-protocol)
- [Step 3: Register the class](#step-3-register-the-class)
- [Step 4: Write the two YAML files](#step-4-write-the-two-yaml-files)
- [Step 5: Run it](#step-5-run-it)
- [Where build_policy finds your class](#where-build_policy-finds-your-class)
- [Common pitfalls](#common-pitfalls)

## Goal

By the end of this walkthrough you will have a new policy class registered as `"my_policy"`, runnable with:

```bash
python run_inference.py --robot igris_b \
  -P env_actor/policy/policies/my_policy/my_policy.yaml
```

…and `build_policy` will auto-import it without you editing any factory or registry file.

## Step 1: Make a copy

Pick the simpler `openpi_policy` as the template (`dsrl_openpi_policy` has four components and more moving parts).

```bash
cd env_actor/policy/policies
cp -r openpi_policy my_policy
cd my_policy
mv openpi_policy.py my_policy.py
mv openpi_policy.yaml my_policy.yaml
# `components/openpi_batched.yaml` stays as-is for now.
```

You should have:

```
env_actor/policy/policies/my_policy/
├── __init__.py
├── my_policy.py
├── my_policy.yaml
└── components/
    └── openpi_batched.yaml
```

## Step 2: Implement the Policy protocol

Edit [my_policy.py](../../env_actor/policy/policies/my_policy/my_policy.py). The protocol you must satisfy is in [templates/policy.py](../../env_actor/policy/templates/policy.py):

```python
class Policy(Protocol):
    def __init__(self, components: dict[str, nn.Module], **kwargs: Any) -> None: ...
    def predict(self, input_data: dict, data_normalization_interface) -> np.ndarray: ...
    def guided_inference(self, input_data: dict, data_normalization_interface,
                         min_num_actions_executed: int,
                         action_chunk_size: int) -> np.ndarray: ...
    def warmup(self) -> None: ...
    def freeze_all_model_params(self) -> None: ...
```

Shapes are in [api.md § Key data shapes](../api.md#key-data-shapes-igris_b-defaults).

### Step 2a: Rename the class

Replace `OpenPiPolicy` with `MyPolicy` throughout. The class name doesn't have to match the registry key, but matching makes it easy to find.

### Step 2b: Update the registry decorator

```python
@POLICY_REGISTRY.register("my_policy")
class MyPolicy:
    ...
```

The string must match `policy.type` in your YAML (Step 4).

### Step 2c: Change one method

For your first version, let's keep most of the logic the same but tweak `predict` so we can see we're running our copy. Add a deterministic offset to the predicted actions:

```python
def predict(self, obs, data_normalization_interface):
    # ... existing logic that produces `batched_actions` ...
    actions = batched_actions[0]
    actions[:, :6] += 0.001   # Tiny offset on left-arm joints; obvious in logs.
    return actions
```

Production policies would do something less silly. The point here is to prove that *your* code runs.

## Step 3: Register the class

Already done in Step 2b. The `@POLICY_REGISTRY.register("my_policy")` decorator fires the moment the module is imported. `build_policy`'s auto-import logic ([loader.py:71-72](../../env_actor/policy/utils/loader.py#L71-L72)) will trigger that import for you.

If you want to be defensive, also import your module from `policies/__init__.py` so it registers regardless of how it's discovered.

## Step 4: Write the two YAML files

### `my_policy.yaml` (top-level)

```yaml
model:
  component_config_paths:
    openpi_model: components/openpi_batched.yaml

policy:
  type: my_policy
```

Two things must match:

- `model.component_config_paths.<key>` matches what your `MyPolicy.__init__` (inherited from `OpenPiPolicy._resolve_wrapper`) looks for. The original code accepts either a single component or one named `openpi_model`; the safest choice is to keep `openpi_model`.
- `policy.type: my_policy` matches the registry key.

### `components/openpi_batched.yaml`

Keep the file from the copy. If you want a different checkpoint or prompt, edit the `params` block — see [configuration_cookbook.md § Change the OpenPI checkpoint path](../configuration_cookbook.md#change-the-openpi-checkpoint-path) and [§ Change the prompt](../configuration_cookbook.md#change-the-prompt).

## Step 5: Run it

```bash
python run_inference.py --robot igris_b --inference_algorithm sequential \
  -P env_actor/policy/policies/my_policy/my_policy.yaml
```

What to look for in the logs:

- `Warming up CUDA kernels...` — your `MyPolicy.warmup()` (which still calls the OpenPI wrapper's warmup) ran.
- A successful first `predict` call (no exception).
- On the robot or in `ros2 topic echo /igris_b/<robot_id>/target_joints`: the left-arm joint targets should be ~0.001 rad higher than what the un-modified `openpi_policy` would have published. That's your offset, end-to-end.

## Where build_policy finds your class

The relevant code is [loader.py:65-76](../../env_actor/policy/utils/loader.py#L65-L76):

```python
policy_type = config["policy"]["type"]        # "my_policy"

if not POLICY_REGISTRY.has(policy_type):
    module_path = f"env_actor.policy.policies.{policy_type}.{policy_type}"
    importlib.import_module(module_path)       # → env_actor.policy.policies.my_policy.my_policy

policy_cls = POLICY_REGISTRY.get(policy_type)
return policy_cls(components=components, **policy_params)
```

So the resolution is:

1. `policy.type` → `"my_policy"`.
2. Auto-import `env_actor.policy.policies.my_policy.my_policy` (which runs the `@POLICY_REGISTRY.register("my_policy")` decorator).
3. Look up `"my_policy"` in the registry → get your class.
4. Call `MyPolicy(components=..., **policy.params)`.

If `policy.type` doesn't exactly match the file name or the registered key, you'll see `KeyError: policy registry has no key 'my_policy'`.

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'env_actor.policy.policies.my_policy.my_policy'` | File misnamed, or you forgot `__init__.py` in the new directory | Make sure both `policies/my_policy/__init__.py` (can be empty) and `my_policy.py` exist |
| `KeyError: policy registry has no key 'my_policy'` | The decorator string doesn't match `policy.type` | Make them identical |
| `KeyError: policy registry already has key 'my_policy'` | You imported the module twice with different definitions (e.g., from `__init__.py` and via auto-import) | Decide once: either auto-import only, or pre-import only |
| `TypeError: ... unexpected keyword argument` | `MyPolicy.__init__` doesn't accept the keys under `policy.params` in YAML | Either widen with `**kwargs: Any` or remove the unwanted YAML param |
| The model loads but actions are all zero | Checkpoint not loaded (path wrong, or `MyPolicy.__init__` doesn't call `load_state_dict`) | Compare with `DsrlOpenpiPolicy.__init__` — does *your* class need to load weights from `checkpoint_path`? |

## What to try next

- Look at [`DsrlOpenpiPolicy`](../../env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.py) for a multi-component example.
- Read [`build_policy`](../../env_actor/policy/utils/loader.py) end to end so the resolution algorithm is one you can describe.
- If you want your policy to load its own checkpoint differently, read how DSRL's `params.checkpoint_path` is handled — it's a `params:` key on the YAML, not the loader-level `checkpoint_path`, and the class handles the `torch.load` itself.
