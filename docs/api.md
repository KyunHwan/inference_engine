# API reference

Public interfaces, expected inputs/outputs, and key data types. If you are looking for shapes first, jump straight to [Key data shapes](#key-data-shapes-igris_b-defaults).

## Table of contents

- [Key data shapes (IGRIS_B defaults)](#key-data-shapes-igris_b-defaults)
- [Policy protocol](#policy-protocol)
- [build_policy](#build_policy)
- [ControllerInterface](#controllerinterface)
- [DataNormalizationInterface](#datanormalizationinterface)
- [RuntimeParams](#runtimeparams)
- [Action inpainting](#action-inpainting)
- [SharedMemoryInterface (RTC only)](#sharedmemoryinterface-rtc-only)

## Key data shapes (IGRIS_B defaults)

| Data | Shape | Dtype | Notes |
|---|---|---|---|
| Proprioception (single frame) | `(24,)` | float32 | 6 left-arm + 6 right-arm joints + 6 left-hand + 6 right-hand fingers |
| Camera image (single frame) | `(3, 240, 320)` | uint8 | RGB, CHW, resized from 1600×1200 |
| Action vector | `(24,)` | float32 | Layout: `[L-arm, R-arm, L-fingers, R-fingers]` |
| Action chunk | `(50, 24)` | float32 | 50-step horizon × 24-DOF |
| Proprio history buffer | `(50, 24)` | float32 | Newest at index 0 |
| Image history buffer | `(1, 3, 240, 320)` | uint8 | `num_img_obs = 1` |

These are the values configured in [inference_runtime_params.json](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json); change them there to retune (see [configuration_cookbook.md](configuration_cookbook.md)).

## Policy protocol

**File**: [env_actor/policy/templates/policy.py](../env_actor/policy/templates/policy.py)

```python
@runtime_checkable
class Policy(Protocol):
    def __init__(self, components: dict[str, nn.Module], **kwargs: Any) -> None: ...
    def predict(self, input_data: dict, data_normalization_interface) -> np.ndarray: ...
    def guided_inference(self, input_data: dict, data_normalization_interface,
                         min_num_actions_executed: int,
                         action_chunk_size: int) -> np.ndarray: ...
    def warmup(self) -> None: ...
    def freeze_all_model_params(self) -> None: ...
```

The protocol is `@runtime_checkable`, so `isinstance(obj, Policy)` works as a duck-type check.

### `__init__(components, **kwargs)`

| Parameter | Type | Description |
|---|---|---|
| `components` | `dict[str, nn.Module]` | Named model components built by `PolicyConstructorModelFactory` |
| `**kwargs` | `Any` | Additional parameters from `policy.params` in the YAML config |

### `predict(input_data, data_normalization_interface)`

Used by the Sequential algorithm. Standard inference.

**`input_data` dict:**

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `"proprio"` | `(proprio_history_size, state_dim)` | float32 | Proprioceptive state history, newest at index 0 |
| `"head"` | `(num_img_obs, 3, H, W)` | uint8 | Head camera image history |
| `"left"` | `(num_img_obs, 3, H, W)` | uint8 | Left camera image history |
| `"right"` | `(num_img_obs, 3, H, W)` | uint8 | Right camera image history |
| `"prompt"` (optional) | `str` | — | Language instruction; falls back to the YAML's `default_prompt` |

**Returns:** `np.ndarray` of shape `(action_chunk_size, action_dim)`, dtype float32.

**Where this is used:**

- [sequential_actor.py:127](../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L127) — `self.policy.predict(obs, self.data_normalization_interface)`

### `guided_inference(input_data, data_normalization_interface, min_num_actions_executed, action_chunk_size)`

Used by the RTC algorithm. Inference followed by action inpainting (see [concepts.md § Action inpainting](concepts.md#action-inpainting)).

**Additional `input_data` keys** beyond `predict`:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `"est_delay"` | scalar `int` | — | Estimated inference latency in control steps (from `MaxDeque.max()`) |
| `"prev_action"` | `(action_chunk_size, action_dim)` | float32 | Tail of the previous action chunk (zero-padded at the end) |
| `"num_control_iters"` | scalar `int` | — | Control iterations since the last inference call (also returned by the SHM read) |

**Additional parameters:**

| Parameter | Type | Description |
|---|---|---|
| `min_num_actions_executed` | `int` | Hardcoded `35` in the inference loop — used as `executed` in the weight schedule |
| `action_chunk_size` | `int` | Total action chunk size; equals `runtime_params.action_chunk_size` |

**Returns:** `np.ndarray` of shape `(action_chunk_size, action_dim)`, dtype float32.

**Where this is used:**

- [inference_loop.py:121](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L121) — inside `torch.inference_mode + autocast(bfloat16)`

### `warmup()`

Optional one-shot dummy forward pass. Triggers `torch.compile` and lets `cudnn.benchmark` pick fast kernels. Both actors wrap it in `try/except` so a warmup failure doesn't crash startup.

**Where this is used:**

- [sequential_actor.py:91](../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L91)
- [inference_loop.py:77](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L77)

### `freeze_all_model_params()`

Sets `requires_grad = False` on every parameter so inference doesn't allocate gradient memory. Not always called by the actors — they rely on `policy.eval() + torch.inference_mode()` and `freeze_all_model_params` is up to the policy's `__init__` to invoke if needed.

---

## build_policy

**File**: [env_actor/policy/utils/loader.py](../env_actor/policy/utils/loader.py)

```python
def build_policy(
    policy_yaml_path: str,
    *,
    map_location: str | torch.device | None = "cpu",
) -> Policy
```

Loads and instantiates a policy from a YAML config file.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `policy_yaml_path` | `str` | — | Path to the policy YAML config |
| `map_location` | `str \| torch.device \| None` | `"cpu"` | Device for `torch.load` when reading checkpoints |

**Returns:** A `Policy` instance.

**Resolution steps:**

1. Load the YAML with defaults composition (trainer's `load_config`).
2. Resolve `model.component_config_paths` relative to the YAML's directory.
3. `PolicyConstructorModelFactory().build(resolved_paths)` → `dict[str, nn.Module]`.
4. If `config["checkpoint_path"]` is set, load `<checkpoint_path>/<component>.pt` per component.
5. Look up `config["policy"]["type"]` in `POLICY_REGISTRY`. If missing, auto-import `env_actor.policy.policies.<type>.<type>` (which will register on import).
6. Instantiate `policy_cls(components=components, **policy_params)`.

**YAML shape:**

```yaml
model:
  component_config_paths:
    component_name: path/to/component.yaml   # Relative to YAML dir or absolute

# Optional: load checkpoint weights
# checkpoint_path: /abs/path/to/checkpoints  # Reads {component_name}.pt files

policy:
  type: registered_policy_name
  params:                                    # Optional, passed as **kwargs to __init__
    key: value
```

**Where this is used:**

- [sequential_actor.py:67](../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L67) — `build_policy(policy_yaml_path=..., map_location="cpu").to(self.device)`
- [inference_loop.py:67](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L67)

---

## ControllerInterface

**File**: [env_actor/robot_io_interface/controller_interface.py](../env_actor/robot_io_interface/controller_interface.py)

```python
class ControllerInterface:
    def __init__(self, runtime_params, inference_runtime_topics_config, robot: str): ...
```

The robot-agnostic façade. All real work is delegated to the robot-specific `ControllerBridge`.

### Required bridge methods

#### `read_state() -> dict`

Returns the current robot state.

```python
{
    "proprio": np.ndarray,  # (state_dim,) float32
    "head":    np.ndarray,  # (3, H, W) uint8
    "left":    np.ndarray,  # (3, H, W) uint8
    "right":   np.ndarray,  # (3, H, W) uint8
}
```

#### `publish_action(action, prev_joint) -> tuple`

Publishes one action step with slew-rate limiting.

| Parameter | Type | Description |
|---|---|---|
| `action` | `np.ndarray` | `(action_dim,)` action vector. IGRIS_B layout: `[L-arm, R-arm, L-finger, R-finger]` |
| `prev_joint` | `np.ndarray` | Previous joint state for delta computation |

**Returns:** `(smoothed_joints, fingers)` — `np.ndarray` tuple. Use `smoothed_joints` as the next call's `prev_joint`.

#### `start_state_readers() -> None`

Launches camera capture threads and the `rclpy` executor in a background thread.

#### `init_robot_position() -> np.ndarray`

Moves the robot to its home pose. Returns the initial joint state.

#### `shutdown() -> None`

Tears down the `rclpy` node and the executor.

#### `recorder_rate_controller()`

Returns an rclpy rate object (`HZ` Hz). Not currently used by the actors — they use `time.perf_counter()` + `DT` directly.

### Properties

| Property | Type | Source |
|---|---|---|
| `DT` | `float` | `1.0 / runtime_params.HZ` |
| `policy_update_period` | `int` | `runtime_params.policy_update_period` |

**Where this is used:**

- [sequential_actor.py:69](../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L69)
- [control_loop.py:51](../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L51)

---

## DataNormalizationInterface

**File**: [env_actor/nom_stats_manager/data_normalization_interface.py](../env_actor/nom_stats_manager/data_normalization_interface.py)

```python
class DataNormalizationInterface:
    def __init__(self, robot: str, data_stats: dict): ...
    def normalize_state(self, state: dict) -> dict: ...
    def normalize_action(self, action: np.ndarray) -> np.ndarray: ...
    def denormalize_action(self, action: np.ndarray) -> np.ndarray: ...
```

### IGRIS_B bridge specifics

[data_normalization_manager.py](../env_actor/nom_stats_manager/robots/igris_b/data_normalization_manager.py):

- `normalize_state(state)` reads the **concatenation** of `norm_stats["observation.state"]["mean"]` and `norm_stats["observation.current"]["mean"]` (similarly for `std`), and uses the prefix of length `state["proprio"].shape[-1]` to z-score the proprio. Images divide by 255.
- `denormalize_action(action)` uses `norm_stats["action"]["mean"]` and `["std"]`.

So the IGRIS_B pickle must contain at least these keys:

```
observation.state.mean / std
observation.current.mean / std
action.mean / std
```

**Where this is used:**

- [sequential_actor.py:73](../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L73)
- [inference_loop.py:94](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L94)

---

## RuntimeParams

**File**: [env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.py](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.py)

| Property | Type | Source field | Notes |
|---|---|---|---|
| `HZ` | `int` | `HZ` | Control frequency |
| `policy_update_period` | `int` | `policy_update_period` | Sequential only |
| `max_delta` | `float` | `np.deg2rad(max_delta_deg)` | Computed |
| `proprio_state_dim` | `int` | `proprio_state_dim` | |
| `proprio_history_size` | `int` | `proprio_history_size` | |
| `camera_names` | `list[str]` | `camera_names` | |
| `num_img_obs` | `int` | `num_img_obs` | |
| `img_obs_every` | `int` | `img_obs_every` | |
| `action_dim` | `int` | `action_dim` | |
| `action_chunk_size` | `int` | `action_chunk_size` | |
| `mono_img_resize_width` | `int` | `mono_image_resize.width` | |
| `mono_img_resize_height` | `int` | `mono_image_resize.height` | |

**Method**: `read_stats_file() -> dict` — `pickle.load` of `norm_stats_file_path`; returns `None` and prints a warning if the file is missing.

---

## Action inpainting

**File**: [env_actor/inference_engine_utils/action_inpainting.py](../env_actor/inference_engine_utils/action_inpainting.py)

### `compute_guided_prefix_weights(delay_steps, executed, total, schedule="exp")`

```python
def compute_guided_prefix_weights(
    delay_steps: int,
    executed: int,
    total: int,
    *,
    schedule: str = "exp",
) -> np.ndarray  # shape (total,), values in [0, 1]
```

| Parameter | Description |
|---|---|
| `delay_steps` | `est_delay` from the SHM manager; how many steps we expect to lose to inference latency |
| `executed` | `min_num_actions_executed`; the minimum executed-since-last-inference threshold |
| `total` | `action_chunk_size` |
| `schedule` | `"exp"` (default), `"ones"`, `"zeros"` |

**Weight layout (with `schedule="exp"`):**

```
indices:    0      ...    start      ...    total-span     ...    total-1
weight:    1.0    1.0    decay      decay       0.0        0.0     0.0
            ╰── keep ──╯ ╰── blend ──╯           ╰── use new ──╯
                          (exp decay)
```

Where `start = max(min(delay_steps, total), 0)` and `span = min(max(executed, 1), max(total - start, 1))`.

**Caller usage** (in both policies):

```python
weights = compute_guided_prefix_weights(est_delay, min_executed, chunk_size).reshape(-1, 1)
blended = prev_action * weights + new_prediction * (1.0 - weights)
```

### `guided_action_chunk_inference(...)`

PyTorch-based version that performs the inpainting *inside* the flow-matching ODE simulation (rather than blending post-hoc). Not currently called by the shipped policies; it's an alternative used when the action decoder is a transformer-with-cross-attention and you want to bend the trajectory toward `prev_action_chunk` during denoising.

```python
def guided_action_chunk_inference(
    action_decoder: torch.nn.Module,
    cond_memory: torch.Tensor,
    discrete_semantic_input: torch.Tensor | None,
    prev_action_chunk: torch.Tensor,
    delay: int,
    executed_steps: int,
    num_ode_sim_steps: int,
    num_queries: int,
    action_dim: int,
    max_guidance_weight: float = 5.0,
    input_noise: torch.Tensor | None = None,
) -> torch.Tensor  # (batch, num_queries, action_dim)
```

---

## SharedMemoryInterface (RTC only)

**File**: [env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py](../env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py)

Robot-agnostic factory; delegates everything to the robot-specific `SharedMemoryManager` (e.g. [`igris_b/shm_manager_bridge.py`](../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py)).

Selected methods (full surface in the bridge):

| Method | Caller | Purpose |
|---|---|---|
| `attach_from_specs(...)` | Both children at startup | Re-attach to existing SHM blocks created in the parent |
| `wait_for_inference_ready()` | Control loop | Block until inference signals it's done initializing |
| `wait_for_min_actions(min_actions)` | Inference loop | Block until N steps have executed since the last call |
| `atomic_read_for_inference()` | Inference loop | Snapshot `proprio`, `head`, `left`, `right`, build `prev_action`, return `dict` |
| `atomic_write_obs_and_increment_get_action(obs, action_chunk_size)` | Control loop | Write the latest observation, increment the counter, return this step's action |
| `write_action_chunk_n_update_iter_val(chunk, executed)` | Inference loop | Write a new chunk, decrement the counter by `executed` |
| `init_action_chunk_obs_history(obs_history)` | Control loop | Bootstrap proprio history + initial action chunk for a new episode |
| `signal_episode_complete()` / `clear_episode_complete()` | Control loop | Episode-transition handshake |
| `signal_stop()` / `stop_event_is_set()` | Either | Shutdown |
| `cleanup()` | Both | Close SHM views (only the creator unlinks) |

Synchronization primitives (created in `RTCActor.start`, shared with both children):

- `RLock` — guards atomic reads/writes against torn updates.
- `control_iter_cond` (`Condition`) — inference waits on this for `num_control_iters` to cross the threshold.
- `inference_ready_cond` (`Condition`) — control waits on this for `inference_ready_flag`.
- `stop_event` (`Event`) — set by anyone to request shutdown.
- `episode_complete_event` (`Event`) — set by control to end an episode.
- `num_control_iters` (`Value[int]`) — counter of control steps since last inference.
- `inference_ready_flag` (`Value[bool]`) — true once inference has warmed up.

Full sequence diagrams in [rtc_shared_memory.md](rtc_shared_memory.md).
