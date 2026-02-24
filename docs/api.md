# API Reference

Public interfaces, expected inputs/outputs, and key data types.

## Policy Protocol

**File:** `env_actor/policy/templates/policy.py`

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

### `__init__(components, **kwargs)`

| Parameter | Type | Description |
|---|---|---|
| `components` | `dict[str, nn.Module]` | Named model components built by `PolicyConstructorModelFactory` |
| `**kwargs` | `Any` | Additional parameters from `policy.params` in the YAML config |

### `predict(input_data, data_normalization_interface)`

Standard inference. Called by the Sequential algorithm.

**`input_data` dict:**

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `"proprio"` | `(proprio_history_size, state_dim)` | float32 | Proprioceptive state history |
| `"head"` | `(num_img_obs, 3, H, W)` | uint8 | Head camera image history |
| `"left"` | `(num_img_obs, 3, H, W)` | uint8 | Left camera image history |
| `"right"` | `(num_img_obs, 3, H, W)` | uint8 | Right camera image history |
| `"prompt"` | `str` (optional) | — | Language instruction |

**Returns:** `np.ndarray` of shape `(action_chunk_size, action_dim)`, dtype float32

### `guided_inference(input_data, data_normalization_interface, min_num_actions_executed, action_chunk_size)`

Inference with action inpainting. Called by the RTC algorithm.

**Additional `input_data` keys** (beyond those in `predict`):

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `"est_delay"` | scalar `int` | — | Estimated inference latency in control steps |
| `"prev_action"` | `(action_chunk_size, action_dim)` | float32 | Previous un-executed action chunk |

**Additional parameters:**

| Parameter | Type | Description |
|---|---|---|
| `min_num_actions_executed` | `int` | Minimum actions executed before re-planning |
| `action_chunk_size` | `int` | Total action chunk size |

**Returns:** `np.ndarray` of shape `(action_chunk_size, action_dim)`, dtype float32

### `warmup()`

Triggers CUDA kernel warmup. Used with `torch.backends.cudnn.benchmark = True` to auto-select the fastest kernel for the input shapes. Typically runs a dummy forward pass.

### `freeze_all_model_params()`

Sets `requires_grad = False` on all model parameters to disable gradient computation during inference.

---

## build_policy

**File:** `env_actor/policy/utils/loader.py`

```python
def build_policy(
    policy_yaml_path: str,
    *,
    map_location: str | torch.device | None = "cpu"
) -> Policy
```

Loads and instantiates a policy from a YAML config file.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `policy_yaml_path` | `str` | — | Path to the policy YAML config |
| `map_location` | `str \| torch.device \| None` | `"cpu"` | Device for checkpoint loading |

**Returns:** An instance satisfying the `Policy` protocol.

**YAML format:**
```yaml
model:
  component_config_paths:
    component_name: path/to/component.yaml  # Relative to YAML dir or absolute

  # Optional: load checkpoint weights
  # checkpoint_path: /path/to/checkpoints/  # Contains {component_name}.pt files

policy:
  type: registered_policy_name
  params:  # Optional, passed as **kwargs to policy __init__
    key: value
```

---

## ControllerInterface

**File:** `env_actor/robot_io_interface/controller_interface.py`

```python
class ControllerInterface:
    def __init__(self, runtime_params, inference_runtime_topics_config, robot: str): ...
```

All methods delegate to the robot-specific `ControllerBridge`.

### Required Bridge Methods

#### `read_state() -> dict`

Returns the current robot state.

```python
{
    "proprio": np.ndarray,  # (state_dim,) float32
    "head": np.ndarray,     # (3, H, W) uint8
    "left": np.ndarray,     # (3, H, W) uint8
    "right": np.ndarray,    # (3, H, W) uint8
}
```

#### `publish_action(action, prev_joint) -> tuple`

Publishes a single action step to the robot with slew-rate limiting.

| Parameter | Type | Description |
|---|---|---|
| `action` | `np.ndarray` | Full action vector `(action_dim,)` |
| `prev_joint` | `np.ndarray` | Previous joint state for delta computation |

**Returns:** `(current_joints, current_fingers)` tuple of `np.ndarray`

#### `start_state_readers() -> None`

Launches background threads for camera capture and ROS2 topic subscriptions.

#### `init_robot_position() -> np.ndarray`

Moves the robot to its initial/home position. Returns the initial joint state as `np.ndarray`.

#### `shutdown() -> None`

Cleans up ROS2 nodes, stops camera threads, releases resources.

#### Properties

| Property | Type | Description |
|---|---|---|
| `DT` | `float` | Control period in seconds (`1.0 / HZ`) |
| `policy_update_period` | `int` | Steps between policy inference calls |

---

## DataNormalizationInterface

**File:** `env_actor/nom_stats_manager/data_normalization_interface.py`

### Required Bridge Methods

#### `normalize_state(state: dict) -> dict`

Normalizes a raw robot state dict for policy input.

- Proprioception: z-score normalization `(x - mean) / (std + eps)`
- Images: scale to [0, 1] by dividing by 255.0

#### `denormalize_action(action: np.ndarray) -> np.ndarray`

Converts normalized policy output back to robot action space: `action * std + mean`

---

## RuntimeParams

**File:** `env_actor/runtime_settings_configs/robots/{robot}/inference_runtime_params.py`

| Property | Type | Description |
|---|---|---|
| `HZ` | `int` | Control frequency in Hz |
| `policy_update_period` | `int` | Steps between policy calls |
| `max_delta` | `float` | Maximum angular change per step (radians) |
| `proprio_state_dim` | `int` | Proprioceptive state dimensionality |
| `proprio_history_size` | `int` | Number of past proprio frames |
| `camera_names` | `list[str]` | Camera names (e.g., `["head", "left", "right"]`) |
| `num_img_obs` | `int` | Number of past image frames |
| `img_obs_every` | `int` | Image subsampling rate |
| `action_dim` | `int` | Action space dimensionality |
| `action_chunk_size` | `int` | Actions per policy call |
| `mono_img_resize_width` | `int` | Target image width |
| `mono_img_resize_height` | `int` | Target image height |

**Method:** `read_stats_file() -> dict` — loads normalization statistics from the pickle file at `norm_stats_file_path`.

---

## Action Inpainting

**File:** `env_actor/inference_engine_utils/action_inpainting.py`

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

### `guided_action_chunk_inference(...)`

```python
def guided_action_chunk_inference(
    action_decoder: torch.nn.Module,
    cond_memory: torch.Tensor,
    discrete_semantic_input: Optional[torch.Tensor],
    prev_action_chunk: torch.Tensor,
    delay: int,
    executed_steps: int,
    num_ode_sim_steps: int,
    num_queries: int,
    action_dim: int,
    max_guidance_weight: float = 5.0,
    input_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor  # shape (batch, num_queries, action_dim)
```

---

## Key Data Shapes (IGRIS_B Defaults)

| Data | Shape | Dtype | Notes |
|---|---|---|---|
| Proprioception | `(24,)` | float32 | 6 arm joints + 6 hand joints per arm |
| Camera image | `(3, 240, 320)` | uint8 | RGB, CHW format |
| Action chunk | `(50, 24)` | float32 | 50-step horizon, 24-DOF |
| Observation history (proprio) | `(1, 24)` | float32 | Single frame default |
| Observation history (image) | `(1, 3, 240, 320)` | uint8 | Single frame default |
