# runtime_settings_configs

Per-robot runtime configuration. Each robot has its own subdirectory under `robots/` containing timing parameters, dimensions, ROS2 topic mappings, and initial joint positions.

## Structure

```
runtime_settings_configs/
└── robots/
    ├── igris_b/
    │   ├── inference_runtime_params.json    # Timing, dimensions, cameras
    │   ├── inference_runtime_topics.json    # ROS2 topic → observation key mapping
    │   ├── inference_runtime_params.py      # RuntimeParams class
    │   └── init_params.py                   # Initial joint positions, state keys
    └── igris_c/
        └── init_params.py                   # Placeholder (empty lists)
```

## Configuration Files

### `inference_runtime_params.json`

Controls inference timing, observation/action dimensions, and camera setup:

```json
{
  "HZ": 20,
  "max_delta_deg": 5,
  "policy_update_period": 50,
  "mono_image_resize": {"width": 320, "height": 240},
  "camera_names": ["head", "left", "right"],
  "proprio_state_dim": 24,
  "action_dim": 24,
  "action_chunk_size": 50,
  "proprio_history_size": 1,
  "num_img_obs": 1,
  "img_obs_every": 1,
  "norm_stats_file_path": ""
}
```

| Field | Description |
|---|---|
| `HZ` | Control loop frequency (Hz) |
| `max_delta_deg` | Maximum angular change per control step (degrees) for slew-rate limiting |
| `policy_update_period` | Number of control steps between policy inference calls |
| `mono_image_resize` | Target camera image resolution |
| `camera_names` | List of camera names matching controller bridge expectations |
| `proprio_state_dim` | Proprioceptive state dimensionality |
| `action_dim` | Action space dimensionality |
| `action_chunk_size` | Number of future actions predicted per inference call |
| `proprio_history_size` | Number of past proprioceptive frames to include |
| `num_img_obs` | Number of past image frames to include |
| `img_obs_every` | Image subsampling rate (1 = every frame) |
| `norm_stats_file_path` | Path to normalization statistics pickle file |

### `inference_runtime_topics.json`

Maps ROS2 topics to observation keys with field extraction rules:

```json
{
  "robot_id": "packy",
  "HZ": 20,
  "topics": {
    "joints": {
      "topic": "/igris_b/packy/joint_states",
      "msg_type": "JointState",
      "fields": {
        "/observation/joint_pos/left": {"slice": [6, 12], "attr": "position"},
        "/observation/joint_pos/right": {"slice": [0, 6], "attr": "position"}
      }
    },
    "finger": {
      "topic": "/igris_b/packy/finger_state",
      "msg_type": "Float32MultiArray",
      "fields": {
        "/observation/hand_joint_pos/left": {"slice": [6, 12]},
        "/observation/hand_joint_pos/right": {"slice": [0, 6]}
      }
    }
  }
}
```

Each topic entry specifies:
- `topic`: ROS2 topic name
- `msg_type`: ROS2 message type (`JointState`, `Float32MultiArray`, etc.)
- `fields`: Mapping from observation key to extraction rule (`slice` for array indexing, `attr` for message attribute)

### `inference_runtime_params.py`

The `RuntimeParams` class encapsulates all runtime parameters and provides computed properties:

```python
class RuntimeParams:
    HZ: int
    policy_update_period: int
    max_delta: float           # Converted from degrees to radians
    proprio_state_dim: int
    proprio_history_size: int
    camera_names: list[str]
    num_img_obs: int
    img_obs_every: int
    action_dim: int
    action_chunk_size: int
    mono_img_resize_width: int
    mono_img_resize_height: int

    def read_stats_file(self) -> dict   # Load normalization stats from pickle
```

### `init_params.py`

Defines initial joint positions and state observation keys:

```python
# IGRIS_B example
INIT_JOINT_LIST = [+20, +30, 0, -120, 0, 0,    # Right arm (degrees)
                   -20, -30, 0, +120, 0, 0]     # Left arm (degrees)
INIT_HAND_LIST = [1.0, 1.0, 1.0, 1.0, 1.0, 0.5,  # Right hand
                  1.0, 1.0, 1.0, 1.0, 1.0, 0.5]   # Left hand

IGRIS_B_STATE_KEYS = [
    "/observation/joint_pos/left",
    "/observation/joint_pos/right",
    "/observation/hand_joint_pos/left",
    "/observation/hand_joint_pos/right",
]
```

## Adding Configuration for a New Robot

1. Create `robots/your_robot/init_params.py` with initial joint positions and state keys
2. Create `robots/your_robot/inference_runtime_params.json` with timing and dimension parameters
3. Create `robots/your_robot/inference_runtime_topics.json` with ROS2 topic mappings
4. Create `robots/your_robot/inference_runtime_params.py` with a `RuntimeParams` class

Use the IGRIS_B configuration as a reference template.
