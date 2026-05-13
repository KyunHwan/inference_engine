# runtime_settings_configs

**Parent:** [env_actor](../README.md)

Per-robot runtime configuration. Each robot has its own subdirectory under [`robots/`](robots/) holding timing parameters, dimensions, ROS2 topic mappings, and initial joint positions.

## Table of contents

- [Structure](#structure)
- [Configuration files](#configuration-files)
- [Per-robot READMEs](#per-robot-readmes)
- [Extension points](#extension-points)
- [Related docs](#related-docs)

## Structure

```
runtime_settings_configs/
└── robots/
    ├── igris_b/
    │   ├── inference_runtime_params.json    # Timing, dimensions, cameras
    │   ├── inference_runtime_topics.json    # ROS2 topic → observation key mapping
    │   ├── inference_runtime_params.py      # RuntimeParams class
    │   ├── init_params.py                   # Initial joint positions, state keys
    │   └── README.md
    └── igris_c/
        ├── init_params.py                   # Placeholder (empty lists)
        └── README.md
```

## Configuration files

### `inference_runtime_params.json`

Controls inference timing, observation/action dimensions, and camera setup.

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
  "proprio_history_size": 50,
  "num_img_obs": 1,
  "img_obs_every": 1,
  "norm_stats_file_path": "/abs/path/to/dataset_stats.pkl"
}
```

| Field | Description |
|---|---|
| `HZ` | Control loop frequency (Hz) |
| `max_delta_deg` | Maximum angular change per control step (degrees) for slew-rate limiting |
| `policy_update_period` | Number of control steps between policy inference calls (Sequential only) |
| `mono_image_resize` | Target camera image resolution (`width`, `height`) |
| `camera_names` | List of camera names matching controller bridge expectations |
| `proprio_state_dim` | Proprioceptive state dimensionality |
| `action_dim` | Action space dimensionality |
| `action_chunk_size` | Number of future actions predicted per inference call |
| `proprio_history_size` | Number of past proprioceptive frames to include |
| `num_img_obs` | Number of past image frames to include |
| `img_obs_every` | Image subsampling rate (`1` = every frame) |
| `norm_stats_file_path` | Absolute path to normalization statistics pickle file |

### `inference_runtime_topics.json`

Maps ROS2 topics to observation keys with field extraction rules. Each entry has:

- `topic`: ROS2 topic name (after `robot_id` substitution).
- `msg_type`: ROS2 message type as a string (resolved by [data_dict.py](../robot_io_interface/robots/igris_b/utils/data_dict.py)).
- `fields`: mapping from observation key to extraction rule (`slice` for array indexing, `attr` for message attribute).

Full annotated example: [igris_b/inference_runtime_topics.json](robots/igris_b/inference_runtime_topics.json).

### `inference_runtime_params.py`

The `RuntimeParams` class. Wraps the JSON dict and exposes typed properties. Converts `max_delta_deg` → `max_delta` (radians) once at construction.

The class also provides `read_stats_file()` which `pickle.load`s `norm_stats_file_path`. The actors call this and pass the result to `DataNormalizationInterface(robot, data_stats=...)`.

### `init_params.py`

Defines `INIT_JOINT_LIST` (degrees), `INIT_HAND_LIST`, `INIT_JOINT` (radians) and the state-keys list (e.g. `IGRIS_B_STATE_KEYS`). Used by the controller bridge (for `init_robot_position` and `_obs_dict_to_np_array`) and by the RTC shared-memory bridge (for `init_action_chunk`).

## Per-robot READMEs

- [robots/igris_b/](robots/igris_b/README.md) — the canonical implementation, with annotated values.
- [robots/igris_c/](robots/igris_c/README.md) — what to fill in once hardware specs land.

## Extension points

To add configuration for a new robot:

1. Create `robots/your_robot/init_params.py` with initial joint positions and state keys.
2. Create `robots/your_robot/inference_runtime_params.json` with timing and dimension parameters.
3. Create `robots/your_robot/inference_runtime_topics.json` with ROS2 topic mappings.
4. Create `robots/your_robot/inference_runtime_params.py` with a `RuntimeParams` class (copy IGRIS_B's verbatim if your JSON shape matches).

Use the IGRIS_B configuration as a reference template. Full walkthrough: [docs/walkthroughs/04_add_a_new_robot.md § Phase 1 Runtime configs](../../docs/walkthroughs/04_add_a_new_robot.md#phase-1-runtime-configs).

## Related docs

- [docs/configuration_cookbook.md](../../docs/configuration_cookbook.md) — "if I want to change X, edit Y" recipes.
- [docs/api.md § RuntimeParams](../../docs/api.md#runtimeparams)
- [docs/walkthroughs/04_add_a_new_robot.md](../../docs/walkthroughs/04_add_a_new_robot.md)
