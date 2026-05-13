# runtime_settings_configs/robots/igris_b

**Parent:** [runtime_settings_configs](../../README.md)

The IGRIS_B configuration. All four files here together define a complete runtime config.

## Table of contents

- [Files](#files)
- [inference_runtime_params.json (annotated)](#inference_runtime_paramsjson-annotated)
- [inference_runtime_topics.json (annotated)](#inference_runtime_topicsjson-annotated)
- [inference_runtime_params.py (RuntimeParams)](#inference_runtime_paramspy-runtimeparams)
- [init_params.py](#init_paramspy)
- [Related docs](#related-docs)

## Files

| File | Format | Purpose |
|---|---|---|
| [`inference_runtime_params.json`](inference_runtime_params.json) | JSON | Timing, dimensions, cameras, stats path |
| [`inference_runtime_topics.json`](inference_runtime_topics.json) | JSON | ROS2 topics + field extraction rules |
| [`inference_runtime_params.py`](inference_runtime_params.py) | Python | `RuntimeParams` class that wraps the JSON |
| [`init_params.py`](init_params.py) | Python | `INIT_JOINT`, `INIT_JOINT_LIST`, `INIT_HAND_LIST`, `IGRIS_B_STATE_KEYS` |

## inference_runtime_params.json (annotated)

```json
{
  "HZ": 20,                             // 20 Hz control = 50 ms per step
  "max_delta_deg": 5,                   // Joints clamped to ±5° per step  (= ~0.087 rad)
  "policy_update_period": 50,           // Sequential: predict() every 2.5 s
  "mono_image_resize": {"width": 320, "height": 240},
  "camera_names": ["head", "left", "right"],
  "proprio_state_dim": 24,              // 4 state keys × 6 dims each — must match IGRIS_B_STATE_KEYS
  "action_dim": 24,                     // Must match the policy's action_dim and the controller's action layout
  "action_chunk_size": 50,              // Must match the policy's action_horizon
  "proprio_history_size": 50,           // Buffered proprio frames, FIFO with newest at [0]
  "num_img_obs": 1,                     // Only latest image per camera
  "img_obs_every": 1,                   // No subsampling
  "norm_stats_file_path": "/home/robros/Projects/inference_engine/trainer/experiment_training/reinforcement_learning/dsrl_openpi/exp1/dataset_stats.pkl"
}
```

To change any of these safely, see [docs/configuration_cookbook.md](../../../../docs/configuration_cookbook.md).

## inference_runtime_topics.json (annotated)

```json
{
  "robot_id": "packy",                  // Substituted into /igris_b/<robot_id>/...
  "HZ": 20,
  "topics": {
    "finger": {
      "topic": "/igris_b/packy/finger_state",
      "msg_type": "Float32MultiArray",
      "fields": {
        "/observation/hand_joint_pos/left":  {"slice": [6, 12]},
        "/observation/hand_joint_pos/right": {"slice": [0, 6]}
      }
    },
    "finger_current": {                 // Currents — published but not in IGRIS_B_STATE_KEYS by default
      "topic": "/igris_b/packy/finger_current",
      "msg_type": "Float32MultiArray",
      "fields": {
        "/observation/hand_joint_cur/left":  {"slice": [6, 12]},
        "/observation/hand_joint_cur/right": {"slice": [0, 6]}
      }
    },
    "joints": {
      "topic": "/igris_b/packy/joint_states",
      "msg_type": "JointState",
      "fields": {
        "/observation/joint_pos/left":  {"slice": [6, 12], "attr": "position"},
        "/observation/joint_pos/right": {"slice": [0, 6],  "attr": "position"},
        "/observation/joint_cur/left":  {"slice": [6, 12], "attr": "effort"},   // Currents from JointState.effort
        "/observation/joint_cur/right": {"slice": [0, 6],  "attr": "effort"}
      }
    }
  }
}
```

The `slice` indexing on the published `JointState`/`Float32MultiArray` is **right then left** (right at `[0:6]`, left at `[6:12]`). The slicer extracts them into separate observation keys. Then `_obs_dict_to_np_array` in the controller bridge re-packs into IGRIS_B_STATE_KEYS order.

## inference_runtime_params.py (RuntimeParams)

Plain wrapper class. Each JSON field becomes a `@property`. `max_delta` is computed once as `np.deg2rad(max_delta_deg)`. `read_stats_file()` lazy-loads the pickle.

The class's constructor takes the **dict**, not the path. Both algorithms parse the JSON before constructing `RuntimeParams(dict)`.

## init_params.py

```python
INIT_JOINT_LIST = [+20, +30, 0, -120, 0, 0,    # Right arm in degrees (6 joints)
                   -20, -30, 0, +120, 0, 0]    # Left arm
INIT_HAND_LIST = [1.0]*5 + [0.5,                # Right hand (6 fingers)
                   1.0]*5 + [0.5]               # Left hand

INIT_JOINT = np.array(INIT_JOINT_LIST, dtype=np.float32) * np.pi / 180.0   # radians

IGRIS_B_STATE_KEYS = [
    "/observation/joint_pos/left",
    "/observation/joint_pos/right",
    "/observation/hand_joint_pos/left",
    "/observation/hand_joint_pos/right",
    # Joint currents and hand currents are commented out — uncomment if your norm-stats pickle has them
    # "/observation/joint_cur/left",
    # "/observation/joint_cur/right",
    # "/observation/hand_joint_cur/left",
    # "/observation/hand_joint_cur/right",
]
```

`INIT_JOINT_LIST` is right-then-left, but the **policy** outputs left-then-right joints. Anywhere we use `INIT_JOINT_LIST` as a *policy-order* template (in the RTC `init_action_chunk`, in the sequential `serve_init_action`), we reshuffle: `INIT_JOINT_LIST[6:] + INIT_JOINT_LIST[:6]`.

`IGRIS_B_STATE_KEYS` has 4 entries; with each entry being a 6-d slice this matches `proprio_state_dim = 24`. If you uncomment the current keys, bump `proprio_state_dim` to 48 in the JSON and re-train the stats pickle accordingly.

## Related docs

- [docs/configuration_cookbook.md](../../../../docs/configuration_cookbook.md) — "if I want to change X, edit Y".
- [docs/api.md § Key data shapes](../../../../docs/api.md#key-data-shapes-igris_b-defaults)
- [robot_io_interface/robots/igris_b/README.md](../../../robot_io_interface/robots/igris_b/README.md) — the consumer of these configs.
