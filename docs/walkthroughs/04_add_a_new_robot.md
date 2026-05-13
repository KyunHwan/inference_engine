# Walkthrough 04 — Add a new robot

The longest walkthrough. We bring up IGRIS_C from the existing stubs. Order of operations is non-trivial: configs first, then the controller, then the data managers, then the normalization bridge, then the factory updates and the argparse line. 1–2 working days excluding hardware bring-up.

## Table of contents

- [What you need before you start](#what-you-need-before-you-start)
- [Phase 1: Runtime configs](#phase-1-runtime-configs)
- [Phase 2: Controller bridge](#phase-2-controller-bridge)
- [Phase 3: Data manager bridges (Sequential + RTC)](#phase-3-data-manager-bridges-sequential--rtc)
- [Phase 4: Normalization bridge](#phase-4-normalization-bridge)
- [Phase 5: Factory + argparse updates](#phase-5-factory--argparse-updates)
- [Phase 6: Verification](#phase-6-verification)

## What you need before you start

A hardware spec sheet. See the questions in [robots/igris_c/README.md](../../env_actor/robot_io_interface/robots/igris_c/README.md). At minimum:

- Joint counts (per-arm, per-hand).
- Action layout (joint space? joint deltas? quaternions?).
- Safe initial joint positions.
- ROS2 topic names + message types.
- Camera device paths + native resolution.
- Maximum allowed angular delta per control step.

Until those are filled in, do not write code — you will guess wrong and have to redo it.

## Phase 1: Runtime configs

These four files live under [env_actor/runtime_settings_configs/robots/your_robot/](../../env_actor/runtime_settings_configs/robots/igris_c/). Copy from `igris_b/` as a template.

### `init_params.py`

```python
import numpy as np

INIT_JOINT_LIST = [...]   # In degrees, length = number of arm joints (both sides)
INIT_HAND_LIST  = [...]   # Initial hand/finger targets, length = number of finger joints
INIT_JOINT = np.array(INIT_JOINT_LIST, dtype=np.float32) * np.pi / 180.0

YOUR_ROBOT_STATE_KEYS = [
    "/observation/joint_pos/left",
    "/observation/joint_pos/right",
    "/observation/hand_joint_pos/left",
    "/observation/hand_joint_pos/right",
    # Add joint_cur / hand_joint_cur if your stats pickle has those means/stds.
]
```

Reference: [igris_b/init_params.py](../../env_actor/runtime_settings_configs/robots/igris_b/init_params.py).

### `inference_runtime_params.json`

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
  "norm_stats_file_path": "/abs/path/to/your_robot_stats.pkl"
}
```

Adjust `proprio_state_dim` and `action_dim` to match your robot. Field meanings are in [runtime_settings_configs/README.md](../../env_actor/runtime_settings_configs/README.md).

### `inference_runtime_topics.json`

This maps ROS2 topics to observation keys. The format is described in [runtime_settings_configs/README.md § inference_runtime_topics.json](../../env_actor/runtime_settings_configs/README.md). The IGRIS_B file ([inference_runtime_topics.json](../../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_topics.json)) is your template.

Watch out for:

- `robot_id` — substituted into your topic names by the controller bridge (`/your_robot/{robot_id}/...`).
- `slice` — for ROS2 array fields, which range of indices to grab.
- `attr` — for ROS2 messages like `JointState`, which attribute (`position`, `effort`, …) to read.

### `inference_runtime_params.py`

The `RuntimeParams` class. Copy [igris_b/inference_runtime_params.py](../../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.py) verbatim — it should be the same for any robot whose JSON has the same fields.

## Phase 2: Controller bridge

[env_actor/robot_io_interface/robots/your_robot/controller_bridge.py](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py).

Use [igris_b/controller_bridge.py](../../env_actor/robot_io_interface/robots/igris_b/controller_bridge.py) as your template. Required methods and properties are in [api.md § ControllerInterface](../api.md#controllerinterface).

The IGRIS_B bridge does these things; your bridge must do equivalents:

1. **`__init__`** — call `rclpy.init()`, build the `GenericRecorder` from the topics config, create `joint_pub` and `finger_pub` (or whatever publishers your robot needs), spin up a `SingleThreadedExecutor` in a daemon thread.
2. **`read_state()`** — gather the latest values from `self.input_recorder.get_observation_dict()`, pack them in the order of `YOUR_ROBOT_STATE_KEYS` into a flat float32 vector, grab the latest frame from each camera, resize to `mono_img_resize`, transpose HWC→CHW, return the dict.
3. **`publish_action(action, prev_joint)`** — slice `action` into the parts your robot expects (joints, fingers), apply slew-rate limiting (`np.clip(target - prev, -max_delta, +max_delta)`), publish the messages, return the smoothed values.
4. **`start_state_readers()`** — open cameras, ensure the subscription queue has filled at least once.
5. **`init_robot_position()`** — publish a `JointState` whose `position` is `INIT_JOINT`, return that vector.
6. **`shutdown()`** — `executor.shutdown()`, `node.destroy_node()`, `rclpy.shutdown()`.

Camera handling: the IGRIS_B bridge uses [`RBRSCamera`](../../env_actor/robot_io_interface/robots/igris_b/utils/camera_utils.py) which itself opens V4L2 devices via OpenCV. Reuse it if your cameras are USB+V4L2; otherwise write a new wrapper that exposes `start()` and `get_image() → HWC uint8 numpy`.

## Phase 3: Data manager bridges (Sequential + RTC)

Two files to write, one for each algorithm.

### Sequential

[env_actor/auto/inference_algorithms/sequential/data_manager/robots/your_robot/data_manager_bridge.py](../../env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_b/data_manager_bridge.py)

Copy from [igris_b/data_manager_bridge.py](../../env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_b/data_manager_bridge.py). Required methods:

| Method | Purpose |
|---|---|
| `__init__(runtime_params)` | Save params, initialize empty history buffers |
| `init_inference_obs_state_buffer(init_data)` | Repeat the first observation `proprio_history_size` times to bootstrap |
| `update_state_history(obs_data)` | FIFO-shift `proprio` history, replace image history |
| `serve_raw_obs_state()` | Return a dict with copies of `proprio` and references to image buffers |
| `buffer_action_chunk(policy_output, t)` | Cache the latest chunk + `t` |
| `get_current_action(t)` | Return `chunk[clip(t - last_t, 0, chunk-1)]` |
| `serve_init_action()` | (Optional) bootstrap an init-action chunk if your RTC path needs one |

### RTC

[env_actor/auto/inference_algorithms/rtc/data_manager/robots/your_robot/shm_manager_bridge.py](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py)

This is the bigger of the two. Copy [igris_b/shm_manager_bridge.py](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py) and adjust:

- The `init_vec` ordering inside `init_action_chunk` and `init_action_chunk_obs_history` — IGRIS_B specifically reshuffles `INIT_JOINT_LIST` and `INIT_HAND_LIST` to match the policy's expected action layout. Yours will be different.
- Any `INIT_JOINT_LIST[6:] + INIT_JOINT_LIST[:6]`-style slicing: re-derive based on your action layout (`[L-arm, R-arm, L-finger, R-finger]`).

Do **not** change the synchronization API surface — those method signatures are what [shm_manager_interface.py](../../env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py) forwards to.

## Phase 4: Normalization bridge

[env_actor/nom_stats_manager/robots/your_robot/data_normalization_manager.py](../../env_actor/nom_stats_manager/robots/igris_b/data_normalization_manager.py).

Reuse the IGRIS_B class if your stats pickle has the same keys (`observation.state`, `observation.current`, `action`). Otherwise adjust which keys are concatenated and how.

Required methods (delegated by [DataNormalizationInterface](../../env_actor/nom_stats_manager/data_normalization_interface.py)):

- `normalize_state(state: dict) → dict`
- `normalize_action(action: np.ndarray) → np.ndarray`
- `denormalize_action(action: np.ndarray) → np.ndarray`

## Phase 5: Factory + argparse updates

You must touch six factories and one argparse line. They are all `if/elif robot == "your_robot":` blocks.

| File | Add block |
|---|---|
| [run_inference.py](../../run_inference.py) | argparse: `--robot ... choices=["igris_b", "igris_c", "your_robot"]` |
| [sequential_actor.py](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py) | `elif self.robot == "your_robot": from env_actor.runtime_settings_configs.robots.your_robot... import RuntimeParams` |
| [rtc_actor.py](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py) | same import |
| [control_loop.py](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py) | same import |
| [inference_loop.py](../../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py) | same import |
| [controller_interface.py](../../env_actor/robot_io_interface/controller_interface.py) | `elif robot == "your_robot": from .robots.your_robot.controller_bridge import ControllerBridge` |
| [data_manager_interface.py](../../env_actor/auto/inference_algorithms/sequential/data_manager/data_manager_interface.py) | same for sequential bridge |
| [shm_manager_interface.py](../../env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py) | same for RTC bridge |
| [data_normalization_interface.py](../../env_actor/nom_stats_manager/data_normalization_interface.py) | same for normalization bridge |

Check that you handled all four `if robot == "igris_b": ... elif robot == "igris_c": ...` blocks. The lints are simple — grep for `"igris_b"` and make sure each occurrence has an `elif` for your robot.

## Phase 6: Verification

In order. Do not skip ahead.

### 6.1 Configs load

```bash
python -c "
from env_actor.runtime_settings_configs.robots.your_robot.inference_runtime_params import RuntimeParams
import json
with open('env_actor/runtime_settings_configs/robots/your_robot/inference_runtime_params.json') as f:
    cfg = json.load(f)
r = RuntimeParams(cfg)
print(r.HZ, r.action_dim, r.proprio_state_dim, r.max_delta)
"
```

Should print your values. Catches typos in property names.

### 6.2 Bridges import

```bash
python -c "
from env_actor.robot_io_interface.controller_interface import ControllerInterface
# Don't instantiate yet — rclpy.init() will run. Just check the import path works.
print('ok')
"
```

Same for the other three bridges.

### 6.3 Argparse accepts the new robot

```bash
python run_inference.py --robot your_robot --help
```

Should not error. (Just `--help`, not actually launching.)

### 6.4 ROS2 mock test

If you have `ros2 bag play` access to a recording, replay it and run with `--inference_algorithm sequential`. Without a real publisher, `_check_proprio_reading` will print `Waiting for proprio data to come in...` forever — that confirms the subscription is alive; you just need data on the topic.

### 6.5 Cameras

Independently:

```bash
python -c "
from env_actor.robot_io_interface.robots.your_robot.utils.camera_utils import YourCamera
cam = YourCamera('/dev/your_robot_head_camera')
cam.start()
import time; time.sleep(1)
img = cam.get_image()
print(img.shape if img is not None else 'no frame')
"
```

### 6.6 `read_state()` smoke test

Build a `ControllerInterface(...)`, call `start_state_readers()`, wait a second, call `read_state()`. The returned dict should have `proprio` of shape `(state_dim,)` and one image per camera at `(3, H, W)`.

### 6.7 `init_robot_position()` smoke test

Call `init_robot_position()`. On the robot side, verify (in the safest way possible — e.g., e-stop ready, current limits low) that the robot moves to the expected pose and stops.

### 6.8 Dry inference

```bash
python run_inference.py --robot your_robot --inference_algorithm sequential \
  -P env_actor/policy/policies/openpi_policy/openpi_policy.yaml
```

If you don't have a checkpoint trained on your robot, expect a shape mismatch when the policy expects `action_dim=24` and you're publishing `action_dim=12`. That's progress — it means the loader, the bridges, and the factories all work; you just need a checkpoint trained on your robot.

### 6.9 Full integration (with a checkpoint)

Once a checkpoint exists, try Sequential first, then RTC. Check actions are reasonable in `ros2 topic echo` before letting the robot execute them.

## What to read while implementing

- [api.md](../api.md) — the contracts you are implementing.
- [rtc_shared_memory.md](../rtc_shared_memory.md) — for the RTC bridge.
- [robots/igris_c/README.md](../../env_actor/robot_io_interface/robots/igris_c/README.md) — the original spec checklist.
- [development.md § New robot](../development.md#new-robot) — the high-density version of this walkthrough.
