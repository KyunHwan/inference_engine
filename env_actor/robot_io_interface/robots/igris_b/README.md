# robot_io_interface/robots/igris_b

**Parent:** [robot_io_interface](../../README.md)

The full IGRIS_B robot bridge. ROS2 publishers/subscribers, V4L2 cameras, slew-rate limiting, init-pose handling.

## Table of contents

- [Files](#files)
- [ControllerBridge overview](#controllerbridge-overview)
- [Action layout and the right/left flip](#action-layout-and-the-rightleft-flip)
- [Cameras](#cameras)
- [Slew-rate limiting](#slew-rate-limiting)
- [GenericRecorder (rclpy subscription manager)](#genericrecorder-rclpy-subscription-manager)
- [Related docs](#related-docs)

## Files

| File | Purpose |
|---|---|
| [`controller_bridge.py`](controller_bridge.py) | `ControllerBridge` class implementing the contract in [robot_io_interface § Required bridge methods](../../README.md#required-bridge-methods). |
| [`utils/data_dict.py`](utils/data_dict.py) | `GenericRecorder(Node)` — generic ROS2 subscription manager driven by the topics JSON. |
| [`utils/camera_utils.py`](utils/camera_utils.py) | `RBRSCamera` — V4L2 USB camera wrapper. Supports up to two devices (`device_id1`, `device_id2`); for IGRIS_B each `cam_name` uses only one. |

## ControllerBridge overview

`__init__`:

- `rclpy.init()` if not already initialized.
- Create the `GenericRecorder` (`Node`) from the topics JSON.
- Create publishers for joints (`JointState` on `/igris_b/<robot_id>/target_joints`) and fingers (`Float32MultiArray` on `/igris_b/<robot_id>/finger_target`).
- QoS: `reliability = RELIABLE`, `depth = 10`.
- Spin up a `SingleThreadedExecutor` with the recorder added; the executor will be driven in a daemon thread when `start_state_readers` is called.

`start_state_readers()`:

- `_start_cam_recording()` — opens cameras per `runtime_params.camera_names`. Each `RBRSCamera` itself launches a per-camera background thread to keep `merged_frame` fresh.
- `_start_proprio_recording()` — daemon thread running `self.executor.spin()`.
- `_check_proprio_reading()` — busy-wait until every key in the observation-keys subset of `recorder.get_dict()` is non-`None`.

`init_robot_position()` publishes a `JointState` with `position = INIT_JOINT.copy()` and returns `INIT_JOINT.copy()`.

`read_state()` returns `{proprio, head, left, right}`. Proprio packs `IGRIS_B_STATE_KEYS` (4 keys × 6 dims = 24) into a flat float32 via `_obs_dict_to_np_array`. Images are pulled from each camera's `get_image()` (returns the merged 800×600 RGB frame from `RBRSCamera`), resized to `mono_img_resize` with `cv2.resize(..., interpolation=cv2.INTER_AREA)`, transposed HWC→CHW.

`publish_action(action, prev_joint)`:

```python
left_joint_pos  = action[:6]
right_joint_pos = action[6:12]
left_finger_pos = action[12:18]
right_finger_pos= action[18:24]

raw_joint = np.concatenate([right_joint_pos, left_joint_pos])   # right then left for the ROS message
delta = np.clip(raw_joint - prev_joint, -max_delta, +max_delta)
smoothed_joints = prev_joint + delta

JointState(position=smoothed_joints.tolist()) → /target_joints
Float32MultiArray(data=list(right_finger_pos) + list(left_finger_pos)) → /finger_target

return smoothed_joints, np.concatenate([left_finger_pos, right_finger_pos])
```

`shutdown()` shuts the executor, destroys the recorder node, and calls `rclpy.shutdown()`.

## Action layout and the right/left flip

The 24-d action vector is `[L-arm, R-arm, L-finger, R-finger]` from the policy's perspective. The on-robot ROS messages, however, want **right then left** for the joint stream and **right then left** for the finger stream. The bridge does the swap inside `publish_action`. The reverse mapping also shows up inside the RTC `SharedMemoryManager.init_action_chunk` (the `INIT_JOINT_LIST[6:] + INIT_JOINT_LIST[:6]` reshuffle).

When adding a new robot, decide your action layout once and document it. Don't mix the orderings across files.

## Cameras

`RBRSCamera` opens one or two V4L2 devices and merges their frames. For IGRIS_B each camera name maps to one physical device:

```python
if cam_name in ['head', 'right']:
    RBRSCamera(device_id1=f"/dev/{cam_name}_camera1", device_id2=None)
elif cam_name == 'left':
    RBRSCamera(device_id1=None, device_id2=f"/dev/{cam_name}_camera2")
```

`singleCamera` sets MJPG @ 1600×1200 @ 60 FPS, rotates 180°, resizes to 800×600, converts BGR→RGB. The bridge then re-resizes to `mono_img_resize` (default 320×240) before transposing to CHW.

If a frame is `None` (camera disconnected or `read()` returned `False`), `get_image()` returns `None` and the bridge logs `<cam_name>: image is None !!`.

## Slew-rate limiting

`np.clip(raw_joint - prev_joint, -max_delta, +max_delta)`. `max_delta` is in radians, computed from `max_delta_deg` in the JSON. Default 5 degrees → about 0.087 rad/step → about 1.75 rad/s at 20 Hz.

This is a per-joint clamp, not a global magnitude clamp. If you need cartesian-space safety, layer it on top.

## GenericRecorder (rclpy subscription manager)

[`utils/data_dict.py`](utils/data_dict.py) reads the topics JSON and:

- For each topic, resolves the message class via a small `mapping` dict (`JointState → sensor_msgs.msg`, etc.).
- For each field (e.g. `/observation/joint_pos/left`), subscribes once and stores the latest value in `self.data[key]`.
- Slice and attribute extraction is driven by the JSON's `slice` and `attr` keys.
- `get_observation_dict()` returns only the entries whose key starts with `/observation/`.
- A timer ticks at `DT * 10` to warn if image topics are stale.

Currently the IGRIS_B configuration only enables joint topics (`joint_pos`, `hand_joint_pos`, and optionally `joint_cur` / `hand_joint_cur`). Images come from V4L2, not ROS2.

## Related docs

- [docs/api.md § ControllerInterface](../../../../docs/api.md#controllerinterface)
- [docs/concepts.md § ROS2 primer](../../../../docs/concepts.md#ros2-primer-scoped-to-this-codebase)
- [docs/troubleshooting.md § Cameras](../../../../docs/troubleshooting.md#cameras) · [§ ROS2](../../../../docs/troubleshooting.md#ros2)
- [docs/walkthroughs/02_trace_one_step.md](../../../../docs/walkthroughs/02_trace_one_step.md) — `read_state` and `publish_action` show up in step 1 / step 8.
