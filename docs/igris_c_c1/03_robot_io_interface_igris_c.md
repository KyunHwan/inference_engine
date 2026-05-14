# 03 — Robot I/O interface: the IGRIS_C controller bridge

**What this covers.** The DDS-based controller bridge at [`env_actor/robot_io_interface/robots/igris_c/controller_bridge.py`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py). How it discovers the robot over Cyclone DDS, what it reads, what it writes, the action layout, the proprio layout, the lifecycle hooks, and where it differs from the IGRIS_B `rclpy`-based bridge.
**Who this is for.** Anyone changing topic names, action dims, gain values, camera layouts, or DDS network configuration.

The bridge is **not** robot-control software in the academic sense — it is a thin, single-process I/O adapter that the `ControllerInterface` factory ([`controller_interface.py:1-12`](../../env_actor/robot_io_interface/controller_interface.py)) selects via `if robot == "igris_c"`. It exposes the same surface as the IGRIS_B bridge (`start_state_readers`, `read_state`, `publish_action`, `init_robot_position`, `shutdown`) so the algorithm layers don't need to know which robot they're driving.

## Table of contents

- [DDS setup](#dds-setup)
- [Topics: what the bridge subscribes and publishes](#topics-what-the-bridge-subscribes-and-publishes)
- [`read_state()` — what comes out](#read_state--what-comes-out)
- [`publish_action()` — what goes in](#publish_action--what-goes-in)
- [Camera initialization](#camera-initialization)
- [Lifecycle hooks](#lifecycle-hooks)
- [State packing layout (the 86-D proprio vector)](#state-packing-layout-the-86-d-proprio-vector)
- [Health monitoring](#health-monitoring)
- [Bridge-only test](#bridge-only-test)
- [IGRIS_B vs IGRIS_C — side by side](#igris_b-vs-igris_c--side-by-side)

## DDS setup

Cyclone DDS configuration on this branch is split across:

| File | Purpose |
|---|---|
| [`controller_bridge.py:104-139`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L104) | Constructs two `DomainParticipant`s (state @ Domain 0, camera @ Domain 1). Sets `os.environ["CYCLONEDDS_URI"]` to the **contents** of `dds_state.xml` so libcyclonedds finds it. |
| [`dds/dds_state.xml`](../../env_actor/runtime_settings_configs/robots/igris_c/dds/dds_state.xml) | Pins Domain 0 to a specific NIC. **Current value on this branch — verify against your environment.** |
| [`dds/dds_camera.xml`](../../env_actor/runtime_settings_configs/robots/igris_c/dds/dds_camera.xml) | Pins Domain 1 to a specific NIC. |
| [`inference_runtime_params.json:18-32`](../../env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json#L18) | Source of truth for domain IDs and XML paths. |

Two QoS profiles ([`controller_bridge.py:120-127`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L120)):

```python
sensor_qos = Qos(
    Policy.Reliability.Reliable(duration(seconds=1)),
    Policy.History.KeepLast(10),
)
camera_qos = Qos(
    Policy.Reliability.BestEffort,
    Policy.History.KeepLast(1),
)
```

Rationale (from the docstring): sensor traffic must be reliable so commands and state are never dropped; camera frames at 30 Hz are useless if stale, so `BestEffort + KeepLast(1)` discards them aggressively.

> **Why poll threads, not callbacks?** `cyclonedds-python` does not expose an `on_data_available` callback. The bridge spawns one daemon thread per reader, each running `reader.take(N=10)` in a tight loop with a `1 ms` wait between iterations (see `_spawn` at [`controller_bridge.py:148-160`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L148)). Each thread writes the latest sample into a `_Latest` cache; the main loop reads them under the cache's lock. This mirrors the pattern used by `record-BE`'s `DDSParticipantContext`, which is already proven against the NUC.

## Topics: what the bridge subscribes and publishes

Topic names are read from `runtime_params.dds_topics` (originally [`inference_runtime_params.json:23-31`](../../env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json#L23)) — **not** from `inference_runtime_topics.json`, which for IGRIS_C only carries `robot_id` and `HZ` (the `_note` in that file documents this).

Current values on this branch (verify against your environment):

### Subscribed (readers)

| Logical name | Topic | Type | QoS | Where cached |
|---|---|---|---|---|
| `lowstate`   | `rt/lowstate`              | `LowState`            | sensor   | `_latest_low` ([line 134](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L134)) |
| `handstate`  | `rt/handstate`             | `HandState`           | sensor   | `_latest_hand` ([line 134](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L134)) |
| `head_camera`  | `igris_c/sensor/eyes_stereo` | `CompressedMessage` | camera | `_latest_img["head"]` ([lines 135-139](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L135)) |
| `left_camera`  | `igris_c/sensor/left_hand`   | `CompressedMessage` | camera | `_latest_img["left"]` |
| `right_camera` | `igris_c/sensor/right_hand`  | `CompressedMessage` | camera | `_latest_img["right"]` |

The per-reader callbacks extract:
- `LowState`: builds `q` (length 31, from `joint_state[i].q`) and `tau` (length 31, from `motor_state[i].tau_est`) — see [lines 166-173](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L166).
- `HandState`: builds 12-element `q` and `tau` arrays positionally (the NUC's `MotorState` has no `id` field, so the bridge reads `seq[i].q` by index — see [lines 179-189](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L179)).
- Cameras: stores the raw JPEG bytes (`msg.image_data`) — see [lines 202-205](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L202).

### Published (writers)

| Logical name | Topic | Type | QoS | Where written |
|---|---|---|---|---|
| `lowcmd`  | `rt/lowcmd`  | `LowCmd`  | sensor | `_publish_lowcmd` ([lines 491-507](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L491)) |
| `handcmd` | `rt/handcmd` | `HandCmd` | sensor | `_publish_handcmd` ([lines 509-522](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L509)) |

## `read_state()` — what comes out

Signature: [`controller_bridge.py:346`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L346):

```python
def read_state(self) -> dict:
    """Returns: {
        'proprio': np.float32 array of shape (86,),
        'head':    np.uint8 array of shape (3, H, W) — RGB-ish (BGR from OpenCV),
        'left':    np.uint8 array of shape (3, H, W),
        'right':   np.uint8 array of shape (3, H, W),
    }
    H = mono_img_resize_height, W = mono_img_resize_width
    (current branch values: 224, 224 — see inference_runtime_params.json:6-7)
    """
```

How proprio is assembled ([lines 347-363](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L347)):

```python
low  = self._latest_low.get()  # (q_31, tau_31) or None
hand = self._latest_hand.get() # (q_12, tau_12) or None

body_q   = q_31      if low  is not None else np.zeros(31, np.float32)
body_tau = tau_31    if low  is not None else np.zeros(31, np.float32)
hand_q   = q_12      if hand is not None else np.zeros(12, np.float32)
hand_tau = tau_12    if hand is not None else np.zeros(12, np.float32)

proprio = np.concatenate([body_q, hand_q, body_tau, hand_tau], dtype=np.float32)
# shape: (31 + 12 + 31 + 12,) == (86,)
```

If `low` or `hand` is `None`, the bridge zero-fills (a one-time "hand is None !!" print accompanies a missing hand sample at [line 361](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L361)). This means the caller can begin polling before the first sample lands — but it also means **the first inference step can run against zero observations** unless `start_state_readers()` blocked first.

Per-camera image handling ([lines 366-383](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L366)):

1. `np.frombuffer(jpeg_bytes, dtype=uint8)`.
2. `cv2.imdecode(..., cv2.IMREAD_COLOR)`. If decode fails, prints `<key> image is None !!` and zero-fills.
3. **Head camera only**: `cv2.rotate(..., ROTATE_180)` then `frame[:, :w//2, :]` (left half — the stereo IDL puts the two eyes side-by-side, the bridge takes only the left eye).
4. `cv2.resize(..., (mono_img_resize_width, mono_img_resize_height), INTER_AREA)`.
5. `np.transpose(frame, (2, 0, 1))` → `(C, H, W)`.

`TODO:` document the stereo IDL frame layout in [10_glossary_and_references.md](10_glossary_and_references.md). The rotation + crop logic at lines 378-381 is camera-mount-dependent; if the head camera is re-mounted, this is the place to update.

## `publish_action()` — what goes in

Signature: [`controller_bridge.py:386`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L386):

```python
def publish_action(self, action: np.ndarray, prev_joint: np.ndarray):
    """
    action     : np.ndarray of shape (17,)  — assertion at line 389
    prev_joint : np.ndarray of shape (31,)  — last published 31-D body command
    returns    : (smoothed31, hand12) where smoothed31 is the 31-D body
                 command after slew-rate limiting, and hand12 is the
                 broadcast 12-D hand command
    """
```

The 17-D action layout (see the bridge docstring at [`controller_bridge.py:12-17`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L12)):

| Indices | Meaning |
|---|---|
| `0:7`   | Left arm joints (in the order of `LEFT_ARM_IDS = [15..21]`) |
| `7:14`  | Right arm joints (in the order of `RIGHT_ARM_IDS = [22..28]`) |
| `14`    | Left hand position — **broadcast to all 6 left fingers** |
| `15`    | Right hand position — broadcast to all 6 right fingers |
| `16`    | Waist yaw — joint index 0 |

Per-step logic ([lines 387-410](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L387)):

1. **Lift to 31-D body command.** Start with `HOME_POSE_RAD.copy()` (all fixed joints held at home). Slot in:
   ```python
   raw31[LEFT_ARM_IDS]  = action[0:7]
   raw31[RIGHT_ARM_IDS] = action[7:14]
   raw31[WAIST_YAW_ID]  = action[16]
   ```
2. **Slew-rate limit** the 31-D delta from `prev_joint`:
   ```python
   delta      = np.clip(raw31 - prev_joint, -max_delta, +max_delta)
   smoothed31 = prev_joint + delta
   ```
   `max_delta = np.deg2rad(max_delta_deg)`, where `max_delta_deg` defaults to **5°** per control step on this branch ([`inference_runtime_params.json:3`](../../env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json#L3)). At 20 Hz, that caps each joint to ~100°/s.
3. **Publish the body command** via `_publish_lowcmd(smoothed31)` — a `LowCmd(kinematic_mode=PJS, motors=[MotorCmd(...)*31])` with per-motor `kp`/`kd` from `runtime_params.joint_kp` / `joint_kd`.
4. **Publish the hand command** via `_publish_handcmd(action[14], action[15])` — broadcasts each scalar to all 6 finger motors per side, with `hand_kp` / `hand_kd` (defaults: 50.0 / 2.0).
5. **Return** `(smoothed31, hand12)` so the caller updates its `prev_joint` for the next step.

> **Why broadcast the hand value to 6 motors?** Recorded teleop on IGRIS_C uses a single grasp scalar per hand; the firmware then distributes that single scalar across all six finger motors with per-motor gain. Mirroring that at inference time keeps the action vector small (17-D total) and avoids the policy learning per-finger dynamics it never saw during training.

## Camera initialization

There is no V4L2 capture path on this branch. Cameras arrive over DDS as JPEG-encoded frames on three topics. **There are no `/dev/...` device paths to configure for IGRIS_C.** The IGRIS_B bridge uses `/dev/<cam_name>_camera{1,2}` (see [`igris_b/controller_bridge.py:130-145`](../../env_actor/robot_io_interface/robots/igris_b/controller_bridge.py#L130)) — that whole code path is absent for IGRIS_C.

What you **do** need to verify:

| Item | Where | Current value on this branch |
|---|---|---|
| Camera topic names | `inference_runtime_params.json:28-30` | `igris_c/sensor/eyes_stereo`, `igris_c/sensor/left_hand`, `igris_c/sensor/right_hand` |
| Camera domain ID | `inference_runtime_params.json:20` | `1` |
| Camera DDS XML path | `inference_runtime_params.json:22` | `/home/robros/Projects/inference_engine/env_actor/runtime_settings_configs/robots/igris_c/dds/dds_camera.xml` |
| Camera NIC | `dds/dds_camera.xml:18` | `enp11s0` |

If your operator setup routes cameras via the NUC bridge to WiFi (Domain 10), the comment block in `dds_camera.xml` documents the two changes you need (the XML and the `camera_domain_id` in JSON).

`TODO:` confirm with the operator team which DDS topology your environment uses (direct Jetson LAN vs. NUC-routed WiFi) before deploying.

## Lifecycle hooks

`ControllerInterface` exposes five lifecycle methods that the inference engine drives in order. All five are forwarded to `controller_bridge`:

| Hook | When called | What the IGRIS_C bridge does |
|---|---|---|
| `start_state_readers()` | First thing inside `start()` of every actor | Waits up to **10 s** for `_latest_low`, `_latest_hand`, and all three camera caches to contain at least one sample. If any is still empty, raises `RuntimeError`. See [lines 321-338](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L321). |
| `init_robot_position()` | After warmup, before the control loop | Linearly interpolates from the current `body_q` to a hard-coded `start_position` over **100 steps × 50 ms** (so 5 s total), publishing a `LowCmd` per step. Then publishes a `HandCmd` with both hands at 0.0. Returns the final 31-D pose. See [lines 340-344, 412-489](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L340). |
| `read_state()` | Each control iteration | See above. |
| `publish_action(action, prev_joint)` | Each control iteration after `t > 100` | See above. |
| `shutdown()` | On Ctrl+C or actor death | Stops the health thread and all poll threads; calls `join(timeout=2.0)` on each. See [lines 524-532](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L524). |

Note that the IGRIS_C `init_robot_position()` ramps the **current state → a hard-coded pose**. If you want a different start pose, change the `start_position` array at [lines 418-450](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L418). Note that this hard-coded array is **not** the `HOME_POSE_RAD` from `init_params.py` — it is a separate, recorded "good first pose." `TODO:` consider either reading this from config or making it identical to `HOME_POSE_RAD` to avoid two sources of truth.

The bridge does **not** implement a separate `recorder_rate_controller()` (the IGRIS_B one wraps `rclpy`'s `create_rate(HZ)`). For IGRIS_C, the property `recorder_rate_controller` ([lines 303-319](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L303)) returns a `time.perf_counter()`-based rate limiter — same effect, no `rclpy` dependency.

## State packing layout (the 86-D proprio vector)

`proprio` is `body_q ⊕ hand_q ⊕ body_tau ⊕ hand_tau`. There is **no** intermediate `_obs_dict_to_np_array()` step like on IGRIS_B — the bridge concatenates raw arrays directly. The dimensions:

| Slice | Source | Length | DDS unit |
|---|---|---|---|
| `[0:31]`   | `LowState.joint_state[i].q`   | 31 | rad (PJS) |
| `[31:43]`  | `HandState.motor_state[i].q`  | 12 | normalized 0–1 (per the NUC schema docstring at [`igris_c_msgs.py:122-127`](../../env_actor/robot_io_interface/robots/igris_c/messages/igris_c_msgs.py#L122)) |
| `[43:74]`  | `LowState.motor_state[i].tau_est` | 31 | Nm (PJS via MS field reuse — recorder convention; see `controller_bridge.py:14`) |
| `[74:86]`  | `HandState.motor_state[i].tau_est` | 12 | current mA (per the NUC schema docstring) |

The names declared in `init_params.py:81` (`["body_q_31", "hand_q_12", "body_tau_31", "hand_tau_12"]`) reflect this layout but are not used at runtime — the bridge concatenates by position.

The `run_bridge_monitor.py:42-45` script also documents these slices explicitly:

```python
body_q   = proprio[0:31]
hand_q   = proprio[31:43]
body_tau = proprio[43:74]
hand_tau = proprio[74:86]
```

> **Why mix units in a single vector?** The trainer's `dataset_stats.pkl` was computed against the **same** concatenated vector, so its mean/std bake in the unit handling. The policy never sees raw units — it sees normalized inputs. See [06_normalization_igris_c.md](06_normalization_igris_c.md).

## Health monitoring

Optional opt-in: set `BRIDGE_HEALTH_LOG=1` (or `=<seconds>` for the print period) before launching, and the bridge will spawn a daemon thread that prints per-topic Hz + sample age every period_sec:

```
[bridge-health] lowstate=200.0Hz(age  0.00s)  handstate=100.0Hz(age  0.01s)  head_cam= 30.0Hz(age  0.03s)  ...
```

See [`controller_bridge.py:228-235`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L228) for the env-var parser and [`enable_health_log()` at lines 261-293](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L261) for the implementation. You can also call `controller_bridge.topic_health()` programmatically to get the same snapshot as a dict.

`TODO:` flag this opt-in in your team's runbook — it is the fastest way to confirm a NIC-pinning issue.

## Bridge-only test

The branch ships [`env_actor/robot_io_interface/robots/igris_c/run_bridge_monitor.py`](../../env_actor/robot_io_interface/robots/igris_c/run_bridge_monitor.py) — a 60-line script that starts the bridge without the policy stack and prints a rolling state summary:

```bash
source .venv/bin/activate
python -m env_actor.robot_io_interface.robots.igris_c.run_bridge_monitor
```

Run this **before** trying to run inference. If the bridge can't see the robot, the policy stack will never get past `start_state_readers()`.

**One caveat:** the `CFG` constant at [line 16](../../env_actor/robot_io_interface/robots/igris_c/run_bridge_monitor.py#L16) is an absolute path on the author's machine. If your repo is not at `/home/robros/Projects/inference_engine`, edit `CFG` or pass `--inference_runtime_params_config` (currently the script does not accept CLI flags — `TODO:` add CLI args).

## IGRIS_B vs IGRIS_C — side by side

| Aspect | IGRIS_B | IGRIS_C |
|---|---|---|
| Wire protocol | ROS2 (`rclpy`) | Cyclone DDS (`cyclonedds-python`) |
| Robot connection | `/igris_b/<robot_id>/...` topics on the local ROS graph | `rt/lowstate`, `rt/handstate`, `rt/lowcmd`, `rt/handcmd`, plus `igris_c/sensor/*` |
| Camera input | V4L2 devices (`/dev/head_camera1`, `/dev/left_camera2`, `/dev/right_camera1`) | DDS topics (`igris_c/sensor/eyes_stereo` etc.) — JPEG bytes |
| Camera capture | `RBRSCamera` USB wrapper | `cv2.imdecode` |
| Action dim | 24 (6 left arm + 6 right arm + 6 left fingers + 6 right fingers) | 17 (7+7+1+1+1) |
| Hand control | Per-finger value | Broadcast scalar to all 6 fingers per side |
| Body command | Sent as `JointState.position`, 12-D | Sent as `LowCmd.motors[31]`, all 31 motors filled (fixed joints at home) |
| Kinematic mode | n/a (ROS-side abstraction) | `PJS` (Parallel Joint Space) explicit in every `LowCmd` |
| Proprio dim | 24 | 86 |
| Proprio layout | 8 keys × 6 elements (`IGRIS_B_STATE_KEYS`) | 4 fixed slices (`body_q, hand_q, body_tau, hand_tau`) |
| State-key file | `inference_runtime_topics.json` (full topic+slice map) | `inference_runtime_params.json:dds.topics` (just topic names); the `_topics.json` file is a stub |
| Network pinning | `ROS_DOMAIN_ID` env | `CYCLONEDDS_URI` env (XML pinning a NIC), set by the bridge from `dds_state.xml` |
| Lifecycle threads | `rclpy.executors.SingleThreadedExecutor` + 1 spin thread | One poll thread per DDS reader (5 total), plus an optional health thread |
| Init pose | `INIT_JOINT` (from `init_params.py`) published once | Linearly interpolated from current to `start_position` over 5 s |
| `_obs_dict_to_np_array` | Yes, splits 8 keys × 6 elements | Not used — bridge concatenates raw arrays directly |
| `recorder_rate_controller` | `input_recorder.create_rate(HZ)` (rclpy) | Inline `time.perf_counter()`-based class |

This table is the single best reference for someone porting another robot — start from this layout, pick whichever wire protocol matches your hardware, and follow the IGRIS_B side for ROS2 or the IGRIS_C side for DDS.

---

← Back to index: [README.md](README.md) · Next → [04_runtime_configuration_igris_c.md](04_runtime_configuration_igris_c.md)
