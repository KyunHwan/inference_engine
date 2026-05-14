# 04 — Runtime configuration for IGRIS_C

**What this covers.** Every IGRIS_C runtime configuration file on this branch — `init_params.py`, `inference_runtime_params.json`, `inference_runtime_params.py` (the `RuntimeParams` class), `inference_runtime_topics.json`, and the two Cyclone DDS XML files. Every field is explained, with type, units, and consumers.
**Who this is for.** Anyone tweaking inference rate, dimensions, network bindings, or kp/kd gains. Read this before editing any of these files.

## Table of contents

- [The four-file layout](#the-four-file-layout)
- [`init_params.py`](#init_paramspy)
- [`inference_runtime_params.json`](#inference_runtime_paramsjson)
- [`inference_runtime_params.py` — the `RuntimeParams` class](#inference_runtime_paramspy--the-runtimeparams-class)
- [`inference_runtime_topics.json`](#inference_runtime_topicsjson)
- [`dds/dds_state.xml`](#ddsdds_statexml)
- [`dds/dds_camera.xml`](#ddsdds_cameraxml)
- [Cross-reference: how the fields propagate](#cross-reference-how-the-fields-propagate)
- [Safe-modification checklist](#safe-modification-checklist)
- [Placeholders you must change for your environment](#placeholders-you-must-change-for-your-environment)

## The four-file layout

```
env_actor/runtime_settings_configs/robots/igris_c/
├── init_params.py                # joint indices, home pose, default kp/kd, INIT_ACTION_17
├── inference_runtime_params.json # cadence, dims, image resize, DDS block, joint_gains
├── inference_runtime_params.py   # RuntimeParams class — reads the JSON above
├── inference_runtime_topics.json # stub for IGRIS_C (topics actually live in the JSON's dds.topics)
└── dds/
    ├── dds_state.xml             # NIC pinning for state/cmd domain
    └── dds_camera.xml            # NIC pinning for camera domain
```

The two "config" files that actually carry runtime data are `init_params.py` (Python constants the bridge imports directly) and `inference_runtime_params.json` (JSON read at startup, parsed into `RuntimeParams`).

## `init_params.py`

[`env_actor/runtime_settings_configs/robots/igris_c/init_params.py`](../../env_actor/runtime_settings_configs/robots/igris_c/init_params.py)

This file is **Python source**, not JSON. It is `import`-ed at module load time by the bridge, the data managers, and the RuntimeParams class. Changing values here requires re-running Python — the JSON is the right place for things you want to tune without an import.

### Joint topology

```python
N_JOINTS = 31

HOME_POSITION = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.13, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
]
HOME_POSE_RAD = np.asarray(HOME_POSITION, dtype=np.float32)
```

The 0-based joint name map (from the module docstring):

| Index | Joint |
|---:|---|
| 0   | Waist_Yaw |
| 1   | Waist_Roll |
| 2   | Waist_Pitch |
| 3–8 | Left leg: Hip_Pitch, Hip_Roll, Hip_Yaw, Knee_Pitch, Ankle_Pitch, Ankle_Roll |
| 9–14 | Right leg: same six joints |
| 15–21 | Left arm: Shoulder_Pitch … Wrist_Pitch |
| 22–28 | Right arm: same seven joints |
| 29 | Neck_Yaw |
| 30 | Neck_Pitch |

The only two non-zero home values are indices 16 (`+0.13` rad ≈ `+7.4°` Shoulder_Roll_L) and 23 (`-0.13` rad Shoulder_Roll_R) — these keep the arms slightly out at start to clear the body.

### Action vs. fixed joints

```python
LEFT_ARM_IDS  = [15, 16, 17, 18, 19, 20, 21]
RIGHT_ARM_IDS = [22, 23, 24, 25, 26, 27, 28]
WAIST_YAW_ID  = 0
ACTIVE_JOINT_IDS = LEFT_ARM_IDS + RIGHT_ARM_IDS + [WAIST_YAW_ID]   # 15 indices
FIXED_JOINT_IDS  = sorted(set(range(N_JOINTS)) - set(ACTIVE_JOINT_IDS))  # 16 indices
```

The policy controls 15 body joints (both arms + waist yaw). The other 16 (legs + waist roll/pitch + neck) are held at `HOME_POSE_RAD`. The bridge enforces this every step by starting from `HOME_POSE_RAD.copy()` before slotting in the active values — see [`controller_bridge.py:391-394`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L391).

### Hand topology

```python
HAND_LEFT_IDS  = [11, 12, 13, 14, 15, 16]
HAND_RIGHT_IDS = [21, 22, 23, 24, 25, 26]
HAND_MOTOR_IDS = HAND_LEFT_IDS + HAND_RIGHT_IDS  # 12 IDs
```

These **look like** body joint IDs (11–16 overlap with leg/arm joints, 21–26 overlap with right arm) but they live on a separate DDS domain entry (`rt/handstate` / `rt/handcmd`), so the wire never confuses them. The `HandCmd.motor_cmd` is a `sequence` (variable length) so the bridge can populate only the 12 it cares about.

### Default gains

```python
DEFAULT_JOINT_KP = np.asarray([
    200.0, 200.0, 200.0,                                 # waist
    500.0, 200.0,  50.0, 500.0, 300.0, 300.0,            # left leg
    500.0, 200.0,  50.0, 500.0, 300.0, 300.0,            # right leg
     75.0, 200.0,  45.0,  45.0,   5.0,   5.0,   5.0,     # left arm
     75.0, 200.0,  45.0,  45.0,   5.0,   5.0,   5.0,     # right arm
      2.0,   5.0,                                        # neck
], dtype=np.float32)
DEFAULT_JOINT_KD = np.asarray([
    15.0, 15.0, 15.0,
     3.0,  0.5,  0.5,  3.0,  1.5,  1.5,
     3.0,  0.5,  0.5,  3.0,  1.5,  1.5,
     0.75, 2.0, 0.225, 0.225, 0.1, 0.1, 0.1,
     0.75, 2.0, 0.225, 0.225, 0.1, 0.1, 0.1,
     0.05, 0.1,
], dtype=np.float32)
DEFAULT_HAND_KP = 50.0
DEFAULT_HAND_KD = 2.0
```

These are the IGRIS_C "masterarm freemode" defaults from `igris_c_fixer_record`. Override them in `inference_runtime_params.json`'s `dds.joint_gains` block — see [the JSON description below](#inference_runtime_paramsjson).

### Action initial vector

```python
INIT_HAND_LEFT  = 0.0
INIT_HAND_RIGHT = 0.0
INIT_WAIST_YAW  = float(HOME_POSE_RAD[WAIST_YAW_ID])  # == 0.0

INIT_ACTION_17 = np.concatenate([
    HOME_POSE_RAD[LEFT_ARM_IDS],   # 7
    HOME_POSE_RAD[RIGHT_ARM_IDS],  # 7
    np.array([INIT_HAND_LEFT, INIT_HAND_RIGHT, INIT_WAIST_YAW], dtype=np.float32),  # 3
]).astype(np.float32)
```

This 17-D vector is what the data managers tile across the action chunk at episode start. **The 17-D order must match `publish_action`'s assertion** at [`controller_bridge.py:389-394`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L389):

```
[ left_arm_7 | right_arm_7 | left_hand_1 | right_hand_1 | waist_yaw_1 ]
```

### Proprio dimension constants

```python
PROPRIO_BODY_Q_DIM   = N_JOINTS              # 31
PROPRIO_HAND_Q_DIM   = len(HAND_MOTOR_IDS)   # 12
PROPRIO_BODY_TAU_DIM = N_JOINTS              # 31
PROPRIO_HAND_TAU_DIM = len(HAND_MOTOR_IDS)   # 12
PROPRIO_STATE_DIM    = 86                     # sum
```

Imported by the bridge to size the zero-fill fallback ([`controller_bridge.py:353-360`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L353)). The runtime JSON's `proprio_state_dim` **must equal 86** for tensor shapes to align.

### State key labels

```python
IGRIS_C_STATE_KEYS = ["body_q_31", "hand_q_12", "body_tau_31", "hand_tau_12"]
```

Descriptive labels for the four proprio slices. **Not used at runtime** — they document the layout for future instrumentation.

## `inference_runtime_params.json`

[`env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json`](../../env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json)

Annotated full file (current values on this branch — verify before depending on them):

```json
{
  "HZ": 20,                                  // control rate (Hz). Sets DT = 1/HZ.
  "max_delta_deg": 5,                        // slew-rate cap on the 31-D body delta per step. Converted to rad in RuntimeParams.
  "policy_update_period": 50,                // sequential algorithm: run policy every N control steps. (RTC ignores; it has its own logic.)
  "mono_image_resize": {
    "width": 224,
    "height": 224
  },                                         // bridge.read_state() resizes each camera frame to this WxH (BGR, CHW).
  "camera_names": ["head", "left", "right"], // ordering matters: matches keys produced by read_state() and the per-camera SHM blocks.
  "proprio_state_dim": 86,                   // MUST equal PROPRIO_STATE_DIM in init_params.py.
  "action_dim": 17,                          // MUST equal the policy's action_dim (openpi_batched.yaml:7) and the bridge assert.
  "action_chunk_size": 50,                   // number of future actions the policy emits per forward pass.
  "proprio_history_size": 50,                // depth of the proprio ring buffer (for policies that consume history).
  "num_img_obs": 1,                          // images per camera kept in history.
  "img_obs_every": 1,                        // sample-image-every-N-steps (1 = every step).
  "norm_stats_file_path": "/home/robros/Projects/inference_engine/trainer/experiment_training/igris_c/dataset_stats.pkl",
                                             // ABSOLUTE PATH — see "Placeholders" below.
  "dds": {
    "namespace": "",                         // DDS partition prefix; empty = default.
    "state_domain_id": 0,                    // DDS domain for state/cmd traffic.
    "camera_domain_id": 1,                   // DDS domain for cameras.
    "state_dds_xml":  "/home/robros/Projects/inference_engine/env_actor/runtime_settings_configs/robots/igris_c/dds/dds_state.xml",
    "camera_dds_xml": "/home/robros/Projects/inference_engine/env_actor/runtime_settings_configs/robots/igris_c/dds/dds_camera.xml",
    "topics": {
      "lowstate":  "rt/lowstate",
      "handstate": "rt/handstate",
      "lowcmd":    "rt/lowcmd",
      "handcmd":   "rt/handcmd",
      "head_camera":  "igris_c/sensor/eyes_stereo",
      "left_camera":  "igris_c/sensor/left_hand",
      "right_camera": "igris_c/sensor/right_hand"
    },
    "init_robot_at_startup": false,          // legacy SDK only; ignored by the active cyclonedds-python bridge.
    "joint_gains": {
      "kp":      [200, 200, 200, 500, 200, 50, 500, 300, 300, 500, 200, 50, 500, 300, 300, 75, 200, 45, 45, 5, 5, 5, 75, 200, 45, 45, 5, 5, 5, 2, 5],
      "kd":      [15, 15, 15, 3, 0.5, 0.5, 3, 1.5, 1.5, 3, 0.5, 0.5, 3, 1.5, 1.5, 0.75, 2, 0.225, 0.225, 0.1, 0.1, 0.1, 0.75, 2, 0.225, 0.225, 0.1, 0.1, 0.1, 0.05, 0.1],
      "hand_kp": 50.0,
      "hand_kd": 2.0
    }
  }
}
```

The `joint_gains.kp` / `kd` arrays here are byte-for-byte the same as `DEFAULT_JOINT_KP` / `KD` in `init_params.py`. They are **both** there so you can tune via JSON without editing Python; the JSON wins when present (see the `RuntimeParams` properties below).

### Field reference

| Field | Type | Units | Consumers |
|---|---|---|---|
| `HZ` | int | Hz | `RuntimeParams.HZ` → `DT = 1/HZ` in [`controller_bridge.py:297`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L297) |
| `max_delta_deg` | float | degrees per control step | Converted to rad in [`inference_runtime_params.py:11`](../../env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.py#L11); applied as `np.clip(delta, -max_delta, +max_delta)` at [`controller_bridge.py:396-398`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L396) |
| `policy_update_period` | int | control steps | Sequential only: gate on `(t % period) == 0` in [`sequential_actor.py:121`](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L121) |
| `mono_image_resize.width` / `.height` | int | px | `cv2.resize` target in `read_state` ([line 382](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L382)); SHM camera block shape in [`rtc_actor.py:59`](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L59) |
| `camera_names` | list[str] | — | Bridge reads images for each name; ordering must match SHM keys |
| `proprio_state_dim` | int | dimensionless | SHM proprio block shape in [`rtc_actor.py:56`](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L56); data-manager state buffer size |
| `action_dim` | int | dimensionless | SHM action block shape in [`rtc_actor.py:68`](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L68); policy `action_dim`; bridge assertion at line 389 |
| `action_chunk_size` | int | — | SHM action rows; data manager tile size |
| `proprio_history_size` | int | — | SHM proprio rows; data manager history depth |
| `num_img_obs` | int | — | SHM image rows per camera |
| `img_obs_every` | int | control steps | Data-manager image-sample gate |
| `norm_stats_file_path` | str | absolute path | `RuntimeParams.read_stats_file()` pickle-loaded for normalization |
| `dds.namespace` | str | — | Cyclone DDS partition prefix (legacy SDK only — bridge ignores) |
| `dds.state_domain_id` | int | domain id | `DomainParticipant(state_domain_id)` |
| `dds.camera_domain_id` | int | domain id | `DomainParticipant(camera_domain_id)` |
| `dds.state_dds_xml` | str | absolute path | Read at bridge construction, set as `CYCLONEDDS_URI` env if unset |
| `dds.camera_dds_xml` | str | absolute path | Not read directly by the active bridge — the camera participant inherits `CYCLONEDDS_URI` set from `state_dds_xml`. (`TODO:` document whether this is intentional.) |
| `dds.topics.*` | str | DDS topic name | Used in the bridge `Topic(...)` constructor for every reader/writer |
| `dds.init_robot_at_startup` | bool | — | Legacy SDK only; the active cyclonedds-python bridge ignores it |
| `dds.joint_gains.kp` / `.kd` | list[float] | depends on motor — per-motor (length 31) | Per-motor gains baked into each `MotorCmd` |
| `dds.joint_gains.hand_kp` / `.hand_kd` | float | — | Single value baked into each finger `MotorCmd` |

## `inference_runtime_params.py` — the `RuntimeParams` class

[`env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.py`](../../env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.py)

This is a simple property-wrapper class. `__init__` accepts the parsed JSON dict (it does not load the file itself — that's the caller's job; see e.g. [`rtc_actor.py:50`](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L50)). Each property surfaces one field.

Notable patterns:

| Property | Source | Notes |
|---|---|---|
| `max_delta` | `np.deg2rad(max_delta_deg)` (set in `__init__`) | Returns radians, not degrees |
| `dds_state_xml` / `dds_camera_xml` | `self._dds.get('state_dds_xml', '')` | Returns empty string if missing, never raises |
| `init_robot_at_startup` | `self._dds.get('init_robot_at_startup', False)` | Default False |
| `joint_kp` | `np.asarray(self._dds['joint_gains']['kp'])` if present, else `DEFAULT_JOINT_KP.copy()` | Falls back to Python defaults |
| `joint_kd` | same pattern | |
| `hand_kp` | `float(...)` of the JSON value, or `DEFAULT_HAND_KP` | |
| `hand_kd` | same pattern | |
| `read_stats_file()` | Opens `norm_stats_file_path`, unpickles | Prints `"File not found at: ..."` and returns `None` if missing — **the caller must handle `None`**. See [09 § Missing norm stats](09_troubleshooting_igris_c.md#missing-norm-stats-pkl). |

The class deliberately exposes **only** what the inference engine needs. If you add a field to the JSON, you also need a property here to make it visible to the rest of the code.

## `inference_runtime_topics.json`

[`env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_topics.json`](../../env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_topics.json)

Three keys only:

```json
{
  "robot_id": "igris_c_robot1",
  "HZ": 20,
  "_note": "DDS topics live in inference_runtime_params.json's dds.topics block; igris_c does not use ROS2."
}
```

The bridge constructor accepts this dict ([`controller_bridge.py:105`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L105)) but does not use any of its fields. It exists so the same `--inference_runtime_topics_config` CLI flag can be passed for any robot.

`TODO:` consider deprecating the `--inference_runtime_topics_config` flag for IGRIS_C, or moving more fields into this file.

## `dds/dds_state.xml`

[`env_actor/runtime_settings_configs/robots/igris_c/dds/dds_state.xml`](../../env_actor/runtime_settings_configs/robots/igris_c/dds/dds_state.xml)

Cyclone DDS configuration XML. Read by libcyclonedds via the `CYCLONEDDS_URI` environment variable — the bridge sets that env var to **the file's contents** (not its path) at [`controller_bridge.py:114-116`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L114).

```xml
<CycloneDDS>
    <Domain id="0">
        <General>
            <Interfaces>
                <NetworkInterface name="enp11s0" autodetermine="false"/>
            </Interfaces>
        </General>
    </Domain>
</CycloneDDS>
```

This pins Domain 0 to NIC `enp11s0`. **Current value on this branch — verify against your environment.** The XML's own comment block documents the rule: the NIC must be the one carrying the `192.168.10.0/24` subnet that talks to the NUC. Verify with:

```bash
ip addr show <iface>           # expects 192.168.10.x/24
ping 192.168.10.1              # expects reachable
ip route | grep 224            # expects a multicast route on this iface
```

## `dds/dds_camera.xml`

[`env_actor/runtime_settings_configs/robots/igris_c/dds/dds_camera.xml`](../../env_actor/runtime_settings_configs/robots/igris_c/dds/dds_camera.xml)

```xml
<CycloneDDS>
    <Domain id="1">
        <General>
            <Interfaces>
                <NetworkInterface name="enp11s0" autodetermine="false"/>
            </Interfaces>
        </General>
    </Domain>
</CycloneDDS>
```

Identical pattern, Domain 1. The comment block documents two deployment variants:

- **Direct LAN** (current default): cameras are on the same NIC as the state/cmd domain. Both XMLs pin the same `enp11s0`.
- **NUC-bridged WiFi**: the NUC's route daemon re-publishes the camera topics onto Domain 10 over WiFi. In that case, change `<Domain id="1">` to `<Domain id="10">` **and** set `dds.camera_domain_id: 10` in `inference_runtime_params.json` so the participant joins the same domain.

**Why isn't this XML actually read by the active bridge?** Because `cyclonedds-python` reads `CYCLONEDDS_URI` once at process startup, and the bridge sets it from the **state** XML's contents. As long as both XMLs declare interfaces for both domains, the single env var works. The `dds_camera.xml` file is kept for documentation and for tools that explicitly read it. `TODO:` confirm with the cyclonedds team whether multi-XML support is feasible — for now the camera XML is informational on this branch.

## Cross-reference: how the fields propagate

The dataflow from JSON → live tensors:

```
inference_runtime_params.json
        │
        ▼ (json.load + RuntimeParams(...))
RuntimeParams instance
        │
        ├─→ proprio_state_dim ─→ SHM proprio block shape (RTC) / np ring buffer (Sequential)
        │                    └─→ bridge zero-fill fallback length
        │
        ├─→ action_dim       ─→ SHM action block shape (RTC) / data-manager INIT tile shape
        │                    └─→ policy build_policy(action_dim) ─→ openpi_batched.yaml:action_dim
        │                    └─→ bridge assert at controller_bridge.py:389
        │
        ├─→ camera_names     ─→ bridge read_state keys
        │                    └─→ SHM camera blocks (one per name in {head, left, right})
        │
        ├─→ mono_image_resize.width/height ─→ bridge cv2.resize target
        │                                   └─→ SHM image block (height, width, 3)
        │
        ├─→ HZ               ─→ ControllerInterface.DT = 1/HZ ─→ control loop `next_t += DT`
        │
        ├─→ max_delta_deg    ─→ RuntimeParams.max_delta (rad) ─→ slew limiter
        │
        ├─→ dds.state_dds_xml contents ─→ os.environ["CYCLONEDDS_URI"]
        │
        ├─→ dds.state_domain_id   ─→ DomainParticipant(state)
        ├─→ dds.camera_domain_id  ─→ DomainParticipant(camera)
        ├─→ dds.topics.*          ─→ Topic(name) for every reader/writer
        │
        ├─→ dds.joint_gains.kp/kd ─→ bridge.publish_lowcmd per-motor MotorCmd.kp/kd
        └─→ dds.joint_gains.hand_kp/kd ─→ bridge.publish_handcmd per-finger MotorCmd.kp/kd

init_params.py
        │
        ├─→ HOME_POSE_RAD            ─→ bridge.publish_action 31-D init + hold-fixed
        ├─→ LEFT_ARM_IDS/...         ─→ bridge 17-D ↔ 31-D index mapping
        ├─→ HAND_LEFT_IDS/...        ─→ bridge.publish_handcmd motor IDs
        ├─→ INIT_ACTION_17           ─→ data managers' action-chunk init
        ├─→ PROPRIO_*_DIM constants  ─→ bridge zero-fill fallback
        └─→ DEFAULT_JOINT_KP/KD,
            DEFAULT_HAND_KP/KD       ─→ RuntimeParams fallback when JSON omits joint_gains
```

## Safe-modification checklist

If you change... | You must also update...
---|---
`action_dim` (17 → other) | `init_params.py:INIT_ACTION_17` (rebuild), `controller_bridge.py:389` (`assert a.shape == (17,)`), `controller_bridge.py:391-394` (slice mapping), `openpi_batched.yaml:7` (`action_dim`), and re-train your policy.
`proprio_state_dim` (86 → other) | `init_params.py:PROPRIO_*_DIM` so they sum to the new value, `controller_bridge.py:347-363` (proprio assembly), and re-train your policy with stats of the new dim.
`HZ` | Verify `max_delta_deg` is still safe at the new rate: `max_velocity_deg_per_s ≈ HZ × max_delta_deg`. The default `20 × 5 = 100°/s` is the IGRIS_C safety reference.
`camera_names` (add/remove) | `controller_bridge.py:193-197` (per-camera reader spawn), `rtc_actor.py:58-66` (SHM blocks per camera), the data managers' history dicts, the policy input layout.
`mono_image_resize` | Re-train your policy or your image encoder — the input H/W is baked into the model.
`max_delta_deg` | Just this file (it propagates via `RuntimeParams.max_delta`). Verify nothing in your simulation/safety expects degrees vs. radians elsewhere.
`norm_stats_file_path` | Must be an **absolute path** to a `.pkl` produced by trainer that contains `observation.state` (mean/std ≥ 86) and `action` (mean/std == 17). See [06](06_normalization_igris_c.md).
`dds.state_dds_xml` / `dds.camera_dds_xml` paths | Re-check that the XML's `<NetworkInterface name="...">` matches a real NIC on the host.
`dds.joint_gains.kp` or `.kd` | Be sure all 31 entries are present. Bridge does `assert self._joint_kp.shape == (N_JOINTS,)` at [`controller_bridge.py:219-220`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L219).
`HAND_*_IDS` in `init_params.py` | Re-test on hardware — these are mapped to physical finger motors on the NUC.

## Placeholders you must change for your environment

| Value | Where | Why it's a placeholder | What to set it to |
|---|---|---|---|
| `/home/robros/Projects/inference_engine/trainer/experiment_training/igris_c/dataset_stats.pkl` | `inference_runtime_params.json:16` | Absolute path on the author's machine | Absolute path to **your** copy of `dataset_stats.pkl`. See [06](06_normalization_igris_c.md). |
| `/home/robros/Projects/inference_engine/.../dds_state.xml` | `inference_runtime_params.json:21` | Absolute path | Absolute path of `dds_state.xml` on your filesystem (typically `$(pwd)/env_actor/runtime_settings_configs/robots/igris_c/dds/dds_state.xml`). |
| `/home/robros/Projects/inference_engine/.../dds_camera.xml` | `inference_runtime_params.json:22` | Absolute path | Same — absolute path of `dds_camera.xml`. |
| `enp11s0` | `dds/dds_state.xml:18` and `dds/dds_camera.xml:18` | NIC name on author's host | The NIC that holds the `192.168.10.x/24` address on **your** host. Find with `ip addr show`. |
| `100.109.184.39`, `100.112.232.50` | `start_ray.sh:25,40` | Tailscale 100.x IPs on author's machines | Your head and worker node IPs. |
| `robros-ai1`, `robros-5090` | `start_ray.sh:29,37` | Author's hostnames | Your hostnames. |
| `/home/robros/Projects/robros_vla_inference_engine/openpi_film/checkpoints/...` | `openpi_batched.yaml:5` | Author's checkpoint path | Your IGRIS_C-trained checkpoint path. |
| `/home/robros/Projects/inference_engine/...` in `run_bridge_monitor.py:16` | `CFG` constant | Absolute path | Your repo's runtime_settings_configs path. |

---

← Back to index: [README.md](README.md) · Next → [05_data_manager_bridges_igris_c.md](05_data_manager_bridges_igris_c.md)
