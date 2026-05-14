# 02 — Changes vs. `main`

**What this covers.** Every file added, modified, or deleted on `igris_c/c1` versus the tip of `main` (`git diff main..igris_c/c1`). For each, the reason it changed, what it now does, how it relates to the IGRIS_B equivalent, and any non-obvious side effects.
**Who this is for.** Reviewers, new contributors trying to grok the branch, and anyone debugging "why does X behave differently than on `main`?"

This document uses the two-dot diff `main..igris_c/c1` (changes between branch tips), not the three-dot diff `main...igris_c/c1` (which also includes files both branches gained independently after the merge base, such as the new `docs/` walkthroughs). The relevant code differences between the branch tips are the 39 entries below.

## Table of contents

- [Summary](#summary)
- [Files added](#files-added)
- [Files modified](#files-modified)
- [Submodule pointer](#submodule-pointer)
- [What did NOT change](#what-did-not-change)

## Summary

```
git diff --stat main..igris_c/c1 →
 39 files changed, 3762 insertions(+), 182 deletions(-)
```

Categorized:

| Category | Added | Modified | Deleted |
|---|---|---|---|
| IGRIS_C controller bridge | 3 | 1 | 0 |
| IGRIS_C runtime configs | 5 | 1 | 0 |
| IGRIS_C data managers | 2 | 1 | 0 |
| IGRIS_C normalization | 2 | 0 | 0 |
| New `rtc_local/` subtree | 13 | 0 | 0 |
| New `sequential_local/` subtree | 2 | 0 | 0 |
| Cross-cutting | 1 | 1 | 0 |
| Top-level | 4 | 0 | 0 |
| IGRIS_B parity | 0 | 1 | 0 |
| Policy YAML | 0 | 1 | 0 |

Commit history (`git log --oneline main..igris_c/c1`):

```
dd4e8a9 updated trainer
8f5d1bf Sync .md files from main
5409e72 modified codebase for igris_c
e26ea14 modified controller bridge to use joint space instead of motor space for igris-c
6e58c88 added igris-c interface
```

## Files added

### `env_actor/robot_io_interface/robots/igris_c/controller_bridge_sdk_legacy.py` (Added)

**Why it changed.** The original IGRIS_C controller bridge was written against the `igris_c_sdk` Python wrapper. That SDK was built against a fork of `igris_c_msgs.idl` whose DDS type hashes do not match the NUC firmware; SEDP discovery matches topics but `type_consistency_enforcement=DISALLOW_TYPE_COERCION` rejects the writer. The decision (recorded in [`controller_bridge.py:3-10`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L3)) was to replace the SDK path with direct `cyclonedds-python`, and keep this file as historical reference / fallback.

**What it now does.** Self-contained legacy bridge that uses two `igris_c_sdk.ChannelFactory` instances (one for state/cmd domain, one for cameras), per the SDK README's "Multiple Domains In One Process" pattern. Same action layout (17-D), same PJS kinematic mode, same broadcast-hand convention as the active bridge. Not wired into the factory dispatch — nothing imports this file on `igris_c/c1`. It is kept as a reference for anyone who later needs to revive the SDK route.

**IGRIS_B counterpart.** None. IGRIS_B uses `rclpy`, not `igris_c_sdk`.

**Non-obvious side effects.** None — the file is dormant.

### `env_actor/robot_io_interface/robots/igris_c/messages/__init__.py` (Added)

Empty package marker for the new `messages` subdirectory.

### `env_actor/robot_io_interface/robots/igris_c/messages/igris_c_msgs.py` (Added)

**Why it changed.** The active controller bridge calls Cyclone DDS directly, so it needs the IDL message types declared in Python. The SDK-shipped types were rejected by the NUC; the replacement here is the **schema record-BE uses**, identical to the NUC firmware.

**What it now does.** Declares dataclass-decorated `IdlStruct`s and `IdlEnum`s that match the NUC's `igris_c_msgs.idl`:
- Enums: `KinematicMode` (MS / PJS), `RelayState`, `EStopState`, `BmsConnState`, `BmsInitState`, `BmsInitType`, `TorqueType`, `ControlMode`.
- Per-motor messages: `MotorCmd`, `MotorState`, `JointState`, `IMUState`.
- Low-level: `LowCmd` (kinematic_mode + 31 `MotorCmd`s), `LowState` (timestamp + tick + IMU + 31 `MotorState`s + 31 `JointState`s).
- Hand: `HandCmd` (sequence of `MotorCmd`s), `HandState` (sequence of `MotorState`s + IMU).
- BMS / service: `BmsState`, `BmsInitCmd`, `TorqueCmd`, `ControlModeCmd`, `ControlModeState`, `ServiceResponse`.
- Cameras: `Header`, `CompressedMessage` (with `CameraFrame = CompressedMessage` back-compat alias).
- Topic name constants under `TopicNames` (`LOW_STATE = "rt/lowstate"`, `CAMERA_EYES_STEREO = "igris_c/sensor/eyes_stereo"`, etc.).

The `typename=` annotations on every struct (`"igris_c.msg.dds.LowCmd"`, etc.) **must** match the C++ side exactly — see the module docstring at [`igris_c_msgs.py:6-9`](../../env_actor/robot_io_interface/robots/igris_c/messages/igris_c_msgs.py#L6). Changing them silently breaks DDS discovery.

**IGRIS_B counterpart.** None. IGRIS_B uses ROS2 standard messages (`sensor_msgs/JointState`, `std_msgs/Float32MultiArray`) which are imported directly in [`igris_b/controller_bridge.py:16-17`](../../env_actor/robot_io_interface/robots/igris_b/controller_bridge.py#L16).

**Non-obvious side effects.** Any addition or removal of fields requires regenerating the IDL hashes on the NUC side too. Do not touch field types or ordering without coordinating with the firmware team. See the docstring at [`messages/igris_c_msgs.py:6-13`](../../env_actor/robot_io_interface/robots/igris_c/messages/igris_c_msgs.py#L6).

### `env_actor/robot_io_interface/robots/igris_c/run_bridge_monitor.py` (Added)

**Why it changed.** Engineers need a way to verify the bridge talks to the NUC before involving the policy stack.

**What it now does.** Imports `ControllerBridge`, calls `start_state_readers()`, then in a loop reads `controller_bridge.read_state()` and prints a single-line summary every 100ms: first 3 body joints, first 3 hand joints, first body tau value, and the shape of each camera frame. See [03 § Bridge-only test](03_robot_io_interface_igris_c.md#bridge-only-test).

**IGRIS_B counterpart.** None on the current branch.

**Non-obvious side effects.** The `CFG` constant on [line 16](../../env_actor/robot_io_interface/robots/igris_c/run_bridge_monitor.py#L16) is an absolute path on the author's machine. Run from the project root or change the path. `TODO:` consider making this a CLI flag.

### `env_actor/runtime_settings_configs/robots/igris_c/dds/dds_state.xml` (Added)

**Why it changed.** `cyclonedds-python` does not have a per-participant config arg the way the C++ SDK does — `DomainParticipant` reads `CYCLONEDDS_URI` from the env. The bridge sets `os.environ["CYCLONEDDS_URI"]` to the *contents* of this file before constructing the participant ([`controller_bridge.py:114-116`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L114)).

**What it now does.** Pins DDS Domain 0 to a specific NIC. Current value: `<NetworkInterface name="enp11s0" autodetermine="false"/>`. The XML's own comment block documents that the NIC must be the one carrying the `192.168.10.0/24` subnet (the NUC's LAN). **Current value on this branch — verify against your environment** before deploying.

**IGRIS_B counterpart.** None.

### `env_actor/runtime_settings_configs/robots/igris_c/dds/dds_camera.xml` (Added)

**Why it changed.** Same reason — Cyclone DDS needs a config to pin the camera participant (Domain 1).

**What it now does.** Pins DDS Domain 1 to NIC `enp11s0`. The XML comment notes that some operator setups bridge cameras onto WiFi via the NUC route daemon and run on Domain 10 instead — in that case change `<Domain id="1">` here to `<Domain id="10">` **and** set `dds.camera_domain_id: 10` in `inference_runtime_params.json` so the bridge participant joins the same domain. **Current value on this branch — verify against your environment.**

**IGRIS_B counterpart.** None.

### `env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json` (Added)

**Why it changed.** The stub README on `main` flagged this as missing. It is the canonical runtime-config object the inference engine constructs `RuntimeParams` from.

**What it now does.** JSON object with the inference cadence, the proprio/action dimensions, the camera resize, the normalization-stats path, and the DDS block (domain IDs, XML paths, topic names, per-motor kp/kd defaults). Full field reference in [04_runtime_configuration_igris_c.md](04_runtime_configuration_igris_c.md). Two values are **absolute paths on the author's machine** and need to change for your environment: `norm_stats_file_path` (line 16) and the two `dds.*_dds_xml` entries (lines 21–22).

**IGRIS_B counterpart.** [`runtime_settings_configs/robots/igris_b/inference_runtime_params.json`](../../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json) — different shape (no `dds` block; uses `proprio_state_dim: 24`, `action_dim: 24`, image resize 320×240).

**Non-obvious side effects.** The `proprio_state_dim` (86) is consumed in three downstream places that must stay aligned: (a) the SharedMemory allocation in [`rtc/rtc_actor.py:55-57`](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L55), (b) the bridge's proprio assembly in [`controller_bridge.py:363`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L363), and (c) the normalizer's slice of `state_mean[:proprio_len]` in [`data_normalization_manager.py:24-25`](../../env_actor/nom_stats_manager/robots/igris_c/data_normalization_manager.py#L24).

### `env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.py` (Added)

**Why it changed.** Each robot has its own `RuntimeParams` class. The IGRIS_B one didn't model the DDS block. A new class is required to expose the DDS-specific fields.

**What it now does.** Defines `RuntimeParams` with the standard properties (HZ, max_delta, dims, history) plus DDS-specific ones: `dds_namespace`, `dds_state_domain_id`, `dds_camera_domain_id`, `dds_state_xml`, `dds_camera_xml`, `dds_topics`, `init_robot_at_startup`, `joint_kp`, `joint_kd`, `hand_kp`, `hand_kd`. The kp/kd properties fall back to `DEFAULT_*` from `init_params.py` when the JSON omits them. Also defines `read_stats_file()` (identical pattern to IGRIS_B).

**IGRIS_B counterpart.** [`igris_b/inference_runtime_params.py`](../../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.py) — no DDS fields, no kp/kd, no `init_robot_at_startup`.

**Non-obvious side effects.** This class is selected by an `elif robot == "igris_c"` block in five places (see [07_factory_registration_igris_c.md](07_factory_registration_igris_c.md)). Any property you add here is only available to code that imports the `igris_c` variant.

### `env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_topics.json` (Added)

**Why it changed.** The IGRIS_B version of this file maps ROS2 topic names + field slices to observation keys. IGRIS_C does not use ROS2; the equivalent mapping lives inside `inference_runtime_params.json`'s `dds.topics` block. A separate file was still added so the same `--inference_runtime_topics_config` CLI flag has something to load.

**What it now does.** Holds `robot_id`, `HZ`, and a `_note` that points the reader to the `dds.topics` block. That's the entire file:

```json
{
  "robot_id": "igris_c_robot1",
  "HZ": 20,
  "_note": "DDS topics live in inference_runtime_params.json's dds.topics block; igris_c does not use ROS2."
}
```

**IGRIS_B counterpart.** [`igris_b/inference_runtime_topics.json`](../../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_topics.json) — large, with full topic-and-slice mapping.

**Non-obvious side effects.** The bridge constructor takes `inference_runtime_topics_config` but ignores it (see [`controller_bridge.py:105`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L105) — the parameter is accepted then unused; topics come from `runtime_params.dds_topics`). The argument exists only because the `ControllerInterface` factory passes it for all robots.

### `env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_c/__init__.py` (Added)

Empty package marker.

### `env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_c/shm_manager_bridge.py` (Added)

**Why it changed.** Implements the IGRIS_C variant of the RTC shared-memory manager. The IGRIS_B version is algorithmically identical; only the **init-action vector** differs.

**What it now does.** Manages the shared-memory blocks (`proprio`, `head`, `left`, `right`, `action`) and the synchronization primitives (RLock, two Conditions, two Events, two Values) used by the RTC two-process loop. Provides:
- `wait_for_min_actions(min_actions)` and `wait_for_inference_ready()` — Condition-variable waits used by the inference and control workers respectively.
- `atomic_read_for_inference()` — snapshot under lock, returns observation + `prev_action` (the tail of the previous chunk not yet executed) + `est_delay`.
- `atomic_write_obs_and_increment_get_action()` — control side: shift proprio history by 1, copy new obs, return the next action.
- `write_action_chunk_n_update_iter_val()` — inference side: write a fresh chunk, decrement `num_control_iters` by however many actions were executed during inference.
- `init_action_chunk()` / `bootstrap_obs_history()` / `init_action_chunk_obs_history()` — episode bootstrap.

The IGRIS_C-specific bit is on lines [222-224](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_c/shm_manager_bridge.py#L222) and [265-266](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_c/shm_manager_bridge.py#L265): the chunk is initialized by tiling `INIT_ACTION_17` (17-D, from `init_params.py`) instead of IGRIS_B's `INIT_JOINT_LIST + INIT_HAND_LIST` concat.

**IGRIS_B counterpart.** [`rtc/data_manager/robots/igris_b/shm_manager_bridge.py`](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py) — same algorithm; uses a 24-D init from IGRIS_B-specific constants.

**Non-obvious side effects.** The bridge is registered in the factory [`rtc/data_manager/shm_manager_interface.py:30-31`](../../env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py#L30). The initial delay-queue value (`self._delay_queue.add(15)`) at line 82 sets the RTC guided-inference delay estimate floor; this matches the new IGRIS_B value (was 5, now 15 — see "Files modified" below).

### `env_actor/auto/inference_algorithms/rtc_local/__init__.py` and the 12 other files under `rtc_local/` (Added — 13 files)

Files in this group:
- `rtc_local/__init__.py`
- `rtc_local/rtc_local_actor.py`
- `rtc_local/actors/__init__.py`
- `rtc_local/actors/control_loop.py`
- `rtc_local/actors/inference_loop.py`
- `rtc_local/data_manager/__init__.py`
- `rtc_local/data_manager/shm_manager_interface.py`
- `rtc_local/data_manager/robots/__init__.py`
- `rtc_local/data_manager/robots/igris_b/__init__.py`
- `rtc_local/data_manager/robots/igris_b/shm_manager_bridge.py`
- `rtc_local/data_manager/robots/igris_c/__init__.py`
- `rtc_local/data_manager/robots/igris_c/shm_manager_bridge.py`
- `rtc_local/data_manager/utils/__init__.py`
- `rtc_local/data_manager/utils/max_deque.py`
- `rtc_local/data_manager/utils/shared_memory_utils.py`

**Why they changed.** Adds a Ray-free path for the RTC algorithm. Same two-process (control + inference) topology, same shared-memory layout, same actors — but the parent process is just `python run_inference_local.py`, not a `@ray.remote` actor on a worker node. Useful for one-machine development and for environments where Ray is overkill (or unavailable).

**What they now do.** Mirror the contents of `rtc/` line-for-line, except:
- The parent class `RTCLocalActor` has no `@ray.remote` decorator (compare [`rtc_local/rtc_local_actor.py:10`](../../env_actor/auto/inference_algorithms/rtc_local/rtc_local_actor.py#L10) with [`rtc/rtc_actor.py:10`](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L10)).
- `actors/control_loop.py` and `actors/inference_loop.py` drop the `import ray; ray.init(address="auto", ...)` block (compare [`rtc_local/actors/control_loop.py:14-22`](../../env_actor/auto/inference_algorithms/rtc_local/actors/control_loop.py#L14) — no ray import — with [`rtc/actors/control_loop.py:20-23`](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L20)).
- Everything else (SharedMemory shapes, lock/condition/event use, the per-step `atomic_*` operations) is byte-identical.

**IGRIS_B / IGRIS_C counterparts.** The IGRIS_B and IGRIS_C variants of `rtc_local/data_manager/robots/{igris_b,igris_c}/shm_manager_bridge.py` mirror the corresponding files under `rtc/data_manager/robots/...`.

**Non-obvious side effects.**
- Maintenance burden: any bug fixed in `rtc/` must be fixed in `rtc_local/` too (and vice versa). There is no shared base — they are parallel copies. `TODO:` future cleanup to share via inheritance or a common module.
- Note: `rtc_local/actors/control_loop.py` collapses the outer `while True` of the Ray variant into a single-episode flow (see [`rtc_local/actors/control_loop.py:112-142`](../../env_actor/auto/inference_algorithms/rtc_local/actors/control_loop.py#L112) — the inner loop is `while True` over `t` increments). The Ray variant's outer `while True` loops over episodes (see [`rtc/actors/control_loop.py:79-118`](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L79)). Multi-episode local runs are not supported on this branch; one episode then process exit. Flag this when you copy a debug fix from `rtc/` to `rtc_local/`.

### `env_actor/auto/inference_algorithms/sequential_local/__init__.py` and `sequential_local_actor.py` (Added — 2 files)

**Why they changed.** Same motivation as `rtc_local/`: Ray-free path for the sequential algorithm.

**What they now do.** [`sequential_local/sequential_local_actor.py`](../../env_actor/auto/inference_algorithms/sequential_local/sequential_local_actor.py) is a copy of `sequential/sequential_actor.py` with the `@ray.remote` decorator and the `import ray` block dropped. **However**, the `start()` method has been **partially commented out** (the inference + publish path on lines 134–176 is in a docstring-style block string `"""..."""`). What actually executes is:

1. Bridge startup + warmup + robot-position init + first observation.
2. Bootstrap observation history.
3. Read state once.
4. Write the three camera frames to `head.png`, `left.png`, `right.png` in the working directory.
5. Return.

This makes it a camera-stream sanity check, **not** a full sequential inference run. `TODO:` un-comment lines 134–176 (and the `DataNormalizationInterface` constructor on line 85) when you need full sequential inference locally. See [01 § Running inference](01_quickstart_igris_c.md#running-inference) for the caveat.

**Non-obvious side effects.** Anyone reading "sequential local works" without checking this code will be surprised that the robot doesn't move. The three `.png` files added at the repo root (`head.png`, `left.png`, `right.png`) are example outputs from this script.

### `env_actor/nom_stats_manager/robots/igris_c/__init__.py` (Added)

Empty package marker.

### `env_actor/nom_stats_manager/robots/igris_c/data_normalization_manager.py` (Added)

**Why it changed.** Each robot needs its own normalization bridge. The IGRIS_C state is 86-D, the action is 17-D — and the layout (`body_q(31) + hand_q(12) + body_tau(31) + hand_tau(12)`) is unique to IGRIS_C.

**What it now does.** Provides `DataNormalizationBridge(norm_stats)` with three methods:
- `normalize_state(state)` — z-score on `state['proprio']` using `norm_stats['observation.state']['mean'][:proprio_len]` and `[:proprio_len]` of std (slicing handles the case where the stats vector is longer than the live proprio); divides each non-`proprio` image array by `255.0`.
- `normalize_action(action)` — z-score using `norm_stats['action']['mean']` and `std`.
- `denormalize_action(action)` — inverse: `action * std + mean`. Called inside the policy on the chunk before it lands in shared memory.

Full math + code citations in [06_normalization_igris_c.md](06_normalization_igris_c.md).

**IGRIS_B counterpart.** [`nom_stats_manager/robots/igris_b/data_normalization_manager.py`](../../env_actor/nom_stats_manager/robots/igris_b/data_normalization_manager.py).

**Non-obvious side effects.** The `:proprio_len` slice assumes the pkl was written with `observation.state` in the **exact concatenation order** the bridge produces (`body_q, hand_q, body_tau, hand_tau`). If the trainer wrote it as four separate keys, the bridge needs an extra concat step — the docstring at lines 5-12 documents the contract.

### `env_actor/robot_io_interface/robots/igris_c/__init__.py` (Added)

Single-line package marker: `"""IGRIS_C robot I/O interface (implementation pending)."""` — the comment is stale on this branch but the file is one line.

### `head.png`, `left.png`, `right.png` (Added at repo root)

**Why they changed.** Output of `sequential_local_actor.start()` during a smoke test (see above).

**What they now do.** Nothing — they're committed JPEGs/PNGs of head, left, right camera frames the author captured.

**IGRIS_B counterpart.** None.

**Non-obvious side effects.** They occupy ~250 KB at the repo root. `TODO:` move into a `examples/` or `.gitignore` them.

### `run_inference_local.py` (Added)

**Why it changed.** Entry point counterpart to `run_inference.py`, without Ray.

**What it now does.** Same CLI as `run_inference.py` (same flags, same defaults). Instead of `ray.init` + creating a `@ray.remote` actor, it instantiates `RTCLocalActor` or `SequentialLocalActor` directly and calls `.start()`. The `torch.multiprocessing.set_start_method("spawn")` call at line 71 still happens (cuda-fork safety).

**Non-obvious side effects.** The `--inference_runtime_params_config` and `--inference_runtime_topics_config` **defaults** in this file point at the **IGRIS_C** config paths (lines 84 and 89), unlike `run_inference.py` which defaults to IGRIS_B paths. So `python run_inference_local.py --robot igris_c` works without --inference_runtime_params_config, but `python run_inference_local.py --robot igris_b` will need the IGRIS_B paths passed explicitly. Inverse to `run_inference.py`. **`TODO`:** consider making the defaults match the `--robot` flag dynamically.

## Files modified

### `env_actor/runtime_settings_configs/robots/igris_c/init_params.py` (Modified)

**Why it changed.** The file on `main` is a stub with empty `INIT_JOINT_LIST`, `INIT_HAND_LIST`, `IGRIS_C_STATE_KEYS`, and a TODO-laden hardware checklist. On `igris_c/c1` it is the real spec.

**What it now does.** Defines:
- `N_JOINTS = 31` (3 waist + 12 legs + 14 arms + 2 neck).
- `HOME_POSITION` / `HOME_POSE_RAD` — the 31-element home pose. Two non-zero entries: indices 16 and 23 (`+0.13` and `-0.13` rad on Shoulder_Roll_L/R).
- Joint index groups: `LEFT_ARM_IDS=[15..21]`, `RIGHT_ARM_IDS=[22..28]`, `WAIST_YAW_ID=0`, `ACTIVE_JOINT_IDS = LEFT_ARM_IDS + RIGHT_ARM_IDS + [WAIST_YAW_ID]`, `FIXED_JOINT_IDS = everything else`.
- Hand motor IDs: `HAND_LEFT_IDS=[11..16]`, `HAND_RIGHT_IDS=[21..26]` (12 total). These are **hand-domain** indices, not body-joint indices — they overlap numerically with body joints but live on a different DDS topic.
- `DEFAULT_JOINT_KP/KD` and `DEFAULT_HAND_KP/KD` defaults (overridable from `inference_runtime_params.json`'s `dds.joint_gains` block).
- `INIT_ACTION_17` — the 17-D init action: `HOME_POSE_RAD[LEFT_ARM_IDS]` (7) + `HOME_POSE_RAD[RIGHT_ARM_IDS]` (7) + `[INIT_HAND_LEFT, INIT_HAND_RIGHT, INIT_WAIST_YAW]` (3) = 17.
- `INIT_JOINT_31 = HOME_POSE_RAD.copy()` — for the full-body command that holds fixed joints at home.
- `PROPRIO_*_DIM` constants that add up to `PROPRIO_STATE_DIM = 31+12+31+12 = 86`.
- `IGRIS_C_STATE_KEYS = ["body_q_31", "hand_q_12", "body_tau_31", "hand_tau_12"]` — descriptive labels (not used by the bridge but available for instrumentation).

**IGRIS_B counterpart.** [`igris_b/init_params.py`](../../env_actor/runtime_settings_configs/robots/igris_b/init_params.py) — 24-D action space, no kp/kd defaults, no joint index groups.

**Non-obvious side effects.** Changing `N_JOINTS` would cascade to: the DDS schema (fixed 31-element arrays in `LowCmd.motors` and `LowState.motor_state` in [`igris_c_msgs.py:151-167`](../../env_actor/robot_io_interface/robots/igris_c/messages/igris_c_msgs.py#L151)), `PROPRIO_STATE_DIM`, the bridge's `_publish_lowcmd` loop, and the per-motor kp/kd arrays. Do not change unless the firmware changes too. Also note that `HAND_LEFT_IDS = [11..16]` indices overlap with body-joint IDs 11–16 — these are addressed via a separate DDS topic (`rt/handstate` / `rt/handcmd`) and never confused at the wire level.

### `env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_c/data_manager_bridge.py` (Modified)

**Why it changed.** On `main` this file was a `NotImplementedError`-only stub. On `igris_c/c1` it is the real bridge.

**What it now does.** Implements `DataManagerBridge` with the standard surface:
- `update_state_history(obs_data)` — shifts the proprio ring buffer by 1 and pushes the latest; conditionally shifts the per-camera image ring buffer.
- `buffer_action_chunk(policy_output, current_step)` — stores the latest chunk and the step it came from.
- `get_current_action(current_step)` — indexes into `last_action_chunk` by `current_step - last_policy_step`, clipped to `[0, chunk_size-1]`.
- `init_inference_obs_state_buffer(init_data)` — repeats the initial proprio across the history dimension; repeats the initial image across `num_img_obs`.
- `serve_raw_obs_state()` — returns `{'proprio': <copy>, 'head': ..., 'left': ..., 'right': ...}`.
- `serve_init_action()` — tiles `INIT_ACTION_17` across `action_chunk_size` rows.

The algorithm is **identical** to IGRIS_B's bridge; only the init-action shape differs.

**IGRIS_B counterpart.** [`sequential/data_manager/robots/igris_b/data_manager_bridge.py`](../../env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_b/data_manager_bridge.py).

**Non-obvious side effects.** `serve_init_action` is called by RTC's guided-inference path indirectly (the SHM init initially fills the action buffer with `INIT_ACTION_17` — see [`rtc/data_manager/robots/igris_c/shm_manager_bridge.py:264-266`](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_c/shm_manager_bridge.py#L264)). The 17-D layout must agree with `controller_bridge.publish_action`'s 17-D assertion at [`controller_bridge.py:389`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L389).

### `env_actor/robot_io_interface/robots/igris_c/controller_bridge.py` (Modified)

**Why it changed.** This is the heart of the IGRIS_C integration. Earlier branches used a stub here; an interim `controller_bridge_sdk_legacy.py` was written against `igris_c_sdk` but the SDK's IDL hashes didn't match the firmware. The branch's middle commit (`e26ea14 modified controller bridge to use joint space instead of motor space for igris-c`) re-grounded the bridge on Parallel Joint Space (PJS) instead of Motor Space (MS), matching the recorder convention. The current code uses `cyclonedds-python` directly.

**What it now does.** Full bridge:
- Two `DomainParticipant`s (state @ domain 0, camera @ domain 1) — different domains so cameras and state can use different network paths.
- Five `DataReader`s — `LowState`, `HandState`, and three `CompressedMessage` camera readers.
- Per-reader poll threads (`cyclonedds-python` lacks an `on_data_available` callback, so the bridge takes `N=10` samples in a loop).
- Two `DataWriter`s for `LowCmd` and `HandCmd`.
- Latest-value thread-safe caches (`_Latest`) for all readers — same pattern as record-BE's `DDSParticipantContext`.
- `read_state()` returns `{'proprio': np.ndarray(86,), 'head': (3,H,W), 'left': (3,H,W), 'right': (3,H,W)}`. Proprio is `body_q ⊕ hand_q ⊕ body_tau ⊕ hand_tau`. Cameras are JPEG-decoded; head is rotated 180° and cropped to its left half (the stereo IDL frame holds two views side-by-side).
- `publish_action(action_17, prev_joint_31)`:
  1. Asserts `action_17.shape == (17,)`.
  2. Starts from `HOME_POSE_RAD.copy()`; writes `LEFT_ARM_IDS`, `RIGHT_ARM_IDS`, `WAIST_YAW_ID` from the action; leaves all fixed joints at home.
  3. Applies a slew-rate limit (`max_delta = np.deg2rad(max_delta_deg)`) on the **31-D** delta from `prev_joint`.
  4. Writes a `LowCmd(kinematic_mode=PJS, motors=[MotorCmd(id=i, q=..., kp=..., kd=...) for i in 0..30])`.
  5. Broadcasts the single left-hand action value (`action[14]`) to all 6 left finger motors and the right value (`action[15]`) to all 6 right finger motors via a `HandCmd`.
  6. Returns `(smoothed31, hand12)` so the caller can use `smoothed31` as `prev_joint` next step.
- `init_robot_position()` ramps from the current state to a hard-coded `start_position` (31 values committed verbatim in [`controller_bridge.py:418-450`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L418)) over 100 steps × 50 ms.
- `start_state_readers()` waits up to 10 s for all five sources to publish at least one sample, then prints `igris_c: state and cameras streaming.`
- `topic_health()` / `enable_health_log(period_sec)` — optional `BRIDGE_HEALTH_LOG` env-var-triggered Hz printer.
- `shutdown()` — stops the health thread and the per-reader poll threads.

**IGRIS_B counterpart.** [`igris_b/controller_bridge.py`](../../env_actor/robot_io_interface/robots/igris_b/controller_bridge.py) — `rclpy`-based, separate proprio assembly via `_obs_dict_to_np_array` and `IGRIS_B_STATE_KEYS`.

**Non-obvious side effects.** The IGRIS_C bridge sets `os.environ["CYCLONEDDS_URI"]` to the contents of `dds_state.xml` **once** at constructor time (line 116), and only if the env var was not already set. The camera participant inherits the same env var; if you want separate XML for cameras, set `CYCLONEDDS_URI` explicitly before launching. The XML files are documented in [04_runtime_configuration_igris_c.md](04_runtime_configuration_igris_c.md). Other side effects: head-camera rotation+crop happens inside `read_state`; if you swap cameras, change the conditional at [line 378-381](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L378).

### `env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py` (Modified)

**Why it changed.** The initial delay-queue value (`self._delay_queue.add(5)` → `add(15)`) was bumped. This is a tuning value used by guided inference to estimate the network/queue delay between the inference and control loops; the higher value matches measured IGRIS_C latencies and is also applied to IGRIS_B for consistency. Both occurrences (`__init__` line 85, `reset` line 405) were bumped.

**What it now does.** Functionally identical, just a different initial delay estimate.

**IGRIS_C counterpart.** [`rtc/data_manager/robots/igris_c/shm_manager_bridge.py`](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_c/shm_manager_bridge.py) — also `add(15)`.

**Non-obvious side effects.** Existing IGRIS_B deployments will see a slightly different initial guided-inference latency estimate after merging this branch. Steady-state behavior is unaffected because `MaxDeque` keeps only the last 5 samples; the initial estimate is replaced after the first inference cycle.

### `env_actor/policy/policies/openpi_policy/components/openpi_batched.yaml` (Modified)

**Why it changed.** IGRIS_C uses a 17-D action vector; the OpenPI policy must be built with `action_dim: 17`. The diff is one line.

```diff
- action_dim: 24
+ action_dim: 17
```

**What it now does.** Configures the OpenPI model factory to emit chunks of shape `(action_chunk_size, 17)`.

**IGRIS_B counterpart.** This is the same YAML used by IGRIS_B's policy. Note that switching `action_dim` here changes the model shape for **everyone**, not just IGRIS_C — IGRIS_B inference using this YAML will now break unless the checkpoint is also 17-D. `TODO:` either split into per-robot YAMLs or guard with `--policy_yaml_path`. The branch's current state has this YAML pointing at `pi05_igris_b_pnp_v3.3.2/film_15000`, an IGRIS_B checkpoint, with `action_dim: 17` — that combination is **inconsistent** and will only run end-to-end on IGRIS_C with a separate, 17-D-aware checkpoint. The default `--policy_yaml_path` in both entry points is the `dsrl_openpi_policy.yaml` (see [`run_inference.py:95`](../../run_inference.py#L95)) which composes differently — verify which YAML is actually used. See [09_troubleshooting_igris_c.md § action_dim mismatch](09_troubleshooting_igris_c.md#actiondim-mismatch-between-yaml-and-checkpoint).

## Submodule pointer

Between branch tips, the `trainer/` submodule SHA is **unchanged**:

```
main:        160000 commit 3ca051a256c9068f77b556df98f538d9a6185ccf trainer
igris_c/c1:  160000 commit 3ca051a256c9068f77b556df98f538d9a6185ccf trainer
```

However, **inside the branch's commit history** there is a commit `dd4e8a9 updated trainer` that bumps the pointer from `6cefd4715c33a731f6b8f57b80423780d8fb1f50` → `3ca051a256c9068f77b556df98f538d9a6185ccf`. The four upstream commits this brought in:

```
3ca051a updated documentation
c71ff08 removed unnecessary folders
4f4b4b9 fixing inconsistency
ca5de96 documented policy_constructor
```

In other words, the branch had been pointing at an older trainer SHA and was caught up to `main` near the end of development. Today the tips agree, and `git submodule update --init --recursive` after checking out `igris_c/c1` will land you at `3ca051a`.

## What did NOT change

- All the `docs/*.md` files that existed on `main` are present and unchanged on `igris_c/c1`. (They appear in `git diff --stat main...igris_c/c1` because they were added after the merge base; both branches gained them and they agree.)
- All folder READMEs (`env_actor/*/README.md`, `env_actor/auto/inference_algorithms/*/README.md`, etc.) are unchanged at the tips.
- `run_inference.py` is unchanged.
- `start_ray.sh`, `uv_setup.sh`, `env_setup.sh`, `openpi_transformer_lib_patch.sh` are unchanged.
- The IGRIS_B bridge (`robot_io_interface/robots/igris_b/controller_bridge.py`) is unchanged.
- The IGRIS_B sequential `data_manager_bridge.py` is unchanged.

---

← Back to index: [README.md](README.md) · Next → [03_robot_io_interface_igris_c.md](03_robot_io_interface_igris_c.md)
