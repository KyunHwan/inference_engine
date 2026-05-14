# 08 — Running and extending IGRIS_C

**What this covers.** The CLI surface on this branch, the day-to-day operating commands, and "if I want to change X" recipes for the most common modifications: action dim, cameras, policy, observation channels. Also: an explicit list of what is intentionally **not yet supported** on `igris_c/c1`.
**Who this is for.** Whoever owns the system once it is running.

## Table of contents

- [CLI reference](#cli-reference)
- [Day-to-day commands](#day-to-day-commands)
  - [Starting Ray](#starting-ray)
  - [Launching inference](#launching-inference)
  - [Stopping cleanly](#stopping-cleanly)
  - [Restarting after a crash](#restarting-after-a-crash)
- ["If I want to change X" recipes](#if-i-want-to-change-x-recipes)
- [What is intentionally NOT yet supported](#what-is-intentionally-not-yet-supported)

## CLI reference

### `run_inference.py` (Ray path)

[`run_inference.py:92-114`](../../run_inference.py#L92), verified against the branch.

| Flag | Type | Default (on this branch) | Description |
|---|---|---|---|
| `--policy_yaml_path`, `-P` | str | `./env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.yaml` | Path to the top-level policy YAML. |
| `--robot` | str | `igris_b` | `{igris_b | igris_c}` |
| `--inference_runtime_params_config` | str | `./env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json` | Path to the runtime params JSON. **Default is IGRIS_B even when `--robot igris_c` — pass the IGRIS_C path explicitly.** |
| `--inference_runtime_topics_config` | str | `./env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_topics.json` | Path to the topics JSON. Same warning. |
| `--inference_algorithm` | str | `rtc` | `{rtc | sequential}` |

There is no `--help` example in the file beyond the default argparse output. Run `python run_inference.py --help` to see them at any time.

### `run_inference_local.py` (no-Ray path)

[`run_inference_local.py:75-97`](../../run_inference_local.py#L75), verified against the branch.

Same flags. Two defaults differ from `run_inference.py`:

| Flag | Default in `run_inference_local.py` |
|---|---|
| `--inference_runtime_params_config` | `./env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json` |
| `--inference_runtime_topics_config` | `./env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_topics.json` |

So `python run_inference_local.py --robot igris_c` works out of the box (config defaults match the robot), but `python run_inference_local.py --robot igris_b` will load IGRIS_C configs unless you pass IGRIS_B paths explicitly. This is the inverse of `run_inference.py`. `TODO:` align the defaults — see [09 § Doc/code inconsistencies](09_troubleshooting_igris_c.md#docscode-inconsistencies-on-this-branch).

### `run_bridge_monitor` (DDS bridge sanity check)

[`env_actor/robot_io_interface/robots/igris_c/run_bridge_monitor.py`](../../env_actor/robot_io_interface/robots/igris_c/run_bridge_monitor.py) — no CLI flags. Hard-codes `CFG` on line 16 (an absolute path on the author's machine; change it).

```bash
python -m env_actor.robot_io_interface.robots.igris_c.run_bridge_monitor
```

Exit with `Ctrl+C`.

### `BRIDGE_HEALTH_LOG` (env var)

Set to `1` (or `<seconds>` for a custom period) before launching any of the above to get per-topic Hz + age prints. See [03 § Health monitoring](03_robot_io_interface_igris_c.md#health-monitoring).

## Day-to-day commands

### Starting Ray

Only required for `run_inference.py` (not `run_inference_local.py`).

```bash
# On each machine (head, worker). Edit start_ray.sh first.
bash start_ray.sh

# Verify
ray status
# Expect: head + worker(s) registered, worker shows {"inference_pc": 1.0}.
```

If `ray status` shows zero workers, see [09_troubleshooting_igris_c.md § Ray cluster connectivity](09_troubleshooting_igris_c.md#ray-cluster-connectivity).

### Launching inference

```bash
# Local, RTC, IGRIS_C (single-machine dev):
source .venv/bin/activate
python run_inference_local.py --robot igris_c

# Cluster, RTC, IGRIS_C (multi-machine production):
source .venv/bin/activate
python run_inference.py \
  --robot igris_c \
  --inference_runtime_params_config ./env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json \
  --inference_runtime_topics_config ./env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_topics.json
```

Expected log progression (see [01 § First-time success signals](01_quickstart_igris_c.md#first-time-success-signals) for the full sequence).

### Stopping cleanly

`Ctrl+C` in the foreground process. The entry points install a `KeyboardInterrupt` handler that calls `ray.shutdown()` (for the Ray path) and exits — see [`run_inference.py:76-83`](../../run_inference.py#L76).

For the RTC path specifically, both `RTCActor.start()` and `RTCLocalActor.start()` install a `finally:` block that:
1. Sets `stop_event` and notifies both Conditions ([`rtc_actor.py:140-145`](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L140)).
2. `join(timeout=5)` then `terminate()` then `join(timeout=3)` for each child.
3. Closes + unlinks every SharedMemory block.
4. Calls `resource_tracker.unregister(...)` for each block to suppress the leak warning.

After exit:

```bash
ray stop                          # only if you started Ray
ls /dev/shm | grep -i psm         # confirm no leftover SHM blocks (Linux)
```

If you see leftover SHM blocks named like `psm_*` (the prefix used by `multiprocessing.shared_memory`), see [09 § Shared memory cleanup after a crash](09_troubleshooting_igris_c.md#shared-memory-cleanup-after-a-crash-rtc-specific).

### Restarting after a crash

If the process died ungracefully (`SIGKILL`, OOM, segfault), the cleanup chain didn't run. To get back to a clean state:

```bash
# 1. Make sure no inference process is still running.
ps -ef | grep -E "run_inference|rtc_actor|sequential_actor" | grep -v grep
# Kill any survivors with `kill <pid>`.

# 2. Clean stale SHM blocks (Linux).
ls /dev/shm | grep -E '^(psm_|np_)' | xargs -r -I{} rm /dev/shm/{}

# 3. (Ray path) restart Ray.
ray stop
bash start_ray.sh

# 4. Relaunch.
python run_inference_local.py --robot igris_c
```

`TODO:` automate steps 1-2 in a wrapper script.

## "If I want to change X" recipes

Each recipe lists **every file you must touch**. Following them in order prevents the "I changed only one place and now things are weird" class of bug.

### Change `action_dim` from 17 to something else

1. [`init_params.py`](../../env_actor/runtime_settings_configs/robots/igris_c/init_params.py): redefine `INIT_ACTION_17` (rename it too, if you want consistency). Recompute `LEFT_ARM_IDS`, `RIGHT_ARM_IDS`, `WAIST_YAW_ID` such that they cover the new action layout.
2. [`controller_bridge.py:389`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L389): update the `assert a.shape == (17,)` to the new dim.
3. [`controller_bridge.py:391-403`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L391): update the 17-D → 31-D slice mapping in `publish_action` and the hand broadcast logic.
4. [`inference_runtime_params.json:11`](../../env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json#L11): update `"action_dim"`.
5. [`openpi_batched.yaml:7`](../../env_actor/policy/policies/openpi_policy/components/openpi_batched.yaml#L7): update `action_dim` to the new value, **and** point `ckpt_dir` at a checkpoint trained with the new dim.
6. Both `shm_manager_bridge.py` files (`rtc/` and `rtc_local/`): the imports use `INIT_ACTION_17`. If you renamed it, update those imports.
7. Both `data_manager_bridge.py` files for sequential — same.

### Change camera setup

This is the riskier change — it touches the bridge, the SHM layout, and the policy input layout.

1. [`controller_bridge.py:193-206`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L193): update the per-camera reader spawn block. Add/remove cache entries in `_latest_img`.
2. [`controller_bridge.py:366-383`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L366): update `read_state()` to produce the new camera keys; remove the head-rotate-and-crop block if it no longer applies.
3. [`inference_runtime_params.json:9`](../../env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json#L9): update `camera_names`. Keep the order in sync with `read_state` and SHM blocks.
4. [`inference_runtime_params.json:28-30`](../../env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json#L28): update the DDS topic names in `dds.topics`.
5. Both `rtc_actor.py` and `rtc_local_actor.py` ([`rtc_actor.py:58-66`](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L58)): add or remove the per-camera SHM block creation. The `shm_specs` dict ([line 71-77](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L71)) must list every key the bridge produces.
6. Sequential and RTC data managers: confirm they iterate `runtime_params.camera_names` instead of hard-coding `["head", "left", "right"]`. As of this branch they do, so no change needed unless you broke that.
7. Re-train your policy with the new image layout, or accept that the policy will see noise on the changed channels.

### Change the policy

1. Most common path: pass `--policy_yaml_path /path/to/your/policy.yaml`. The CLI accepts any path; the policy loader builds whatever is named.
2. If your policy has a different name, register it in [`env_actor/policy/registry/`](../../env_actor/policy/registry/) (see [`docs/walkthroughs/03_add_a_new_policy.md`](../walkthroughs/03_add_a_new_policy.md)).
3. Confirm your policy's `action_dim` matches `inference_runtime_params.json:action_dim` (17).
4. Confirm your policy's expected proprio dim matches `inference_runtime_params.json:proprio_state_dim` (86) **and** matches your `dataset_stats.pkl`'s `observation.state` length.

### Add a new observation channel (e.g. an IMU or F/T sensor)

Currently the bridge reads only proprio + 3 cameras. To add a channel:

1. **Wire side**: confirm the new sensor is published on a DDS topic (probably under `rt/...` or `igris_c/sensor/...`).
2. **Message type**: if the new sensor's message type isn't already in [`messages/igris_c_msgs.py`](../../env_actor/robot_io_interface/robots/igris_c/messages/igris_c_msgs.py), add the dataclass with the right `typename=...` matching the firmware-side IDL.
3. **Bridge**: add a reader spawn block in `controller_bridge.py` similar to `_spawn(self._hand_reader, on_hand, "handstate")`. Add a `_latest_<sensor>` cache.
4. **`read_state()`**: surface the new sensor's value as a new key in the returned dict, or concatenate into `proprio` if it's a continuous-valued vector.
5. **`init_params.py`**: if you concatenated, update `PROPRIO_*_DIM` constants and `PROPRIO_STATE_DIM`. Update `IGRIS_C_STATE_KEYS`.
6. **`inference_runtime_params.json:proprio_state_dim`**: match the new total.
7. **Policy + stats**: retrain. The `dataset_stats.pkl` will need to be regenerated with the new proprio length.

### Adjust max velocity (slew limit)

Easiest single-knob change.

1. [`inference_runtime_params.json:3`](../../env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json#L3): change `max_delta_deg`. The bridge converts to radians automatically.
2. Rule of thumb: max velocity (deg/s) ≈ `HZ × max_delta_deg`. Current `20 × 5 = 100°/s`. Doubling `max_delta_deg` to `10` allows 200°/s — be cautious, this can exceed safe joint velocities.

### Adjust per-motor gains

1. [`inference_runtime_params.json:33-53`](../../env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json#L33): edit the `kp` and `kd` arrays under `dds.joint_gains`. Both must have exactly 31 entries — the bridge asserts this at construction ([`controller_bridge.py:219-220`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L219)).
2. For hand gains, edit `hand_kp` / `hand_kd` in the same block.
3. Restart the inference process; gains are read once at bridge construction and baked into every `MotorCmd`.

### Use a different `dataset_stats.pkl`

1. [`inference_runtime_params.json:16`](../../env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json#L16): change `norm_stats_file_path` (absolute path).
2. The pkl must contain `observation.state` (mean/std of length ≥ 86) and `action` (mean/std of length 17). See [06](06_normalization_igris_c.md).
3. No code change needed.

## What is intentionally NOT yet supported

- **Multi-episode local RTC.** `rtc_local/actors/control_loop.py` collapses the per-episode outer `while True` of the Ray variant into a single-episode flow. After one episode, the control process exits. To run multiple episodes locally, use the Ray path (`run_inference.py`), or `TODO:` port the outer loop from `rtc/actors/control_loop.py`. Tracked in [02 § rtc_local](02_changes_vs_main.md#env_actorautoinference_algorithmsrtc_local-and-the-12-other-files-under-rtc_local-added--13-files).

- **End-to-end sequential local inference.** `sequential_local/sequential_local_actor.py:start()` is **currently a smoke test**. It reads one observation, dumps three camera frames to `head.png`, `left.png`, `right.png`, then returns. The actual `policy.predict` → `publish_action` chain is in a commented-out triple-quoted string ([lines 134-176](../../env_actor/auto/inference_algorithms/sequential_local/sequential_local_actor.py#L134)). Uncomment that block (and the `DataNormalizationInterface` constructor on line 85) to get full sequential inference locally. `TODO:` finish this — the branch ships the smoke-test version intentionally.

- **`igris_c_sdk` path.** The `controller_bridge_sdk_legacy.py` file is dormant. The active bridge uses `cyclonedds-python` directly because of IDL hash mismatches; reviving the SDK path requires regenerating `igris_c_sdk` against the NUC's actual `igris_c_msgs.idl`. See [02 § controller_bridge_sdk_legacy.py](02_changes_vs_main.md#env_actorrobot_io_interfacerobotsigris_ccontroller_bridge_sdk_legacypy-added).

- **Service calls (BMS init, torque control, control-mode switch).** The DDS message types for these are declared in [`messages/igris_c_msgs.py`](../../env_actor/robot_io_interface/robots/igris_c/messages/igris_c_msgs.py) (`BmsInitCmd`, `TorqueCmd`, `ControlModeCmd`, `ServiceResponse`). The active bridge does not call them — bring-up assumes the operator has already brought the BMS up and torque on via an external tool. `TODO:` if you need automated BMS init at startup, see the legacy SDK bridge's `init_robot_at_startup` branch ([`controller_bridge_sdk_legacy.py:92-100`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge_sdk_legacy.py#L92)) for the pattern.

- **Camera XML separation.** `dds_camera.xml` exists but `cyclonedds-python` reads `CYCLONEDDS_URI` once — the bridge sets it from the **state** XML. Both XMLs currently pin the same NIC, so this works, but if you ever need genuinely different camera config, you'd have to pre-set `CYCLONEDDS_URI` before launch. `TODO:` evaluate whether multi-XML support is feasible.

- **`init_robot_at_startup` for the active bridge.** The flag exists in `inference_runtime_params.json` but the active `controller_bridge.py` does not honor it (only the SDK legacy file does). To enable, port the BMS/torque init logic from the SDK file.

- **Dynamic camera resize.** `mono_image_resize` width/height are baked into the SHM blocks at parent-process startup and into the policy's expected input shape. Changing them mid-run is not supported; restart is required.

- **Robot-specific safety limits beyond `max_delta_deg`.** There is no per-joint position limit check in the bridge. The slew-rate limit catches a single-step velocity violation, but the policy could still command a long sequence of small deltas that walks the robot into a joint limit. `TODO:` add per-joint position clipping if your hardware does not enforce it firmware-side.

- **Telemetry beyond `BRIDGE_HEALTH_LOG`.** No Prometheus, no Grafana, no structured logs. `topic_health()` is callable programmatically; if you want a long-term monitoring story, build on top of it.

---

← Back to index: [README.md](README.md) · Next → [09_troubleshooting_igris_c.md](09_troubleshooting_igris_c.md)
