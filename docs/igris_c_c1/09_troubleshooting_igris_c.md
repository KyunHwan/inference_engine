# 09 — Troubleshooting IGRIS_C failures

**What this covers.** The failure modes you will most likely hit when running IGRIS_C on this branch, with for each: the symptom, the likely cause, a diagnostic command, and a concrete fix.
**Who this is for.** You at 2 a.m. with the robot half-up and something not working.

If your problem is generic (not IGRIS_C-specific), start at [`docs/troubleshooting.md`](../troubleshooting.md). This file picks up where that leaves off, plus documents the discrepancies between the existing IGRIS_C subtree READMEs and what's actually on this branch.

## Table of contents

- [CUDA / spawn errors](#cuda--spawn-errors)
- [Ray cluster connectivity](#ray-cluster-connectivity)
- [Resource `inference_pc` not available](#resource-inference_pc-not-available)
- [Missing checkpoint](#missing-checkpoint)
- [Missing norm-stats pkl](#missing-norm-stats-pkl)
- [DDS topic mismatch / timeout in `start_state_readers`](#dds-topic-mismatch--timeout-in-start_state_readers)
- [Camera topic mismatch (no images, blank frames)](#camera-topic-mismatch-no-images-blank-frames)
- [Action denormalization producing out-of-range joint commands](#action-denormalization-producing-out-of-range-joint-commands)
- [Shared memory cleanup after a crash (RTC-specific)](#shared-memory-cleanup-after-a-crash-rtc-specific)
- [`proprio_state_dim` / `action_dim` mismatch](#proprio_state_dim--action_dim-mismatch)
- [`action_dim` mismatch between YAML and checkpoint](#actiondim-mismatch-between-yaml-and-checkpoint)
- [`cyclonedds` not installed](#cyclonedds-not-installed)
- [Docs/code inconsistencies on this branch](#docscode-inconsistencies-on-this-branch)

## CUDA / spawn errors

**Symptom.** Crash early at process startup with:

```
RuntimeError: Cannot re-initialize CUDA in forked subprocess.
To use CUDA with multiprocessing, you must use the 'spawn' start method.
```

**Likely cause.** Something in your environment forked the Python process **after** torch/CUDA was initialized in the parent. The entrypoints call `torch.multiprocessing.set_start_method("spawn")` at the very top ([`run_inference.py:88`](../../run_inference.py#L88), [`run_inference_local.py:71`](../../run_inference_local.py#L71)), but if a wrapper script or notebook imports torch first, the call has no effect (because Python's `multiprocessing` only honors the start-method change before the context is first used).

**Diagnostic.** Add a print right at the top of the wrapper:

```python
import torch
print(torch.cuda.is_initialized())
# if True before set_start_method, you have the bug.
```

**Fix.** Move `set_start_method("spawn")` ahead of any torch / CUDA import in your wrapper. Or, equivalently, call `import torch.multiprocessing as mp; mp.set_start_method("spawn", force=True)` and pay attention to the force= kwarg if you've already used the context. The RTC actor itself uses `mp.get_context("spawn")` explicitly ([`rtc_actor.py:47`](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L47)), so once you make it past the parent's startup you're fine.

## Ray cluster connectivity

**Symptom.** `python run_inference.py ...` hangs at `ray.init(address="auto", ...)`, or worker count stays at 0 in `ray status`.

**Likely causes:**

1. `start_ray.sh` didn't actually run on the worker (or ran with `Unknown host` because the hostname doesn't match the `case` arms). The script's hostname switch is at [`start_ray.sh:28-49`](../../start_ray.sh#L28).
2. `HEAD_IP` in `start_ray.sh` is wrong, or the IP is unreachable from the worker.
3. Firewall blocking Ray's GCS port (`6379` on this branch's `start_ray.sh:32`) or worker ports.
4. Different Python/Ray versions on head and worker.

**Diagnostic.**

```bash
# On worker:
ping <HEAD_IP>            # expect reachable
nc -vz <HEAD_IP> 6379     # expect "open" / "succeeded"
ray status                # expect head + this worker listed

# Compare versions:
python -c "import ray, sys; print(sys.executable, ray.__version__)"
# Same Python interpreter path + Ray version on head and worker.
```

**Fix.**

- Edit `start_ray.sh:25` (`HEAD_IP`) and `start_ray.sh:28-49` (`case "$HOSTNAME"` arms) to match your hosts. **Current values on this branch — verify against your environment.**
- If the worker IP changes when on/off VPN, hard-code `--node-ip-address=<your_worker_ip>` (line 40).
- If firewall blocks 6379, open it or pick another port (then update `--port=` on the head and `--address=...:` on the worker).

## Resource `inference_pc` not available

**Symptom.**

```
RuntimeError: No available resources. The actor request is asking for: 1 inference_pc.
```

**Likely cause.** The worker didn't register the `inference_pc:1` custom resource. The actor placement at [`run_inference.py:54`](../../run_inference.py#L54) (`options(resources={"inference_pc": 1}, num_cpus=3, num_gpus=1)`) requires this resource to exist somewhere in the cluster.

**Diagnostic.**

```bash
ray status
# Expect "Resources" section to include 'inference_pc: 1.0/1.0' on the worker line.
```

**Fix.** Make sure `start_ray.sh:43` includes `--resources='{"inference_pc": 1}'` on the worker arm, and re-run `bash start_ray.sh` on the worker.

## Missing checkpoint

**Symptom.** Policy build fails with `FileNotFoundError` or `OSError: ckpt_dir does not exist` during `build_policy(...)`.

**Likely cause.** The `ckpt_dir` in [`openpi_batched.yaml:5`](../../env_actor/policy/policies/openpi_policy/components/openpi_batched.yaml#L5) points at an absolute path on the author's machine (`/home/robros/Projects/robros_vla_inference_engine/openpi_film/checkpoints/...`).

**Fix.** Change `ckpt_dir` to your local checkpoint path. The directory should contain whatever your policy's loader expects (typically a Flax/Pytorch state dict). See [04 § Placeholders](04_runtime_configuration_igris_c.md#placeholders-you-must-change-for-your-environment).

## Missing norm-stats pkl

**Symptom.**

```
File not found at: /home/robros/Projects/inference_engine/trainer/experiment_training/igris_c/dataset_stats.pkl
... (later) ...
TypeError: 'NoneType' object is not subscriptable
```

The first line is from [`inference_runtime_params.py:134`](../../env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.py#L134) (print, not raise). The second comes from the first attempt to subscript `norm_stats['observation.state']` inside the bridge.

**Likely cause.** `norm_stats_file_path` in `inference_runtime_params.json` is wrong, the file doesn't exist on this host, or it's not readable.

**Diagnostic.**

```bash
python -c "import json; print(json.load(open('./env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json'))['norm_stats_file_path'])"
ls -l "<the path it printed>"
```

**Fix.** Point `norm_stats_file_path` at a real pkl with the schema documented in [06 § Stats source](06_normalization_igris_c.md#stats-source-dataset_statspkl).

## DDS topic mismatch / timeout in `start_state_readers`

**Symptom.**

```
[Internal] Starting state reading...
low=.. hand=.. head=.. left=.. right=..
low=.. hand=.. head=.. left=.. right=..
...
RuntimeError: Timeout waiting for igris_c lowstate/handstate/cameras.
```

(The `..` means "no sample yet" — `OK` would mean "got at least one sample.")

**Likely causes:**

1. **NIC pinning wrong.** Your host's `enp11s0` interface doesn't carry the `192.168.10.x/24` subnet, or the interface isn't called `enp11s0` at all.
2. **Topic name mismatch.** The NUC publishes on a different topic than `rt/lowstate`, `rt/handstate`.
3. **Wrong domain.** The NUC publishes on a domain other than 0 (e.g. 10 for some configurations).
4. **IDL hash mismatch.** Your `messages/igris_c_msgs.py` has been edited and the `typename=` no longer matches the NUC firmware.
5. **NUC/Jetson not running.** The robot side simply isn't publishing.

**Diagnostic.**

```bash
# Show interfaces and their IPs:
ip addr show

# On your host, observe DDS traffic with cyclonedds tooling (if installed):
CYCLONEDDS_URI="$(cat env_actor/runtime_settings_configs/robots/igris_c/dds/dds_state.xml)" \
  cyclonedds ps                       # lists participants on Domain 0

# Or use record-BE's monitor if you have it.

# Confirm you can reach the NUC:
ping 192.168.10.1   # or whatever your NUC's IP is

# Enable health logging on the bridge:
BRIDGE_HEALTH_LOG=1 python -m env_actor.robot_io_interface.robots.igris_c.run_bridge_monitor
```

**Fixes:**

- **NIC**: edit `dds/dds_state.xml` and `dds/dds_camera.xml` to use the right `<NetworkInterface name="...">`.
- **Topics**: edit `dds.topics.*` in `inference_runtime_params.json`.
- **Domain**: edit `dds.state_domain_id` / `dds.camera_domain_id` and the matching `<Domain id="...">` in the XMLs.
- **IDL**: revert `messages/igris_c_msgs.py` to the branch version; do not change `typename=` annotations.

## Camera topic mismatch (no images, blank frames)

**Symptom.** `body_q` and `hand_q` are coming through fine but the camera image keys are `None` for the first 10 s, then `read_state` returns zero-filled images. The bridge prints `<key> image is None !!` repeatedly (from [`controller_bridge.py:374`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L374)).

**Likely cause.** Camera topic name in `dds.topics.head_camera/left_camera/right_camera` is wrong, or the camera domain is wrong (Domain 1 by default; some operator setups use Domain 10).

**Diagnostic.**

```bash
BRIDGE_HEALTH_LOG=1 python -m env_actor.robot_io_interface.robots.igris_c.run_bridge_monitor
# Look for head_cam=0.0Hz(age n/a)  — confirms zero camera frames are arriving.
```

**Fixes:**

- Update the three camera topic names in `inference_runtime_params.json:28-30`.
- If on WiFi/NUC bridge: change `dds.camera_domain_id` to `10` **and** change `<Domain id="1">` to `<Domain id="10">` in `dds/dds_camera.xml`. The XML's own comment block at [`dds/dds_camera.xml:1-13`](../../env_actor/runtime_settings_configs/robots/igris_c/dds/dds_camera.xml#L1) documents this.

## Action denormalization producing out-of-range joint commands

**Symptom.** Robot snaps to an unsafe pose on the first action, or oscillates wildly, or you see joint-limit violations on the NUC.

**Likely causes:**

1. **Wrong `dataset_stats.pkl`** — using IGRIS_B stats with IGRIS_C runs, or stats from a different training run, will denormalize the policy output into completely wrong real-units actions.
2. **`max_delta_deg`** is too generous for your hardware — the bridge applies a slew limit at 5°/step × 20 Hz = 100°/s. That is appropriate for IGRIS_C arms; verify the limit is sensible if you have changed it.
3. **Stale `prev_joint`** — if you bypassed `controller_interface.publish_action`'s return value, `prev_joint` falls behind, the slew limit clamps the delta to ±5° regardless, and the bot creeps slowly toward an unrelated pose.
4. **Hand value out of range.** The hand command's `q` is normalized 0–1 per the NUC schema docstring at [`igris_c_msgs.py:122-127`](../../env_actor/robot_io_interface/robots/igris_c/messages/igris_c_msgs.py#L122). If your policy emits a hand action of, say, -3.0, that may either saturate or trigger a fault.

**Diagnostic.** Add a print in `publish_action` to log the raw action vs. the smoothed:

```python
print(f"raw17={a.tolist()} smoothed31_first3={smoothed31[:3].tolist()}")
```

If `raw17` has values far outside `[-π, π]` for joints or far outside `[0, 1]` for hand values, the denormalization is the culprit.

**Fixes:**

- Re-verify `norm_stats_file_path` points at IGRIS_C-trained stats with `action` mean/std of length 17.
- Lower `max_delta_deg` to 2 or 1 for first runs.
- Confirm the calling code uses `smoothed31` (the first return value of `publish_action`) as `prev_joint` for the next call. The control loops in [`rtc/actors/control_loop.py:135-138`](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L135) and [`sequential_actor.py:136-142`](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L136) already do.

## Shared memory cleanup after a crash (RTC-specific)

**Symptom.** On the next launch:

```
FileExistsError: [Errno 17] File exists: '/psm_abcd1234'
```

or on shutdown:

```
.../resource_tracker.py: UserWarning: resource_tracker: There appear to be N leaked shared_memory objects to clean up at shutdown
```

**Likely cause.** A previous run ended uncleanly (SIGKILL, segfault, OOM) before the parent ran `shm.close()` / `shm.unlink()` / `resource_tracker.unregister(...)` for each block ([`rtc_actor.py:152-166`](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L152)).

**Diagnostic / fix (Linux).**

```bash
ls /dev/shm | grep -E '^(psm_|np_)'    # see the survivors
# Remove them — safe ONLY if no current run is using them:
ls /dev/shm | grep -E '^psm_' | xargs -r -I{} rm /dev/shm/{}
```

If you run multiple instances concurrently, do not blanket-`rm` — match against your dead PID's block names instead. See `multiprocessing.shared_memory.SharedMemory.name` format.

## `proprio_state_dim` / `action_dim` mismatch

**Symptom (proprio).**

```
ValueError: could not broadcast input array from shape (86,) into shape (24,)
```

or

```
IndexError: index out of range  (slicing state_mean[:86] when stats vector is shorter)
```

**Likely cause.** `proprio_state_dim` in `inference_runtime_params.json` does not match what the bridge actually produces (86 for IGRIS_C), or does not match the trained stats file's `observation.state` length.

**Symptom (action).**

```
AssertionError: expected (17,), got (24,)
```

(From [`controller_bridge.py:389`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L389).)

**Likely cause.** Policy was built with `action_dim: 24` but you set `inference_runtime_params.json:action_dim: 17`, or vice versa.

**Fix.** Make all three agree:
- `init_params.py` (`PROPRIO_STATE_DIM`, `INIT_ACTION_17`),
- `inference_runtime_params.json` (`proprio_state_dim`, `action_dim`),
- `openpi_batched.yaml` (`action_dim`).

And re-train the policy if you change these.

## `action_dim` mismatch between YAML and checkpoint

**Symptom.** Policy build silently succeeds but inference produces actions of the wrong shape, leading to the assertion above; or you get a mismatched-shape error inside the policy's forward call.

**Likely cause.** [`openpi_batched.yaml:7`](../../env_actor/policy/policies/openpi_policy/components/openpi_batched.yaml#L7) was bumped from `24 → 17` on this branch, but the `ckpt_dir` on the same file still points at `pi05_igris_b_pnp_v3.3.2/film_15000` — an IGRIS_B checkpoint trained with 24-D actions. The two are inconsistent; running with the default `--policy_yaml_path` flag pointing at `dsrl_openpi_policy.yaml` may mask this depending on how that YAML composes the openpi component.

**Diagnostic.** Read both your `--policy_yaml_path` and the component YAMLs it includes. Confirm `action_dim` and `ckpt_dir` are mutually consistent — same dim end-to-end.

**Fix.** Either:
- Train an IGRIS_C-specific 17-D OpenPI checkpoint and point `ckpt_dir` at it; or
- If you're testing the IGRIS_C pipeline without a real policy, mock the policy.

`TODO:` split `openpi_batched.yaml` into `openpi_batched_igris_b.yaml` and `openpi_batched_igris_c.yaml` so this can't drift again.

## `cyclonedds` not installed

**Symptom.**

```
ModuleNotFoundError: No module named 'cyclonedds'
```

at the import block of [`controller_bridge.py:31-36`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L31).

**Likely cause.** `env_setup.sh` does not install `cyclonedds-python`. The IGRIS_C branch added the dependency to the active bridge but didn't add the `uv pip install` line.

**Fix.**

```bash
source .venv/bin/activate
uv pip install cyclonedds
```

`TODO:` add this to `env_setup.sh` (do not do it as part of this docs-only change).

## Docs/code inconsistencies on this branch

This branch ships some folder READMEs that **predate** the implementation. They aren't wrong per se — they were accurate at the time — but they no longer reflect the current state on `igris_c/c1`. Where they disagree, the **code** is the source of truth.

| Inconsistency | What the README says | What the code shows |
|---|---|---|
| [`env_actor/robot_io_interface/robots/igris_c/README.md`](../../env_actor/robot_io_interface/robots/igris_c/README.md) | "Status: Interface Design Only — Full implementation deferred until hardware specifications are available." | Full implementation present in [`controller_bridge.py`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py), `controller_bridge_sdk_legacy.py`, `messages/igris_c_msgs.py`. Hardware specs are now embodied in [`init_params.py`](../../env_actor/runtime_settings_configs/robots/igris_c/init_params.py). |
| [`env_actor/robot_io_interface/robots/igris_c/README.md`](../../env_actor/robot_io_interface/robots/igris_c/README.md) (Reference paths) | References `/env_actor/auto/io_interface/robots/igris_b/controller_bridge.py` and `/env_actor/auto/data_manager/robots/igris_b/data_manager_bridge.py` | The actual locations are `env_actor/robot_io_interface/robots/igris_b/...` and `env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_b/...`. The README's paths are pre-refactor. |
| [`env_actor/runtime_settings_configs/robots/igris_c/README.md`](../../env_actor/runtime_settings_configs/robots/igris_c/README.md) | "`inference_runtime_params.json` — **Missing.** Required before any IGRIS_C run can start." | The file exists on this branch — [`inference_runtime_params.json`](../../env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json). The "What needs to be created" checklist is done. |
| [`env_actor/runtime_settings_configs/robots/igris_c/README.md`](../../env_actor/runtime_settings_configs/robots/igris_c/README.md) | "`init_params.py` — Stub: `INIT_JOINT_LIST = []`, `IGRIS_C_STATE_KEYS = []`, etc." | The file now contains the real values (`N_JOINTS = 31`, full `HOME_POSITION`, `INIT_ACTION_17`, kp/kd, etc.). |
| `run_inference.py` defaults | `--inference_runtime_params_config` defaults to **IGRIS_B** path. | When using `--robot igris_c`, you **must** pass the IGRIS_C path explicitly. |
| `run_inference_local.py` defaults | `--inference_runtime_params_config` defaults to **IGRIS_C** path. | Inverse of `run_inference.py`. Confusing — if you use the wrong default with the wrong `--robot`, the loader silently succeeds and shapes break much later. |

I have **not** modified the stale READMEs (per the hard constraint to leave existing files alone). If you want to fix them, do it in a follow-up branch.

---

← Back to index: [README.md](README.md) · Next → [10_glossary_and_references.md](10_glossary_and_references.md)
