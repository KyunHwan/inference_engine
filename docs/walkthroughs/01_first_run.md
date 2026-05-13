# Walkthrough 01 — First run

End-to-end "Hello, robot." Installs, starts Ray, runs inference. Includes annotated success-log output and "no robot" stopping points.

## Table of contents

- [Preconditions checklist](#preconditions-checklist)
- [Step 1: Configure start_ray.sh](#step-1-configure-start_raysh)
- [Step 2: Start Ray](#step-2-start-ray)
- [Step 3: Run inference (sequential first)](#step-3-run-inference-sequential-first)
- [Step 4: What a healthy startup log looks like](#step-4-what-a-healthy-startup-log-looks-like)
- [Step 5: Stop cleanly](#step-5-stop-cleanly)
- [Dry run / no robot](#dry-run--no-robot)
- [What to try next](#what-to-try-next)

## Preconditions checklist

You should be able to answer "yes" to all of these. If not, fix that thing before proceeding.

- [ ] `source .venv/bin/activate` puts you in a venv where `python --version` says `3.12.3` and `python -c "import torch; print(torch.cuda.is_available())"` says `True`.
- [ ] `nvidia-smi` shows your GPU and current drivers (we expect CUDA 13.0 wheels — `torch==2.9.0+cu130`).
- [ ] You've sourced ROS2: `python -c "import rclpy"` doesn't raise.
- [ ] You ran [openpi_transformer_lib_patch.sh](../../openpi_transformer_lib_patch.sh) if you plan to use OpenPI.
- [ ] You have a real OpenPI checkpoint at the path in `params.ckpt_dir` of [openpi_batched.yaml](../../env_actor/policy/policies/openpi_policy/components/openpi_batched.yaml). If you don't, skip ahead to [Dry run / no robot](#dry-run--no-robot).
- [ ] You have a valid `norm_stats_file_path` in [inference_runtime_params.json](../../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json), pointing to a pickle with at minimum `observation.state`, `observation.current`, and `action` keys with `mean` / `std` subkeys.

## Step 1: Configure start_ray.sh

Open [start_ray.sh](../../start_ray.sh). It is hostname-keyed: the script uses `hostname` to decide whether this machine should be the head or the worker.

For multi-machine setups, edit `HEAD_IP` and add a `case` for each hostname in your cluster.

For a **single-machine smoke test** (head and worker on the same box), you can skip the script entirely and run two commands in two terminals (see Step 2 below).

## Step 2: Start Ray

### Multi-machine

```bash
bash start_ray.sh
```

On the head:

```
Starting Ray Head Node on robros-ai1...
Local node IP: 100.109.184.39
...
```

On the worker:

```
Starting Ray Worker Node on robros-5090...
Ray runtime started.
```

### Single machine

Terminal 1 (head):

```bash
ray start --head --port=6379
```

Terminal 2 (worker, with `inference_pc` advertised):

```bash
ray start --address=127.0.0.1:6379 --resources='{"inference_pc": 1}'
```

Then verify:

```bash
ray status
```

Expected output fragment:

```
Resources
---------------------------------------------------------------
Usage:
 0.0/<n> CPU
 0.0/<n> GPU
 0.0/1.0 inference_pc            ← this must be non-zero
```

If `inference_pc` is missing, see [troubleshooting.md § Ray](../troubleshooting.md#ray).

## Step 3: Run inference (sequential first)

Sequential is easier to debug than RTC: one process, blocking calls, plain Python flow.

```bash
python run_inference.py --robot igris_b --inference_algorithm sequential
```

If you want to override the policy (e.g., to use plain OpenPI rather than the default DSRL+OpenPI):

```bash
python run_inference.py --robot igris_b --inference_algorithm sequential \
  -P ./env_actor/policy/policies/openpi_policy/openpi_policy.yaml
```

The default `--policy_yaml_path` is `./env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.yaml` — confirm with `python run_inference.py --help`.

## Step 4: What a healthy startup log looks like

Annotated:

```
2026-05-13 09:47:01,012  INFO ...  Ray runtime started.            ← ray.init connected
...
(SequentialActor pid=12345) Starting state readers...              ← controller_bridge.start_state_readers()
(SequentialActor pid=12345) Camera /dev/head_camera1 initialized:  1600.0x1200.0@60.0fps FOURCC=MJPG
(SequentialActor pid=12345) Camera /dev/right_camera1 initialized: 1600.0x1200.0@60.0fps FOURCC=MJPG
(SequentialActor pid=12345) Camera /dev/left_camera2 initialized:  1600.0x1200.0@60.0fps FOURCC=MJPG
(SequentialActor pid=12345) Camera recording started
(SequentialActor pid=12345) Waiting for proprio data to come in... ← rclpy subscription not yet populated
(SequentialActor pid=12345) Proprio state recording started        ← all observation keys delivered
(SequentialActor pid=12345) Warming up CUDA kernels...             ← policy.warmup()
(SequentialActor pid=12345) Initializing robot position...         ← controller_bridge.init_robot_position()
(SequentialActor pid=12345) Bootstrapping observation history...   ← data_manager_interface.init_inference_obs_state_buffer()
```

After this you're in the inner `for t in range(9000):` loop. You won't see `print` per step — the loop is intentionally quiet. If it's silent and `nvidia-smi` shows GPU activity once every ~2.5 s (= `policy_update_period / HZ`), it's working.

For RTC the startup is similar but you'll see logs from two child processes:

```
(start_control)   Starting control loop...
(start_inference) Warming up CUDA kernels...
(start_inference) Signaling inference ready...
(start_control)   Waiting for inference actor to be ready...
(start_control)   Starting episode 0...
(start_control)   Control loop started...
```

## Step 5: Stop cleanly

`Ctrl+C` in the terminal that ran `run_inference.py`. The main process catches `KeyboardInterrupt` and runs `ray.shutdown()`. The actor's `try/finally` then closes shared memory and shuts down rclpy.

Then on each Ray machine:

```bash
ray stop
```

This terminates Ray and frees the ports. If you skip it, your next `bash start_ray.sh` will complain about an already-running cluster.

## Dry run / no robot

You can exercise everything up to robot I/O without a robot — useful for verifying that policy loading works after a code change.

### What works without ROS2 / cameras

- Python imports.
- `build_policy()` and `policy.to("cuda")`.
- `policy.warmup()`.

### What fails without ROS2 / cameras

- `controller_bridge.__init__` — calls `rclpy.init()`. If ROS2 isn't sourced, this is your first crash.
- `controller_bridge._start_cam_recording` — `cv2.VideoCapture(/dev/head_camera1, CAP_V4L2)` raises `ValueError` if the device path doesn't exist.

### A targeted dry test

If you just want to confirm "my new policy class loads", you can write a small script:

```python
# scripts/load_only.py
import torch
torch.multiprocessing.set_start_method("spawn", force=True)

from env_actor.policy.utils.loader import build_policy

policy = build_policy(
    "./env_actor/policy/policies/openpi_policy/openpi_policy.yaml",
    map_location="cpu",
)
policy.to("cuda").eval()
print("loaded:", type(policy).__name__)
try:
    policy.warmup()
    print("warmup ok")
except Exception as e:
    print("warmup raised:", e)
```

Run it:

```bash
python scripts/load_only.py
```

No Ray, no cameras, no robot. If this passes, your loader path is healthy.

## What to try next

- [02_trace_one_step.md](02_trace_one_step.md) — line-by-line trace of one control iteration.
- If something crashed: [troubleshooting.md](../troubleshooting.md).
- If you want to change a parameter (HZ, image size, chunk size): [configuration_cookbook.md](../configuration_cookbook.md).
