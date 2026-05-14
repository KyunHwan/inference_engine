# 01 — Quickstart: zero to running on IGRIS_C

**What this covers.** Everything a junior engineer needs to go from `git clone` to a running inference loop on the IGRIS_C robot, on the `igris_c/c1` branch.
**Who this is for.** Someone who has not run this repo before. If you have, skip to [§ Running inference](#running-inference).

This guide aims to be specific. Where a value is environment-dependent (an IP, a path, a NIC name), it is either pulled verbatim from this branch and labelled "current value on this branch — verify against your environment," or shown as `<PLACEHOLDER>` with a `TODO`.

## Table of contents

- [Prerequisites](#prerequisites)
- [Clone and check out `igris_c/c1`](#clone-and-check-out-igris_cc1)
- [Install Python + dependencies](#install-python--dependencies)
- [Sanity-check the IGRIS_C config](#sanity-check-the-igris_c-config)
- [Two ways to run: Ray cluster vs. local](#two-ways-to-run-ray-cluster-vs-local)
- [Running inference](#running-inference)
- [First-time success signals](#first-time-success-signals)
- [If it crashes immediately](#if-it-crashes-immediately)

## Prerequisites

The root [`README.md § Prerequisites`](../../README.md#prerequisites) lists the canonical requirements. They are reproduced here so you can verify them in one place — but **verify these are still current on this branch** by re-reading the root README before depending on them. As of `igris_c/c1`:

| Requirement | Details (current on this branch) | Where it lives |
|---|---|---|
| Python | 3.12.3 (pinned by `uv venv`) | [`uv_setup.sh:5`](../../uv_setup.sh#L5) |
| GPU | NVIDIA, CUDA 13.0 (PyTorch wheel `cu130`) | [`env_setup.sh:4`](../../env_setup.sh#L4) |
| OS | Linux. Tested on Ubuntu. | root README |
| Robot comms | Cyclone DDS (via `cyclonedds-python`). **No ROS2 needed for IGRIS_C.** This is a departure from IGRIS_B which uses `rclpy`. | [`controller_bridge.py:31-46`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L31) |
| Robot network | LAN cabled to the NUC (state/cmd domain) and to the Jetson (camera domain). DDS domain IDs `0` (state) and `1` (camera) are the current branch defaults — see [`inference_runtime_params.json:19-20`](../../env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json#L19). | this branch |

Two things specifically **not** required for IGRIS_C (but required for IGRIS_B): a working `rclpy`/ROS2 install on the worker node, and V4L2 device nodes like `/dev/head_camera1`. IGRIS_C reads cameras off the DDS topic `igris_c/sensor/eyes_stereo` etc. — they arrive as JPEG bytes that the bridge decodes with OpenCV ([`controller_bridge.py:372-383`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L372)).

## Clone and check out `igris_c/c1`

```bash
# 1. Clone with submodules. The default submodule URL is SSH; if you don't have
#    SSH keys for KyunHwan/trainer, see the "No SSH key" note in the root README.
git clone --recurse-submodules <repo-url>
cd inference_engine

# 2. Check out the branch.
git checkout igris_c/c1

# 3. Re-sync submodules — this branch may have a different trainer SHA than main
#    inside its history. Today the tips agree, but always run this after a checkout.
git submodule update --init --recursive

# 4. Verify.
git rev-parse HEAD           # expected: a commit on origin/igris_c/c1
git ls-tree HEAD -- trainer  # expected: 160000 commit 3ca051a256c9068f77b556df98f538d9a6185ccf
```

Expected `git ls-tree` output (current on this branch):

```
160000 commit 3ca051a256c9068f77b556df98f538d9a6185ccf	trainer
```

If the SHA differs, your `git submodule update` either didn't run or your local main has drifted. Re-run the submodule update and re-check.

## Install Python + dependencies

The branch does **not** change `uv_setup.sh` or `env_setup.sh` between `main` and `igris_c/c1`. The root README instructions apply verbatim. Restated here for one-stop reading:

```bash
# Install uv and create a Python 3.12.3 venv at .venv/
bash uv_setup.sh

# Activate the venv (do this in every shell that runs inference)
source .venv/bin/activate

# Install Python packages — PyTorch 2.9.0 + CUDA 13.0, Ray, transformers, etc.
bash env_setup.sh

# Only if your policy actually uses OpenPI / Pi-0.5 (the default dsrl_openpi_policy
# does), patch the transformers lib. Skipped otherwise.
bash openpi_transformer_lib_patch.sh
```

Verify after install:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
# Expected: 2.9.0+cu130 True <your GPU name>

python -c "import cyclonedds; print(cyclonedds.__version__)"
# Expected: a 0.10.x version (the bridge targets CycloneDDS v0.10.5 — see messages/igris_c_msgs.py:4)

python -c "import ray; print(ray.__version__)"
# Expected: some 2.x version (whatever ray[default] resolves to)
```

`env_setup.sh` does not install `cyclonedds-python` explicitly. If `import cyclonedds` fails, install it manually in the same venv:

```bash
uv pip install cyclonedds
```

`TODO:` confirm with your team whether `cyclonedds` should be added to `env_setup.sh` — see [09_troubleshooting_igris_c.md § cyclonedds not installed](09_troubleshooting_igris_c.md#cyclonedds-not-installed).

## Sanity-check the IGRIS_C config

Open [`env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json`](../../env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json). The values shown below are the current branch defaults you must check before running.

```json
{
  "HZ": 20,
  "max_delta_deg": 5,
  "policy_update_period": 50,
  "mono_image_resize": { "width": 224, "height": 224 },
  "camera_names": ["head", "left", "right"],
  "proprio_state_dim": 86,
  "action_dim": 17,
  "action_chunk_size": 50,
  "proprio_history_size": 50,
  "num_img_obs": 1,
  "img_obs_every": 1,
  "norm_stats_file_path": "/home/robros/Projects/inference_engine/trainer/experiment_training/igris_c/dataset_stats.pkl",
  "dds": {
    "state_domain_id": 0,
    "camera_domain_id": 1,
    "state_dds_xml": "/home/robros/Projects/inference_engine/env_actor/runtime_settings_configs/robots/igris_c/dds/dds_state.xml",
    "camera_dds_xml": "/home/robros/Projects/inference_engine/env_actor/runtime_settings_configs/robots/igris_c/dds/dds_camera.xml",
    "topics": {
      "lowstate": "rt/lowstate",
      "handstate": "rt/handstate",
      "lowcmd":   "rt/lowcmd",
      "handcmd":  "rt/handcmd",
      "head_camera":  "igris_c/sensor/eyes_stereo",
      "left_camera":  "igris_c/sensor/left_hand",
      "right_camera": "igris_c/sensor/right_hand"
    }
  }
}
```

Two things you will almost certainly need to change before your first run:

1. **`norm_stats_file_path`** (line 16) — an absolute path to a `dataset_stats.pkl` produced by trainer. The branch default points at `/home/robros/Projects/inference_engine/trainer/experiment_training/igris_c/dataset_stats.pkl`, which exists on the author's machine, not yours. **TODO:** point this at your local copy. The pickle must contain keys `observation.state` (with `mean`, `std` of length ≥ 86) and `action` (with `mean`, `std` of length 17). See [06_normalization_igris_c.md](06_normalization_igris_c.md) for the schema.

2. **`state_dds_xml` and `camera_dds_xml`** (lines 21–22) — absolute paths to two Cyclone DDS XML configs that pin DDS to a specific NIC. Both XMLs on this branch hard-code `<NetworkInterface name="enp11s0">` (see [`dds/dds_state.xml:18`](../../env_actor/runtime_settings_configs/robots/igris_c/dds/dds_state.xml#L18) and [`dds/dds_camera.xml:18`](../../env_actor/runtime_settings_configs/robots/igris_c/dds/dds_camera.xml#L18)). **TODO:** if your NIC is not `enp11s0`, change both files. `ip addr show` tells you which interface has the `192.168.10.x/24` address that talks to the NUC; that's the one to use. The XML comment block documents the rule.

Also check [`env_actor/policy/policies/openpi_policy/components/openpi_batched.yaml`](../../env_actor/policy/policies/openpi_policy/components/openpi_batched.yaml):

```yaml
params:
  train_config_name: "pi05_igris"
  ckpt_dir: "/home/robros/Projects/robros_vla_inference_engine/openpi_film/checkpoints/pi05_igris/pi05_igris_b_pnp_v3.3.2/film_15000"
  default_prompt: "Pick up objects on the table with the left hand and place them into the box."
  action_dim: 17
  action_horizon: 50
  num_inference_steps: 10
```

`ckpt_dir` is again an absolute path on the author's machine. **TODO:** replace it with your local checkpoint path before running. The `action_dim: 17` is correct for IGRIS_C (this was the diff vs. `main` — see [02_changes_vs_main.md](02_changes_vs_main.md)); do not change it unless you also change `action_dim` in `inference_runtime_params.json` and rebuild your policy.

## Two ways to run: Ray cluster vs. local

`igris_c/c1` introduces a second, no-Ray entrypoint:

| Entrypoint | When to use | Where |
|---|---|---|
| [`run_inference.py`](../../run_inference.py) | Production / multi-machine — head node hosts policy logic, worker hosts the GPU. Requires a running Ray cluster. | The original entrypoint (unchanged). |
| [`run_inference_local.py`](../../run_inference_local.py) | Single-machine development / debugging. No Ray. Spawns the same RTC two-process layout (control + inference) directly via `multiprocessing.spawn`. | **New on this branch** — see [02_changes_vs_main.md](02_changes_vs_main.md). |

If you are debugging the bridge or running on one box, use `run_inference_local.py`. If you have a head + worker setup and want the production Ray path, use `run_inference.py`.

### Ray path (only if using `run_inference.py`)

Edit [`start_ray.sh`](../../start_ray.sh) to match your network. The values currently in that file:

```bash
HEAD_IP="100.109.184.39"        # current value on this branch — verify against your environment
WORKER_IP="100.112.232.50"      # current value on this branch — verify against your environment
# Hostnames the script switches on:
#   robros-ai1   → starts head
#   robros-5090  → starts worker with --resources='{"inference_pc": 1}'
```

These are the author's tailscale-style 100.x IPs. **TODO:** change `HEAD_IP`, `WORKER_IP`, and the `case "$HOSTNAME"` arms to match your hosts. The custom resource string `'{"inference_pc": 1}'` must stay — both `run_inference.py:54` (`resources={"inference_pc": 1}`) and `run_inference.py:65` rely on it for actor placement.

```bash
# On both head and worker (each with its own venv activated)
bash start_ray.sh

# Verify
ray status
# expected: head + worker nodes, worker advertises 'inference_pc: 1'
```

If `inference_pc` is missing, see [09_troubleshooting_igris_c.md § Resource 'inference_pc' not available](09_troubleshooting_igris_c.md#resource-inference_pc-not-available).

### Local path (only if using `run_inference_local.py`)

No setup beyond the venv activation. The local path uses `multiprocessing.spawn` directly — see [`rtc_local/rtc_local_actor.py:47`](../../env_actor/auto/inference_algorithms/rtc_local/rtc_local_actor.py#L47) (`ctx = mp.get_context("spawn")`) and the top-of-script call to `torch.multiprocessing.set_start_method("spawn")` in [`run_inference_local.py:71`](../../run_inference_local.py#L71).

> **Why `spawn`?** PyTorch CUDA contexts cannot be safely shared across `fork`ed processes — the child inherits a half-initialized CUDA state and `torch.cuda.*` then raises `Cannot re-initialize CUDA in forked subprocess`. The `spawn` start method launches a fresh Python interpreter per child, which avoids this. The RTC/RTC-local actor parent reuses the same context (`mp.get_context("spawn")`) when it forks off the `start_control` and `start_inference` workers.

## Running inference

### IGRIS_C with the local (no-Ray) entrypoint

```bash
# Activate the venv first.
source .venv/bin/activate

# RTC algorithm (default) — control loop + inference loop in two child processes.
python run_inference_local.py \
  --robot igris_c \
  --inference_algorithm rtc \
  --inference_runtime_params_config ./env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json \
  --inference_runtime_topics_config ./env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_topics.json

# Sequential algorithm — single-threaded; easier to debug.
python run_inference_local.py \
  --robot igris_c \
  --inference_algorithm sequential \
  --inference_runtime_params_config ./env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json \
  --inference_runtime_topics_config ./env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_topics.json
```

**Heads-up on the local sequential path.** On this branch, [`sequential_local_actor.SequentialLocalActor.start()`](../../env_actor/auto/inference_algorithms/sequential_local/sequential_local_actor.py#L87) reads a single observation and **writes `head.png`, `left.png`, `right.png` to the working directory, then returns** — the actual inference + publish path is commented out (see lines 134–176). Use it to verify your camera streams; do **not** expect it to control the robot. The Ray-decorated sequential path ([`sequential/sequential_actor.SequentialActor`](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py)) is the complete version.

`TODO:` if you need to drive the robot from the local sequential path, uncomment the inference + publish block in `sequential_local_actor.py:134-176`. Note this also requires un-commenting the `DataNormalizationInterface` constructor at line 85.

### IGRIS_C with the Ray entrypoint

```bash
# On the head node (after start_ray.sh ran successfully on both head and worker):
source .venv/bin/activate

python run_inference.py \
  --robot igris_c \
  --inference_algorithm rtc \
  --inference_runtime_params_config ./env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json \
  --inference_runtime_topics_config ./env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_topics.json
```

Note: the `--inference_runtime_params_config` and `--inference_runtime_topics_config` defaults on this branch still point at the **IGRIS_B** config files (see [`run_inference.py:100-107`](../../run_inference.py#L100)). You **must** pass the IGRIS_C paths explicitly when `--robot igris_c`, or the policy will be fed IGRIS_B-shaped tensors and crash on shape mismatch.

### Bridge-only test (no policy)

Before involving the policy, verify the bridge alone talks to the NUC:

```bash
source .venv/bin/activate
python -m env_actor.robot_io_interface.robots.igris_c.run_bridge_monitor
```

This is implemented in [`run_bridge_monitor.py`](../../env_actor/robot_io_interface/robots/igris_c/run_bridge_monitor.py). It starts the DDS readers, waits for state + cameras, then prints a one-line rolling status (`body_q[0:3]`, `hand_q[0:3]`, `body_tau[0]`, camera frame shapes) until you hit `Ctrl+C`. The hard-coded `CFG` path on line 16 is again an absolute path on the author's machine; **TODO:** change it if you cloned elsewhere.

## First-time success signals

When the bridge is healthy and inference is running, expect to see (in order):

1. `Starting state readers...` followed by repeated `low=OK hand=OK head=OK left=OK right=OK` (from [`controller_bridge.py:333`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L333)) until all five sources are streaming, then `igris_c: state and cameras streaming.` (line 335).
2. `Warming up CUDA kernels...` from the inference process (see [`rtc/actors/inference_loop.py:74`](../../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L74) or `rtc_local`).
3. `Signaling inference ready...` followed by `Starting episode 0...` and then `Control loop started...`.
4. After ~100 control iterations, actions start being published (the `if t > 100` gate in [`rtc/actors/control_loop.py:133`](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L133)). The first 100 steps update SHM but do not move the robot — this is intentional warmup, not a bug.

For RTC, you may also set `BRIDGE_HEALTH_LOG=1` in the environment to get per-topic Hz + staleness prints from the bridge ([`controller_bridge.py:261`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L261)):

```bash
BRIDGE_HEALTH_LOG=1 python run_inference_local.py --robot igris_c ...
# expected output every second:
# [bridge-health] lowstate=200.0Hz(age  0.00s)  handstate=100.0Hz(age  0.01s)  head_cam= 30.0Hz(age  0.03s)  ...
```

If any of those Hz drop to `0.0` mid-run, you have a DDS subscriber dropout — see [09_troubleshooting_igris_c.md](09_troubleshooting_igris_c.md).

## If it crashes immediately

The most common first-run failures, ranked by frequency in our experience:

| Symptom | First place to look |
|---|---|
| `FileNotFoundError: dataset_stats.pkl` | `norm_stats_file_path` in `inference_runtime_params.json` is wrong. See [04 § norm_stats_file_path](04_runtime_configuration_igris_c.md#norm_stats_file_path). |
| `Cannot re-initialize CUDA in forked subprocess` | `torch.multiprocessing.set_start_method("spawn")` was not called early enough; you imported torch+CUDA in the parent and then forked. The entrypoints call it for you — make sure you are not wrapping them. |
| `Resource 'inference_pc' not available` (Ray path only) | Worker did not register the `inference_pc:1` custom resource. Check [`start_ray.sh:43`](../../start_ray.sh#L43) and `ray status`. |
| Timeout in `start_state_readers` with `low=.. hand=.. head=..` all stuck on `..` | DDS NIC pinning is wrong, or the NUC/Jetson isn't publishing. Check `dds/dds_state.xml` and `dds/dds_camera.xml`. |
| `assert a.shape == (17,)` | Policy output dimension does not match IGRIS_C action_dim. Check that you ran with `--robot igris_c` and that `openpi_batched.yaml` has `action_dim: 17`. |

The full table is in [09_troubleshooting_igris_c.md](09_troubleshooting_igris_c.md).

---

← Back to index: [README.md](README.md) · Next → [02_changes_vs_main.md](02_changes_vs_main.md)
