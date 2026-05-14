# Inference Engine

A distributed, real-time inference engine for running vision-language-action (VLA) policies on physical robots. Built on **Ray** for distributed computing, **PyTorch** for GPU-accelerated inference, and **ROS2** for robot communication. Designed for robotics researchers who need to deploy learned manipulation policies on dual-arm platforms at 20 Hz control rates.

## Table of contents

- [Features](#features)
- [Where to go next](#where-to-go-next)
- [Repository layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Ray cluster setup](#ray-cluster-setup)
- [Quickstart](#quickstart)
- [CLI arguments](#cli-arguments)
- [Configuration](#configuration)
- [Architecture overview](#architecture-overview)
- [Extending the system](#extending-the-system)
- [Most important terms](#most-important-terms)
- [Most common errors](#most-common-errors)
- [Acknowledgements](#acknowledgements)

## Features

- **Two inference algorithms** — Sequential (single-threaded, for debugging) and RTC (dual-process with shared memory, for real-time deployment)
- **Action inpainting** — smooth blending between consecutive action chunks using exponential weighting ([Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/pdf/2506.07339))
- **Protocol-based policy interface** — extend with new models without inheriting from a base class
- **Registry system** — configuration-driven policy instantiation via `@POLICY_REGISTRY.register()`
- **Multi-camera observation** — head, left, and right cameras with configurable resolution and history
- **Multi-robot support** — factory pattern isolates robot-specific I/O behind a common interface
- **CUDA-optimized** — `torch.compile`, `torch.autocast` (bfloat16), `cudnn.benchmark`, `torch.inference_mode`
- **Ray-distributed** — head/worker topology with custom resource scheduling (`inference_pc`)

## Where to go next

| I want to... | Read |
|---|---|
| Spend my first day in the repo | [docs/onboarding.md](docs/onboarding.md) |
| Build a mental model (VLA, RTC, action chunks, flow matching) | [docs/concepts.md](docs/concepts.md) |
| Understand the architecture | [docs/architecture.md](docs/architecture.md) |
| Look up a function or shape | [docs/api.md](docs/api.md) |
| Run a first inference end-to-end | [docs/walkthroughs/01_first_run.md](docs/walkthroughs/01_first_run.md) |
| Trace one inference step | [docs/walkthroughs/02_trace_one_step.md](docs/walkthroughs/02_trace_one_step.md) |
| Add a new policy | [docs/walkthroughs/03_add_a_new_policy.md](docs/walkthroughs/03_add_a_new_policy.md) |
| Add a new robot | [docs/walkthroughs/04_add_a_new_robot.md](docs/walkthroughs/04_add_a_new_robot.md) |
| Understand the RTC shared-memory design | [docs/rtc_shared_memory.md](docs/rtc_shared_memory.md) |
| Modify configuration safely | [docs/configuration_cookbook.md](docs/configuration_cookbook.md) |
| Debug a crash | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Get unstuck on a common question | [docs/faq.md](docs/faq.md) |
| Look up a term | [docs/glossary.md](docs/glossary.md) |
| Browse all docs | [docs/README.md](docs/README.md) |
| **You are on the `igris_c/c1` branch** — get the IGRIS_C robot running | [docs/igris_c_c1/README.md](docs/igris_c_c1/README.md) |

## Repository layout

```
inference_engine/
├── run_inference.py                 # Main entrypoint (supports RTC + Sequential)
├── uv_setup.sh                      # Install uv package manager + create venv
├── env_setup.sh                     # Install all Python dependencies
├── openpi_transformer_lib_patch.sh  # Patch transformers lib for OpenPI compatibility
├── start_ray.sh                     # Start Ray cluster (edit hostnames first)
│
├── env_actor/                       # Core inference runtime
│   ├── policy/                      # Policy protocol, registry, loader
│   ├── auto/                        # Inference algorithms (RTC, Sequential)
│   ├── robot_io_interface/          # Robot hardware abstraction (ROS2)
│   ├── runtime_settings_configs/    # Per-robot runtime configuration
│   ├── nom_stats_manager/           # Data normalization/denormalization
│   └── inference_engine_utils/      # Shared utilities (action inpainting)
│
├── trainer/                         # Training framework (git submodule, pinned)
└── docs/                            # Architecture, API, walkthroughs, troubleshooting
```

Each first-level directory is hyperlinked: [run_inference.py](run_inference.py) · [env_actor/](env_actor/README.md) · [env_actor/policy/](env_actor/policy/README.md) · [env_actor/auto/](env_actor/auto/README.md) · [env_actor/robot_io_interface/](env_actor/robot_io_interface/README.md) · [env_actor/runtime_settings_configs/](env_actor/runtime_settings_configs/README.md) · [env_actor/nom_stats_manager/](env_actor/nom_stats_manager/README.md) · [env_actor/inference_engine_utils/](env_actor/inference_engine_utils/README.md) · [docs/](docs/README.md).

## Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.12+ (the `uv` setup pins 3.12.3) |
| GPU | NVIDIA with CUDA 13.0 support |
| ROS2 | Required for robot communication (IGRIS_B uses `rclpy`, `sensor_msgs`, `geometry_msgs`) |
| Cameras | V4L2-compatible USB cameras (for IGRIS_B: head, left, right) |
| OS | Linux (tested on Ubuntu) |

## Installation

> **No SSH key on this machine?** The default submodule URL uses SSH (`git@github.com:...`). Before running `git submodule update --init --recursive`, switch it to HTTPS:
> ```bash
> git submodule set-url trainer https://github.com/KyunHwan/trainer.git
> # And for the nested submodule (after the outer one is checked out):
> cd trainer && git submodule set-url policy_constructor https://github.com/KyunHwan/policy_constructor.git && cd ..
> ```
> See [docs/troubleshooting.md](docs/troubleshooting.md) for more.

```bash
# 1. Clone with submodules
git clone --recurse-submodules <repo-url>
cd inference_engine

# 2. Install uv and create virtual environment (Python 3.12.3)
bash uv_setup.sh

# 3. Activate the environment
source .venv/bin/activate

# 4. Install Python dependencies (PyTorch 2.9.0 + CUDA 13.0, Ray, Transformers, etc.)
bash env_setup.sh

# 5. (Only if your policy uses OpenPI/Pi0.5) Patch the transformers library
bash openpi_transformer_lib_patch.sh

# 6. If you cloned without --recurse-submodules:
git submodule update --init --recursive
```

### What `env_setup.sh` installs

Core: `torch==2.9.0`, `torchvision==0.24.0` (CUDA 13.0), `numpy`, `einops`, `timm`, `opencv-python`

ML: `transformers==4.53.2`, `flow_matching`, `schedulefree`, `geomloss`, `flax`, `jaxtyping==0.2.34`, `beartype`, `augmax`, `chex`, `sentencepiece`, `ml_collections`

Distributed: `ray[default]`, `cloudpickle`, `gcsfs`

Robot learning: `lerobot` (no deps), `datasets`, `accelerate`, `av`, `tyro`

It also installs Depth Anything v3 from `trainer/policy_constructor/.../depth_anything_3` as an editable package.

## Ray cluster setup

The inference engine uses Ray for distributed actor management. Edit [start_ray.sh](start_ray.sh) to match your network before running:

```bash
# start_ray.sh uses hostname-based switching:
#   Head node  → ray start --head --port=6379
#   Worker node → ray start --address=<HEAD_IP>:6379 --resources='{"inference_pc": 1}'
#
# Edit HEAD_IP and the hostname cases to match your setup.
bash start_ray.sh
```

The `inference_pc` custom resource ensures inference actors are scheduled on the GPU-equipped worker node. A single-machine smoke-test variant is described in [docs/walkthroughs/01_first_run.md](docs/walkthroughs/01_first_run.md).

## Quickstart

Once Ray is running and dependencies are installed:

```bash
# Run with RTC algorithm (default) on IGRIS_B
python run_inference.py --robot igris_b

# Run with Sequential algorithm (easier to debug)
python run_inference.py --robot igris_b --inference_algorithm sequential
```

Press `Ctrl+C` to stop. Then `ray stop` on each machine to tear down the cluster.

## CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--robot` | `igris_b` | Robot platform: `igris_b` or `igris_c` |
| `--inference_algorithm` | `rtc` | Algorithm: `rtc` (real-time, dual-process) or `sequential` (single-threaded) |
| `--policy_yaml_path`, `-P` | `./env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.yaml` | Path to policy YAML config |
| `--inference_runtime_params_config` | `./env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json` | Path to runtime parameters JSON |
| `--inference_runtime_topics_config` | `./env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_topics.json` | Path to ROS2 topic mapping JSON |

The full reference is in [docs/api.md](docs/api.md).

## Configuration

The system uses three configuration layers — full recipes are in [docs/configuration_cookbook.md](docs/configuration_cookbook.md).

### 1. Policy YAML (e.g. [openpi_policy.yaml](env_actor/policy/policies/openpi_policy/openpi_policy.yaml))

Defines which model components to load and which policy class to use:

```yaml
model:
  component_config_paths:
    openpi_model: components/openpi_batched.yaml

policy:
  type: openpi_policy
```

Component YAML files (e.g., [components/openpi_batched.yaml](env_actor/policy/policies/openpi_policy/components/openpi_batched.yaml)) define model architecture, checkpoint paths, and inference parameters:

```yaml
params:
  train_config_name: "pi05_igris"
  ckpt_dir: "/path/to/checkpoints"
  default_prompt: "Pick up objects on the table with the left hand and place them into the box."
  action_dim: 24
  action_horizon: 50
  num_inference_steps: 10
```

### 2. Runtime parameters JSON ([inference_runtime_params.json](env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json))

Controls inference timing, dimensions, and camera setup. Values shown are the current IGRIS_B defaults:

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
  "norm_stats_file_path": "/abs/path/to/dataset_stats.pkl"
}
```

### 3. Runtime topics JSON ([inference_runtime_topics.json](env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_topics.json))

Maps ROS2 topics to observation keys with field slicing. See the file inline for the full IGRIS_B layout.

## Architecture overview

```
┌─────────────────────────────────────────────────────────────┐
│  ENTRYPOINT                                                 │
│  run_inference.py → Ray init → Actor creation               │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  INFERENCE ALGORITHM LAYER (robot-agnostic)                 │
│                                                             │
│  ┌─────────────────────┐  ┌──────────────────────────────┐  │
│  │ SequentialActor     │  │ RTCActor                     │  │
│  │ Single-threaded     │  │ ControlLoop + InferenceLoop  │  │
│  │ Sync predict()      │  │ Shared memory IPC            │  │
│  └─────────────────────┘  └──────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  POLICY LAYER                                               │
│  Policy (Protocol) ← build_policy(yaml) ← POLICY_REGISTRY   │
│  predict() / guided_inference() / warmup() / freeze()       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  ROBOT I/O LAYER (hardware-specific bridges)                │
│  ControllerInterface → controller_bridge (per robot)        │
│  DataNormalizationInterface → normalization_bridge          │
│  read_state() → normalize → policy → denormalize → publish  │
└─────────────────────────────────────────────────────────────┘
```

A more detailed diagram (with the RTC two-process topology and the shared-memory regions) is in [docs/architecture.md](docs/architecture.md).

## Extending the system

| Task | Where to start |
|---|---|
| Add a new policy | [docs/walkthroughs/03_add_a_new_policy.md](docs/walkthroughs/03_add_a_new_policy.md) |
| Add a new robot | [docs/walkthroughs/04_add_a_new_robot.md](docs/walkthroughs/04_add_a_new_robot.md) |
| Add a new inference algorithm | [docs/development.md](docs/development.md) |

## Most important terms

| Term | One-line |
|---|---|
| **VLA** | Vision–Language–Action: a policy taking camera images + a sentence prompt + robot state, outputting motor commands |
| **Action chunk** | A fixed-length trajectory (default 50 steps) of future actions predicted in one forward pass |
| **RTC** | Real-Time Control — dual-process inference (control + inference) communicating via shared memory |
| **Action inpainting** | Exponential-weighted blending between the tail of the previous action chunk and a fresh prediction |
| **inference_pc** | Ray custom resource name that pins inference actors to the GPU worker |

The full glossary is at [docs/glossary.md](docs/glossary.md).

## Most common errors

| Error fragment | Likely cause | Where to look |
|---|---|---|
| `Cannot re-initialize CUDA in forked subprocess` | Missing `set_start_method("spawn")` before CUDA | [docs/troubleshooting.md § CUDA](docs/troubleshooting.md#cuda) |
| `Resource 'inference_pc' not available` | Worker did not register `inference_pc:1` | [docs/troubleshooting.md § Ray](docs/troubleshooting.md#ray) |
| `rclpy` import errors | ROS2 not sourced | [docs/troubleshooting.md § ROS2](docs/troubleshooting.md#ros2) |

The full table is at [docs/troubleshooting.md](docs/troubleshooting.md).

## Acknowledgements

- Action inpainting technique from [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/pdf/2506.07339)
- Training framework: [trainer](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/README.md) submodule, pinned at `3ca051a`
- Model construction: [policy_constructor](https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/README.md) (nested submodule inside `trainer/`), pinned at `00663cc`
