# Inference Engine

A distributed, real-time inference engine for running vision-language-action (VLA) policies on physical robots. Built on **Ray** for distributed computing, **PyTorch** for GPU-accelerated inference, and **ROS2** for robot communication. Designed for robotics researchers who need to deploy learned manipulation policies on dual-arm platforms at 20 Hz control rates.

## Features

- **Two inference algorithms** — Sequential (single-threaded, for debugging) and RTC (dual-process with shared memory, for real-time deployment)
- **Action inpainting** — smooth blending between consecutive action chunks using exponential weighting ([Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/pdf/2506.07339))
- **Protocol-based policy interface** — extend with new models without inheriting from a base class
- **Registry system** — configuration-driven policy instantiation via `@POLICY_REGISTRY.register()`
- **Multi-camera observation** — head, left, and right cameras with configurable resolution and history
- **Multi-robot support** — factory pattern isolates robot-specific I/O behind a common interface
- **CUDA-optimized** — `torch.compile`, `torch.autocast` (bfloat16), `cudnn.benchmark`, `torch.inference_mode`
- **Ray-distributed** — head/worker topology with custom resource scheduling (`inference_pc`)

## Repository Layout

```
inference_engine/
├── run_inference.py                 # Main entrypoint (supports RTC + Sequential)
├── run_inference_openpi.py          # Simplified entrypoint (RTC only)
├── uv_setup.sh                     # Install uv package manager + create venv
├── env_setup.sh                    # Install all Python dependencies
├── start_ray.sh                    # Start Ray cluster (edit hostnames first)
│
├── env_actor/                      # Core inference runtime
│   ├── policy/                     # Policy protocol, registry, loader
│   ├── auto/                       # Inference algorithms (RTC, Sequential)
│   ├── robot_io_interface/         # Robot hardware abstraction (ROS2)
│   ├── runtime_settings_configs/   # Per-robot runtime configuration
│   ├── nom_stats_manager/          # Data normalization/denormalization
│   └── inference_engine_utils/     # Shared utilities (action inpainting)
│
├── trainer/                        # Training framework (git submodule)
└── docs/                           # Architecture, API, and development guides
```

## Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.12+ |
| GPU | NVIDIA with CUDA 13.0 support |
| ROS2 | Required for robot communication (IGRIS_B uses `rclpy`, `sensor_msgs`, `geometry_msgs`) |
| Cameras | V4L2-compatible USB cameras (for IGRIS_B: head, left, right) |
| OS | Linux (tested on Ubuntu) |

## Installation

```bash
# 1. Clone with submodules
git clone --recurse-submodules <repo-url>
cd inference_engine

# 2. Install uv and create virtual environment
bash uv_setup.sh

# 3. Activate the environment
source .venv/bin/activate

# 4. Install Python dependencies (PyTorch 2.9.0 + CUDA 13.0, Ray, Transformers, etc.)
bash env_setup.sh

# 5. If you cloned without --recurse-submodules:
git submodule update --init --recursive
```

### What `env_setup.sh` installs

Core: `torch==2.9.0`, `torchvision==0.24.0` (CUDA 13.0), `numpy`, `einops`, `timm`

ML: `transformers==4.53.2`, `flow_matching`, `schedulefree`, `geomloss`, `flax`, `jaxtyping==0.2.34`, `beartype`

Distributed: `ray[default]`, `cloudpickle`

Robot learning: `lerobot` (no deps), `datasets`, `accelerate`

## Ray Cluster Setup

The inference engine uses Ray for distributed actor management. Edit `start_ray.sh` to match your network before running:

```bash
# start_ray.sh uses hostname-based switching:
#   Head node  → ray start --head --port=6379
#   Worker node → ray start --address=<HEAD_IP>:6379 --resources='{"inference_pc": 1}'
#
# Edit HEAD_IP and the hostname cases to match your setup.
bash start_ray.sh
```

The `inference_pc` custom resource ensures inference actors are scheduled on the GPU-equipped worker node.

## Quickstart

Once Ray is running and dependencies are installed:

```bash
# Run with RTC algorithm (default) on IGRIS_B
python run_inference.py --robot igris_b

# Run with Sequential algorithm
python run_inference.py --robot igris_b --inference_algorithm sequential
```

### CLI Arguments (`run_inference.py`)

| Argument | Default | Description |
|---|---|---|
| `--robot` | *(required)* | Robot platform: `igris_b` or `igris_c` |
| `--inference_algorithm` | `rtc` | Algorithm: `rtc` (real-time, dual-process) or `sequential` (single-threaded) |
| `--policy_yaml_path`, `-P` | `./env_actor/policy/policies/openpi_policy/openpi_policy.yaml` | Path to policy YAML config |
| `--inference_runtime_params_config` | `./env_actor/runtime_settings_configs/igris_b/inference_runtime_params.json` | Path to runtime parameters JSON |
| `--inference_runtime_topics_config` | `./env_actor/runtime_settings_configs/igris_b/inference_runtime_topics.json` | Path to ROS2 topic mapping JSON |

There is also `run_inference_openpi.py`, a simplified entrypoint that always uses the RTC algorithm (same arguments minus `--inference_algorithm`).

## Configuration

The system uses three configuration layers:

### 1. Policy YAML (`openpi_policy.yaml`)

Defines which model components to load and which policy class to use:

```yaml
model:
  component_config_paths:
    openpi_model: components/openpi_batched.yaml

policy:
  type: openpi_policy
```

Component YAML files (e.g., `components/openpi_batched.yaml`) define model architecture, checkpoint paths, and inference parameters:

```yaml
params:
  train_config_name: "pi05_igris"
  ckpt_dir: "/path/to/checkpoints"
  default_prompt: "Pick up objects on the table with the left hand and place them into the box."
  action_dim: 24
  action_horizon: 50
  num_inference_steps: 10
```

### 2. Runtime Parameters JSON (`inference_runtime_params.json`)

Controls inference timing, dimensions, and camera setup:

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
  "proprio_history_size": 1,
  "num_img_obs": 1,
  "img_obs_every": 1,
  "norm_stats_file_path": ""
}
```

### 3. Runtime Topics JSON (`inference_runtime_topics.json`)

Maps ROS2 topics to observation keys with field slicing:

```json
{
  "robot_id": "packy",
  "HZ": 20,
  "topics": {
    "joints": {
      "topic": "/igris_b/packy/joint_states",
      "msg_type": "JointState",
      "fields": {
        "/observation/joint_pos/left": {"slice": [6, 12], "attr": "position"},
        "/observation/joint_pos/right": {"slice": [0, 6], "attr": "position"}
      }
    }
  }
}
```

## Architecture Overview

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
│  │ SequentialActor      │  │ RTCActor                     │  │
│  │ Single-threaded      │  │ ControlLoop + InferenceLoop  │  │
│  │ Sync predict()       │  │ Shared memory IPC            │  │
│  └─────────────────────┘  └──────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  POLICY LAYER                                               │
│  Policy (Protocol) ← build_policy(yaml) ← POLICY_REGISTRY  │
│  predict() / guided_inference() / warmup() / freeze()       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  ROBOT I/O LAYER (hardware-specific bridges)                │
│  ControllerInterface → controller_bridge (per robot)        │
│  DataNormalizationInterface → normalization_bridge           │
│  read_state() → normalize → policy → denormalize → publish  │
└─────────────────────────────────────────────────────────────┘
```

### Execution Flow

1. `run_inference.py` parses CLI args, initializes Ray
2. Creates a Ray remote actor (`RTCActor` or `SequentialActor`)
3. Actor loads policy via `build_policy()` (YAML → model factory → checkpoint loading)
4. Actor initializes robot controller (ROS2 node, cameras, joint subscribers)
5. Control loop runs at configured HZ (default 20):
   - Read robot state (proprioception + camera images)
   - Normalize observations
   - Run policy inference (every `policy_update_period` steps)
   - Denormalize actions
   - Publish actions with slew-rate limiting

## Extending the System

### Adding a New Policy

1. Create a directory under `env_actor/policy/policies/your_policy/`
2. Implement the `Policy` protocol:

```python
# env_actor/policy/policies/your_policy/your_policy.py
from env_actor.policy.registry import POLICY_REGISTRY

@POLICY_REGISTRY.register("your_policy")
class YourPolicy:
    def __init__(self, components, **kwargs):
        self.model = components["main"]

    def predict(self, input_data, data_normalization_interface):
        # input_data["proprio"]: (history, state_dim) float32
        # input_data["head"/"left"/"right"]: (num_img_obs, 3, H, W) uint8
        # Return: np.ndarray of shape (action_chunk_size, action_dim)
        ...

    def guided_inference(self, input_data, data_normalization_interface,
                         min_num_actions_executed, action_chunk_size):
        # Same as predict but with action inpainting for RTC
        # input_data also contains "est_delay" and "prev_action"
        ...

    def warmup(self):
        # Optional: torch.compile warmup for CUDA kernel selection
        ...

    def freeze_all_model_params(self):
        # Freeze parameters for inference
        ...
```

3. Create a policy YAML config:

```yaml
# env_actor/policy/policies/your_policy/your_policy.yaml
model:
  component_config_paths:
    main: components/your_model.yaml

policy:
  type: your_policy
```

### Adding a New Robot

See the [IGRIS_C interface documentation](env_actor/robot_io_interface/robots/igris_c/README.md) for a detailed implementation checklist. In summary:

1. **Runtime configs** in `env_actor/runtime_settings_configs/robots/<robot>/`:
   - `init_params.py` — initial joint positions and state keys
   - `inference_runtime_params.json` — timing, dimensions, cameras
   - `inference_runtime_topics.json` — ROS2 topic mapping
   - `inference_runtime_params.py` — `RuntimeParams` class

2. **Controller bridge** in `env_actor/robot_io_interface/robots/<robot>/controller_bridge.py`:
   - Implement `read_state()`, `publish_action()`, `start_state_readers()`, `init_robot_position()`, `shutdown()`

3. **Data manager bridges** in both algorithm directories:
   - `env_actor/auto/inference_algorithms/sequential/data_manager/robots/<robot>/`
   - `env_actor/auto/inference_algorithms/rtc/data_manager/robots/<robot>/`

4. **Normalization bridge** in `env_actor/nom_stats_manager/robots/<robot>/`

5. **Update factory** `if/elif` blocks in:
   - `env_actor/robot_io_interface/controller_interface.py`
   - `env_actor/auto/inference_algorithms/sequential/data_manager/data_manager_interface.py`
   - `env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py`
   - `env_actor/nom_stats_manager/data_normalization_interface.py`

## Glossary

| Term | Definition |
|---|---|
| **RTC** | Real-Time Control — dual-process inference algorithm where control and inference run in parallel via shared memory |
| **Action chunk** | A fixed-length trajectory of future actions predicted by the policy in a single forward pass (default: 50 steps) |
| **Action inpainting** | Technique that blends the tail of the previous action chunk with a new prediction to ensure smooth transitions |
| **Guided inference** | Policy inference mode used by RTC that incorporates the previous action chunk via inpainting weights |
| **Policy update period** | Number of control steps between policy inference calls (default: 50 at 20 Hz = 2.5 seconds) |
| **Slew-rate limiting** | Safety mechanism that clamps the maximum angular change per control step (`max_delta_deg`) |
| **Proprio** | Proprioceptive state — joint positions, hand positions, and optionally joint currents/torques |

## Troubleshooting

**`RuntimeError: Cannot re-initialize CUDA in forked subprocess`**
The entrypoints set `torch.multiprocessing.set_start_method("spawn")`. This must happen before any CUDA calls. If you see this error, ensure no code imports CUDA tensors before the multiprocessing setup.

**`ConnectionError: Ray is not initialized`**
Run `bash start_ray.sh` on all machines before launching inference. The script expects `ray.init(address="auto")` to find an existing cluster.

**`RuntimeError: Resource 'inference_pc' not available`**
The worker node must register the `inference_pc` resource. Check that `start_ray.sh` ran successfully on the GPU worker with `--resources='{"inference_pc": 1}'`.

**`FileNotFoundError: checkpoint .pt not found`**
`build_policy()` loads checkpoints from the path specified in the component YAML's `ckpt_dir` parameter. Verify the path exists and contains the expected `.pt` files.

**`rclpy` import errors**
ROS2 must be sourced before running (`source /opt/ros/<distro>/setup.bash`). The IGRIS_B controller bridge depends on `rclpy`, `sensor_msgs`, `std_msgs`, and `geometry_msgs`.

**Camera not opening / black frames**
IGRIS_B expects V4L2 USB cameras at specific device paths (`/dev/head_camera1`, `/dev/left_camera2`, `/dev/right_camera1`). Check `udevadm` rules or update the controller bridge.

## Acknowledgements

- Action inpainting technique from [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/pdf/2506.07339)
- Training framework provided by the [trainer](trainer/) submodule
- Policy model construction via the `policy_constructor` library in `trainer/policy_constructor/`
