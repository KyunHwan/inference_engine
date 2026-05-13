# Glossary

Every domain-specific term used in this repo, in alphabetical order. When a term is implemented somewhere concrete, the entry links to the code or to the deeper doc.

## Table of contents

- [A](#a)
- [B](#b)
- [C](#c)
- [D](#d)
- [E](#e)
- [F](#f)
- [G](#g)
- [H](#h)
- [I](#i)
- [M](#m)
- [N](#n)
- [O](#o)
- [P](#p)
- [R](#r)
- [S](#s)
- [T](#t)
- [V](#v)
- [W](#w)

## A

### Action chunk
A contiguous sequence of future actions predicted in one forward pass of the policy. For IGRIS_B the default is 50 timesteps × 24 action dims. See [`action_chunk_size`](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json) and [concepts.md](concepts.md#action-chunks).

### Action dim
Dimensionality of one action vector. IGRIS_B uses `action_dim: 24` — 6 left-arm + 6 right-arm joints + 6 left-hand + 6 right-hand fingers. See [controller_bridge.py:85-112](../env_actor/robot_io_interface/robots/igris_b/controller_bridge.py#L85-L112) for the slice layout.

### Action horizon
Synonym for `action_chunk_size` in the OpenPI component YAMLs ([openpi_batched.yaml](../env_actor/policy/policies/openpi_policy/components/openpi_batched.yaml)). Both terms denote the number of future steps predicted in one call.

### Action inpainting
A technique that blends the tail of the previous action chunk into a freshly predicted chunk to avoid hard jumps at chunk boundaries. Implemented in [action_inpainting.py](../env_actor/inference_engine_utils/action_inpainting.py) and used inside [`OpenPiPolicy.guided_inference`](../env_actor/policy/policies/openpi_policy/openpi_policy.py) and [`DsrlOpenpiPolicy.guided_inference`](../env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.py). The reference paper is [arXiv:2506.07339](https://arxiv.org/pdf/2506.07339). See [concepts.md](concepts.md#action-inpainting).

### Autocast (`torch.autocast`)
Context manager that runs ops in a lower-precision dtype (bfloat16 here) while keeping accumulators in float32. Used in the RTC inference loop: [inference_loop.py:119](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L119).

## B

### bfloat16
16-bit floating-point format with float32's exponent range but only ~3 decimal digits of precision. Stable for inference on Ada/Hopper-class NVIDIA GPUs. Used by `autocast` in the RTC inference loop.

### build_policy
Function in [loader.py](../env_actor/policy/utils/loader.py) that loads a YAML, builds component `nn.Module`s via `PolicyConstructorModelFactory`, optionally loads a checkpoint, and instantiates a `Policy` from the registry. The single entry point used by both actors.

### Bridge (pattern)
A small adapter class that satisfies a public interface for one specific robot (or algorithm-robot combination). Files named `*_bridge.py` (e.g. `controller_bridge.py`, `shm_manager_bridge.py`). The matching interface class (`controller_interface.py`, `shm_manager_interface.py`) picks the bridge by robot name and forwards every call.

## C

### Checkpoint
Saved PyTorch state dict, one `.pt` file per component. `build_policy` reads `checkpoint_path/<component>.pt` if `checkpoint_path` is set in the policy YAML. The DSRL+OpenPI policy and the OpenPI policy handle this differently — see [walkthroughs/03_add_a_new_policy.md](walkthroughs/03_add_a_new_policy.md).

### Control loop
The 20 Hz loop that reads robot state, advances buffers, asks the policy for the current action, and publishes it. In RTC it lives in [control_loop.py](../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py); in Sequential it is inline in [`SequentialActor.start`](../env_actor/auto/inference_algorithms/sequential/sequential_actor.py).

### Controller interface
[ControllerInterface](../env_actor/robot_io_interface/controller_interface.py) — the public façade for robot I/O. Each method delegates to a robot-specific `ControllerBridge`.

### cuDNN benchmark
`torch.backends.cudnn.benchmark = True` tells cuDNN to time several algorithm variants on the first forward pass and cache the fastest one for that input shape. Set in [inference_loop.py:63](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L63).

## D

### Data manager
A per-(algorithm × robot) class that owns observation history, action buffering, and any pre/post processing the actors do *between* the controller and the policy. Two flavors exist: [sequential data_manager](../env_actor/auto/inference_algorithms/sequential/data_manager/) (in-process numpy buffers) and [RTC data_manager](../env_actor/auto/inference_algorithms/rtc/data_manager/) (shared-memory arrays).

### Data normalization interface
[DataNormalizationInterface](../env_actor/nom_stats_manager/data_normalization_interface.py) — applies z-score normalization to proprioception and scales images to [0, 1] before inference; reverses the proprio transform on outputs.

### Denormalization
Inverse z-score: `action = normalized_action * std + mean`. In this codebase each policy decides where this happens — `OpenPiPolicy` lets OpenPI's internal pipeline handle it; `DsrlOpenpiPolicy` calls `normalize_state` itself on the proprio input.

### DSRL
Diffusion Steering for Reinforcement Learning. The [`dsrl_openpi_policy`](../env_actor/policy/policies/dsrl_openpi_policy/) composes a small noise-actor on top of a frozen OpenPI/Pi0.5 backbone so the policy can be fine-tuned with RL on top of a large pretrained model. The DSRL components are loaded from a separate `checkpoint_path`; OpenPI loads its own weights from `ckpt_dir` inside its component YAML.

## E

### est_delay
Estimated number of control steps a single inference call took. Tracked by [`MaxDeque`](../env_actor/auto/inference_algorithms/rtc/data_manager/utils/max_deque.py) inside the RTC shared-memory manager. Used as `delay_steps` in `compute_guided_prefix_weights`.

## F

### Flow matching
A generative-model training objective that learns a velocity field along a noise→data probability path. OpenPI's action decoder is a flow-matching transformer. `num_inference_steps` is the number of ODE integration steps at inference time (default 10). See [concepts.md](concepts.md#flow-matching).

### freeze_all_model_params
Method on the `Policy` protocol that disables `requires_grad` on every parameter so no autograd memory is allocated at inference time.

## G

### GraphModel
The `nn.Module` produced by [`policy_constructor.build_model`](https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/README.md). It owns a `graph_modules` `ModuleDict` and runs a fixed DAG over them on `forward()`. The `OpenPiPolicy` unwraps the actual `OpenPiBatchedWrapper` out of this graph in `_resolve_wrapper`.

### Guided inference
The inference call used by RTC. It returns an `action_chunk_size × action_dim` array that is the blend of the previous chunk's tail (weighted by `compute_guided_prefix_weights`) and the new prediction.

## H

### Head/left/right camera
The three USB cameras used by IGRIS_B. Each is wrapped in [`RBRSCamera`](../env_actor/robot_io_interface/robots/igris_b/utils/camera_utils.py) and resized to `mono_image_resize` (default 320×240). Devices:

| Name | V4L2 device |
|---|---|
| `head` | `/dev/head_camera1` |
| `right` | `/dev/right_camera1` |
| `left` | `/dev/left_camera2` |

### HZ
Control loop frequency in Hertz. Default `20`. The control period is `DT = 1/HZ`. Defined in [inference_runtime_params.json](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json).

## I

### IGRIS_B
The currently-supported robot. Dual-arm, 24-DOF total (6 arm + 6 hand per side). All bridges under `robots/igris_b/` implement against it.

### IGRIS_C
A future robot. Interface stubs only — every method raises `NotImplementedError`. See [walkthroughs/04_add_a_new_robot.md](walkthroughs/04_add_a_new_robot.md).

### inference_pc resource
A Ray custom resource string (`"inference_pc": 1`) declared in [start_ray.sh](../start_ray.sh) on the GPU worker. Inference actors request `resources={"inference_pc": 1}` so Ray schedules them on the right machine. Documented in [concepts.md](concepts.md#ray-actors).

### inference_mode (`torch.inference_mode`)
A stronger no-grad context manager than `torch.no_grad`: also turns off autograd version counters and tensor-view tracking. Used in the RTC inference loop ([inference_loop.py:119](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L119)).

## M

### MaxDeque
Sliding-window max tracker in [max_deque.py](../env_actor/auto/inference_algorithms/rtc/data_manager/utils/max_deque.py). Keeps the last 5 inference latencies and returns the maximum; used as a conservative `est_delay`.

### max_delta
Maximum per-step joint change in radians. Computed in [`RuntimeParams.__init__`](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.py) from `max_delta_deg` in the JSON. Enforced inside `publish_action` via `np.clip(raw_joint - prev_joint, -max_delta, +max_delta)`.

### max_delta_deg
JSON-side name for `max_delta` (in degrees, before the deg→rad conversion). Default `5`.

### min_num_actions_executed
Hardcoded `35` in [inference_loop.py:17](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L17). The inference loop blocks until the control loop has executed at least this many actions since the last inference; this prevents inference from re-firing too eagerly.

## N

### Normalization stats
A pickle file at the path given by `norm_stats_file_path`. The IGRIS_B normalizer reads `observation.state.{mean,std}`, `observation.current.{mean,std}`, and `action.{mean,std}`; missing keys will raise.

### num_inference_steps
Number of ODE steps used by the flow-matching action decoder at inference time. Default `10` in [openpi_batched.yaml](../env_actor/policy/policies/openpi_policy/components/openpi_batched.yaml). Lower = faster, less accurate.

### num_img_obs
Number of past camera frames per camera fed to the policy. Default `1` (latest only).

## O

### OpenPI / Pi0.5
A pretrained vision-language-action transformer ("Pi-zero-five"). This repo wraps an existing OpenPI checkpoint via `OpenPiBatchedWrapper` (built by `policy_constructor`'s `openpi_batched` block) and exposes it through [`OpenPiPolicy`](../env_actor/policy/policies/openpi_policy/openpi_policy.py).

## P

### policy_update_period
Number of control steps between policy inference calls in the Sequential algorithm. `50` at 20 Hz = one inference every 2.5 s. RTC ignores this and re-fires based on `min_num_actions_executed`.

### POLICY_REGISTRY
The global `Registry` instance in [registry/__init__.py](../env_actor/policy/registry/__init__.py). Classes decorated with `@POLICY_REGISTRY.register("name")` are discoverable by `build_policy` by their string `policy.type` key.

### predict
The `Policy` protocol method used by Sequential. Single-shot inference; returns the new action chunk and discards whatever was queued.

### Prompt
The natural-language instruction passed alongside images and proprio to a VLA. Each component YAML carries a `default_prompt` field that the wrapper uses if the observation dict has no `"prompt"` key.

### Proprio / proprioception
The robot's own sense of where it is — joint positions, hand positions, optionally currents. Always a flat float32 vector of length `proprio_state_dim` (24 on IGRIS_B).

### proprio_history_size
Number of past proprio frames fed to the policy. IGRIS_B JSON sets `50`. Note that some policies use only the most recent frame (`obs["proprio"][0]`) regardless.

## R

### Ray actor
A Python class decorated with `@ray.remote`. Ray creates the instance in a separate Python process and forwards method calls over RPC. See [`RTCActor`](../env_actor/auto/inference_algorithms/rtc/rtc_actor.py) and [`SequentialActor`](../env_actor/auto/inference_algorithms/sequential/sequential_actor.py).

### rclpy
Python bindings for ROS2. Used in [controller_bridge.py](../env_actor/robot_io_interface/robots/igris_b/controller_bridge.py) to publish joint/finger targets and subscribe to state topics.

### Registry
String → class lookup with a `@register` decorator. Implementation: [registry/core.py](../env_actor/policy/registry/core.py) (local copy, used by trainer side) and [`trainer/trainer/registry/core.Registry`](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/trainer/registry/core.py) (the class actually instantiated by `POLICY_REGISTRY`).

### RTC
Real-Time Control. The dual-process inference algorithm in [`rtc/`](../env_actor/auto/inference_algorithms/rtc/) where a control loop (20 Hz) and an inference loop run in separate processes, communicating through shared memory. See [rtc_shared_memory.md](rtc_shared_memory.md).

### RuntimeParams
Per-robot configuration class in [inference_runtime_params.py](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.py). Loads the JSON, converts degrees to radians for `max_delta`, exposes everything as properties, and lazily loads the normalization-stats pickle.

## S

### Sequential algorithm
The simpler inference algorithm — a single Ray actor running one synchronous control loop that calls `policy.predict()` every `policy_update_period` steps. See [`SequentialActor`](../env_actor/auto/inference_algorithms/sequential/sequential_actor.py).

### Shared memory
`multiprocessing.shared_memory.SharedMemory` — an OS-level RAM region two processes can `mmap`. RTC uses five regions (`proprio`, `head`, `left`, `right`, `action`) so the control and inference processes exchange data without serialization. Created by the parent in [`RTCActor.start`](../env_actor/auto/inference_algorithms/rtc/rtc_actor.py); attached by the children via [`attach_shared_ndarray`](../env_actor/auto/inference_algorithms/rtc/data_manager/utils/shared_memory_utils.py). See [rtc_shared_memory.md](rtc_shared_memory.md).

### Slew-rate limiting
Per-joint clip of `target - current` to `±max_delta` rad. Lives in [`ControllerBridge.publish_action`](../env_actor/robot_io_interface/robots/igris_b/controller_bridge.py).

### Spawn vs fork
Two Python multiprocessing start methods. `spawn` creates a fresh interpreter and re-imports modules; `fork` clones the current process. CUDA cannot be re-initialized in a forked process, so this codebase calls `torch.multiprocessing.set_start_method("spawn")` at the top of [run_inference.py](../run_inference.py) and uses `mp.get_context("spawn")` in [`RTCActor.start`](../env_actor/auto/inference_algorithms/rtc/rtc_actor.py).

### Structural subtyping (Protocol)
Python typing pattern: a class satisfies an interface by *shape* (having the right methods), not by inheritance. The `Policy` interface in [templates/policy.py](../env_actor/policy/templates/policy.py) is a `@runtime_checkable Protocol` so policies can be plain classes — no base class to import.

## T

### torch.compile
JIT compiler that traces a `nn.Module`'s `forward` and emits an optimized kernel. The `OpenPiBatchedWrapper.warmup` (called by `OpenPiPolicy.warmup`) triggers it on first invocation.

## V

### V4L2
Video4Linux2, the Linux kernel API for USB cameras. `RBRSCamera` opens devices with `cv2.VideoCapture(device, cv2.CAP_V4L2)`.

### VLA
Vision-Language-Action policy. A neural network that takes images, a text prompt, and robot state and outputs a sequence of motor commands.

## W

### Warmup
A dummy forward pass at startup. Runs once after the policy is moved to GPU so `cudnn.benchmark` and any `torch.compile` paths cache the right kernels before the first real inference.

### Weight transfer
The CPU→GPU state-dict mover in [weight_transfer.py](../env_actor/policy/utils/weight_transfer.py). Matches each loaded tensor to the target module's current device and dtype.
