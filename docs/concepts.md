# Concepts

The mental model you need before reading code. Each section ends with code references so you can drop into the implementation.

## Table of contents

- [What is a VLA policy?](#what-is-a-vla-policy)
- [Action chunks](#action-chunks)
- [Flow matching, in one paragraph](#flow-matching-in-one-paragraph)
- [Action inpainting](#action-inpainting)
- [Sequential vs RTC](#sequential-vs-rtc)
- [Ray actors](#ray-actors)
- [Shared memory](#shared-memory)
- [Spawn vs fork](#spawn-vs-fork)
- [ROS2 primer, scoped to this codebase](#ros2-primer-scoped-to-this-codebase)
- [The trainer/inference split](#the-trainerinference-split)

## What is a VLA policy?

VLA = **V**ision–**L**anguage–**A**ction. A model whose inputs are:

- One or more **camera images** (here: head, left, right, each `(3, 240, 320)` uint8).
- A **language prompt** (e.g. `"Pick up the sock and place it in the box."`).
- The **robot's own state**, called proprioception (here: a 24-d vector of arm and hand joint positions).

…and whose output is a **sequence of motor commands** — a "chunk" of future actions for the robot to execute. The model has been pretrained on robot demonstrations and natural-language captions, so it can be prompted at inference time the way a chat model can.

In this codebase the VLA is OpenPI/Pi0.5, wrapped by either [`OpenPiPolicy`](../env_actor/policy/policies/openpi_policy/openpi_policy.py) (the pretrained backbone alone) or [`DsrlOpenpiPolicy`](../env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.py) (an extra "noise actor" on top, trained with RL).

**Code references**: [policy/templates/policy.py](../env_actor/policy/templates/policy.py), [policy/policies/openpi_policy/openpi_policy.py](../env_actor/policy/policies/openpi_policy/openpi_policy.py), [policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.py](../env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.py).

## Action chunks

Predicting one action at a time forces the model to run on every control step (here 20 Hz = every 50 ms). Predicting **chunks** of `K` actions in one forward pass means the model runs at most every `K` steps, and the robot executes the buffered actions in between.

In this repo `K = action_chunk_size = 50`, so one inference produces 2.5 s of motion at 20 Hz. The trade-offs:

| Chunk size | Latency between inferences | Smoothness | Reactivity |
|---|---|---|---|
| **Small** (e.g. 5) | Frequent — inference runs hot | Less drift | High |
| **Large** (e.g. 50) | Rare — inference is cheap on average | Can drift far from the world before re-planning | Low |

You need a way to handle the transition when a new chunk arrives mid-execution: see [action inpainting](#action-inpainting).

**Code references**: `action_chunk_size` in [inference_runtime_params.json](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json), buffer logic in [data_manager_bridge.py](../env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_b/data_manager_bridge.py).

## Flow matching, in one paragraph

Modern generative models like OpenPI don't predict an action chunk directly — they **denoise** it. They start from pure Gaussian noise of shape `(50, 24)` and apply a learned "velocity field" `K` times that pushes the sample toward a plausible action chunk for the current observation. `num_inference_steps` is `K` — default 10. Each step is one forward pass of the model's action decoder. More steps → higher fidelity, more latency. This is the *flow-matching* family of objectives; you can think of it as a fast, simulation-free cousin of diffusion.

**Code references**: `num_inference_steps` in [openpi_batched.yaml](../env_actor/policy/policies/openpi_policy/components/openpi_batched.yaml); the ODE simulation in [`guided_action_chunk_inference`](../env_actor/inference_engine_utils/action_inpainting.py). External: OpenPI's paper.

## Action inpainting

When a new chunk arrives, the simplest thing is to drop the old one and switch. But the robot is mid-motion at the old chunk's `(t, t+1, t+2)`-th positions, and the new chunk's first positions correspond to *now*. If the policy is at all stochastic, switching hard will jerk the robot.

Action inpainting solves this by **blending** the two chunks. For each index `i` of the new chunk it computes a weight `w_i ∈ [0, 1]` and returns `prev_chunk[i] * w_i + new_chunk[i] * (1 - w_i)`. The weight schedule has three regions:

1. **Keep old** (`i < start = delay_steps`): weight is `1.0`. These actions were already in flight when inference began.
2. **Blend** (middle): weight decays exponentially toward 0 (`"exp"` schedule).
3. **Use new** (tail): weight is `0`. Long enough into the future that we trust the new prediction completely.

The math is in [`compute_guided_prefix_weights`](../env_actor/inference_engine_utils/action_inpainting.py); both policies call it from `guided_inference`. The technique is from [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/pdf/2506.07339).

**Code references**: [action_inpainting.py](../env_actor/inference_engine_utils/action_inpainting.py), [openpi_policy.py](../env_actor/policy/policies/openpi_policy/openpi_policy.py), [dsrl_openpi_policy.py](../env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.py).

## Sequential vs RTC

Two inference algorithms ship with the engine, and you pick one at the command line:

```bash
python run_inference.py --robot igris_b --inference_algorithm sequential
python run_inference.py --robot igris_b --inference_algorithm rtc  # default
```

### Sequential

One process. The control loop and inference call run in series.

```
Time ─────────────────────────────────────────────────→
read → predict ───────→ publish → read → [buffered] → publish → ...
        (blocks here)   chunk[0]                       chunk[1]
```

While `predict` is running, the robot publishes nothing. If `predict` takes 200 ms, the robot stops for 200 ms. Fine for offline debugging, dangerous for real-time control.

### RTC (Real-Time Control)

Two processes. The control loop runs at 20 Hz and never blocks; the inference loop runs whenever a new chunk is needed and writes into shared memory.

```
Control:   read → publish → read → publish → read → publish → read → publish → ...
           ↑ pulls action[i]      action[i+1]     action[i+2]     new_action[0]

Inference:       ╰── guided_inference() ──────────────────╯
                       (writes new chunk while control keeps moving)
```

The cost of RTC is complexity: shared memory, two processes, synchronization primitives. See [rtc_shared_memory.md](rtc_shared_memory.md) for the deep dive.

**Code references**: [`SequentialActor`](../env_actor/auto/inference_algorithms/sequential/sequential_actor.py), [`RTCActor`](../env_actor/auto/inference_algorithms/rtc/rtc_actor.py), [`control_loop.py`](../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py), [`inference_loop.py`](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py).

## Ray actors

[Ray](https://www.ray.io/) is a distributed-Python framework. The bits we use:

- A **Ray cluster** has one head node and zero or more workers. They are started by [start_ray.sh](../start_ray.sh).
- `@ray.remote` on a class makes it a **Ray actor**. Calling `SequentialActor.remote(...)` creates the instance in a separate Python process — possibly on a different machine — and returns an `ActorHandle`.
- `actor.method.remote(args)` does the RPC; it returns an `ObjectRef`.
- A **custom resource** like `"inference_pc": 1` is a label a worker declares so that actors requesting that resource (`SequentialActor.options(resources={"inference_pc": 1})`) get scheduled there. We use it to pin inference to the GPU worker.

In this codebase the entry point creates exactly **one** actor (`RTCActor` or `SequentialActor`) and calls `start.remote()` once. The actor owns the inference work for the lifetime of the program. The main process then sits on `signal.pause()` and forwards `Ctrl+C` into `ray.shutdown()`.

**Code references**: [run_inference.py](../run_inference.py), [`@ray.remote(num_gpus=1, num_cpus=3)`](../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L10), [`@ray.remote(num_gpus=1)`](../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L18), [start_ray.sh](../start_ray.sh).

## Shared memory

In Sequential, the control loop and the inference call live in the same Python process; data flow is a function call. In RTC, the control loop and the inference loop are *different processes*, and we need them to exchange ~250 KB of camera images every 50 ms without serializing through Ray or a `multiprocessing.Queue`.

The answer: a chunk of OS memory both processes `mmap`. Python's [`multiprocessing.shared_memory`](https://docs.python.org/3/library/multiprocessing.shared_memory.html) gives you exactly that — a `SharedMemory` object you can wrap in a numpy array and read or write zero-copy.

In this codebase the **parent process** (the `RTCActor` Ray worker) creates five regions in [`RTCActor.start`](../env_actor/auto/inference_algorithms/rtc/rtc_actor.py):

| Region | Shape (IGRIS_B) | Dtype | Writer | Reader |
|---|---|---|---|---|
| `proprio` | `(50, 24)` | float32 | control | inference |
| `head` | `(1, 3, 240, 320)` | uint8 | control | inference |
| `left` | `(1, 3, 240, 320)` | uint8 | control | inference |
| `right` | `(1, 3, 240, 320)` | uint8 | control | inference |
| `action` | `(50, 24)` | float32 | inference | control |

Synchronization uses `RLock`, two `Condition`s, two `Event`s, and two `Value`s — the deep dive is in [rtc_shared_memory.md](rtc_shared_memory.md).

**Code references**: [shared_memory_utils.py](../env_actor/auto/inference_algorithms/rtc/data_manager/utils/shared_memory_utils.py), [shm_manager_interface.py](../env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py), [shm_manager_bridge.py](../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py).

## Spawn vs fork

When Python spawns a child process via `multiprocessing`, it has two ways to do it:

- **`fork`** (the default on Linux) — copies the parent's address space, including all loaded modules, file descriptors, and CUDA contexts. Cheap.
- **`spawn`** — launches a brand-new Python interpreter, runs imports from scratch, then passes pickled arguments.

If the parent has already initialized CUDA (e.g. by importing `torch` and calling `torch.cuda.is_available()`), `fork` carries an inconsistent CUDA state into the child and you get:

```
RuntimeError: Cannot re-initialize CUDA in forked subprocess.
```

To avoid this the entrypoint sets the global start method early:

```python
# run_inference.py
try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass
```

And `RTCActor.start` uses a `spawn` context explicitly when creating the two child processes:

```python
ctx = mp.get_context("spawn")
# ...
ctx.Process(target=start_inference, args=(...)).start()
```

**Code references**: [run_inference.py:87-90](../run_inference.py#L87-L90), [rtc_actor.py:47](../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L47).

## ROS2 primer, scoped to this codebase

You don't need to be a ROS2 expert. The only things this codebase uses:

- **`rclpy`** — Python ROS2 client library.
- **A node** is a long-lived Python object that publishes and subscribes. Created in [`GenericRecorder.__init__`](../env_actor/robot_io_interface/robots/igris_b/utils/data_dict.py).
- **A publisher** sends messages on a topic. Joint targets go to `/igris_b/<robot_id>/target_joints` (`JointState`); finger targets go to `/igris_b/<robot_id>/finger_target` (`Float32MultiArray`).
- **A subscription** receives messages. Subscribed topics are listed in [inference_runtime_topics.json](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_topics.json).
- **An executor** drives subscription callbacks on a background thread. `controller_bridge.py` runs a `SingleThreadedExecutor` in a `threading.Thread`.

You must **source** ROS2 before launching anything — `source /opt/ros/<distro>/setup.bash`. If you don't, every `rclpy` import fails. See [troubleshooting.md](troubleshooting.md#rclpy-not-found--ros2-not-sourced).

**Code references**: [data_dict.py](../env_actor/robot_io_interface/robots/igris_b/utils/data_dict.py), [controller_bridge.py](../env_actor/robot_io_interface/robots/igris_b/controller_bridge.py). External: [ROS2 documentation](https://docs.ros.org/).

## The trainer/inference split

A clean separation is enforced by the submodule layout:

- **`inference_engine/`** (this repo) — runtime. Loads a trained model, runs a control loop, talks to the robot.
- **`inference_engine/trainer/`** (submodule) — training framework. Defines the `nn.Module` classes, runs the offline/online training loops, saves checkpoints. Read-only from here.
- **`inference_engine/trainer/policy_constructor/`** (nested submodule) — YAML-to-`nn.Module` builder. Owns the `_type_` registry, the graph compiler, and all the pretrained model wrappers (including `openpi_batched`).

`build_policy` (here) calls `PolicyConstructorModelFactory` (in trainer) which uses `policy_constructor` to assemble the components, then we wrap the result in a `Policy` class registered in `POLICY_REGISTRY`. The same YAML files used at training time work at inference time, so a checkpoint moves cleanly from one repo to the other.

**Code references**: [policy/utils/loader.py](../env_actor/policy/utils/loader.py) (`build_policy`), [policy/registry/__init__.py](../env_actor/policy/registry/__init__.py), [trainer/README.md](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/README.md), [policy_constructor/README.md](https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/README.md).
