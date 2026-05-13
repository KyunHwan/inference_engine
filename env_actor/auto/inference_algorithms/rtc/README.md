# rtc

**Parent:** [inference_algorithms](../README.md)

Real-Time Control. Dual-process inference algorithm: a 20 Hz control loop and an asynchronous inference loop, communicating through OS shared memory.

## Table of contents

- [Purpose](#purpose)
- [Key files](#key-files)
- [Contracts](#contracts)
- [How it plugs into the system](#how-it-plugs-into-the-system)
- [Extension points](#extension-points)
- [Related docs](#related-docs)

## Purpose

When a policy forward pass takes longer than one control period (e.g., 200 ms vs 50 ms at 20 Hz), Sequential blocks the robot during inference. RTC avoids that by running inference in a separate process: the control loop keeps publishing buffered actions from the previous chunk while the inference loop computes the next chunk into shared memory.

## Key files

| File | What it does |
|---|---|
| [`rtc_actor.py`](rtc_actor.py) | Ray remote actor. Creates the 5 SHM regions and the 6 sync primitives, spawns two child processes (`spawn` context), supervises them, cleans up on shutdown. |
| [`actors/control_loop.py`](actors/control_loop.py) | Control-side child. Runs `for t in range(1200):` at 20 Hz: read state → atomic SHM write → publish action (after a 100-step warmup gate). |
| [`actors/inference_loop.py`](actors/inference_loop.py) | Inference-side child. Loops: `wait_for_min_actions(35)` → snapshot SHM → `policy.guided_inference(...)` under `inference_mode + autocast(bfloat16)` → write chunk back. |
| [`data_manager/shm_manager_interface.py`](data_manager/shm_manager_interface.py) | Robot factory for the shared-memory manager. |
| [`data_manager/robots/igris_b/shm_manager_bridge.py`](data_manager/robots/igris_b/shm_manager_bridge.py) | IGRIS_B implementation. Owns all atomic SHM ops + the `MaxDeque` delay estimator. |
| [`data_manager/utils/shared_memory_utils.py`](data_manager/utils/shared_memory_utils.py) | `ShmArraySpec`, `create_shared_ndarray`, `attach_shared_ndarray`. |
| [`data_manager/utils/max_deque.py`](data_manager/utils/max_deque.py) | Sliding-window max over recent inference latencies → `est_delay`. |

## Contracts

The control loop and inference loop both implement [`SharedMemoryInterface`](data_manager/shm_manager_interface.py)'s API surface — there is no separate Python interface class; the bridge methods are the contract. See [docs/api.md § SharedMemoryInterface (RTC only)](../../../../docs/api.md#sharedmemoryinterface-rtc-only).

The shared-memory layout is fixed by [`RTCActor.start`](rtc_actor.py):

```
proprio:  (proprio_history_size, proprio_state_dim) float32
head:     (num_img_obs, 3, H, W)                    uint8
left:     (num_img_obs, 3, H, W)                    uint8
right:    (num_img_obs, 3, H, W)                    uint8
action:   (action_chunk_size, action_dim)           float32
```

Shapes come from the robot's [`RuntimeParams`](../../../runtime_settings_configs/robots/igris_b/inference_runtime_params.py).

## How it plugs into the system

[run_inference.py](../../../../run_inference.py):

```python
RTCActor.options(resources={"inference_pc": 1}, num_cpus=3, num_gpus=1).remote(
    robot=robot, policy_yaml_path=..., inference_runtime_params_config=..., inference_runtime_topics_config=...,
)
```

The actor then:

1. Loads `RuntimeParams` from the JSON.
2. Allocates 5 `SharedMemory` blocks (`create_shared_ndarray`) — the actor is the *sole creator*; both children attach and call `resource_tracker.unregister` so only the parent unlinks.
3. Creates `RLock`, two `Condition`s, two `Event`s, two `Value`s.
4. Spawns `start_inference` (loads policy on GPU) and `start_control` (opens cameras, rclpy).
5. Joins both children; if either exits non-zero, sets `stop_event` and notifies waiters.
6. On exit: closes/unlinks each SHM block, unregisters from `resource_tracker`.

## Extension points

- **New robot** — add `shm_manager_bridge.py` under [data_manager/robots/your_robot/](data_manager/robots/igris_b/README.md) and an `elif robot == "your_robot":` in [shm_manager_interface.py](data_manager/shm_manager_interface.py). Walkthrough: [docs/walkthroughs/04_add_a_new_robot.md § Phase 3 RTC](../../../../docs/walkthroughs/04_add_a_new_robot.md#phase-3-data-manager-bridges-sequential--rtc).
- **Tune `min_num_actions_executed`** — currently hardcoded to `35` at the top of [`inference_loop.py`](actors/inference_loop.py). Lowering it makes inference re-fire sooner (more responsive, more GPU load).
- **Tune `episode_length`** — `1200` in [`control_loop.py`](actors/control_loop.py). 60 s episodes at 20 Hz.
- **Tune the publish warmup** — `if t > 100` in [`control_loop.py`](actors/control_loop.py). 5 s of silent burn-in.

## Related docs

- [docs/rtc_shared_memory.md](../../../../docs/rtc_shared_memory.md) — the deep dive on SHM layout and synchronization.
- [docs/walkthroughs/02_trace_one_step.md § RTC](../../../../docs/walkthroughs/02_trace_one_step.md#rtc-one-control-loop-iteration--one-inference-loop-iteration) — the sequence diagram.
- [docs/architecture.md § Process and thread topology](../../../../docs/architecture.md#process-and-thread-topology).
