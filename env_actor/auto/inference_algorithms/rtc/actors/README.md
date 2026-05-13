# rtc/actors

**Parent:** [rtc](../README.md)

The two child processes that `RTCActor` spawns. Each is a top-level function (not a class) because `spawn`-context multiprocessing pickles the target callable.

## Table of contents

- [Files](#files)
- [Contracts](#contracts)
- [Lifecycle](#lifecycle)
- [Related docs](#related-docs)

## Files

| File | Process role |
|---|---|
| [`control_loop.py`](control_loop.py) | Control-side child. 20 Hz loop reading state from cameras + ROS2 and publishing actions. |
| [`inference_loop.py`](inference_loop.py) | Inference-side child. Owns the policy on GPU. Runs `guided_inference` whenever enough control steps have elapsed. |

## Contracts

Both functions accept exactly the same arguments — the SHM specs and synchronization primitives created by the parent:

```python
def start_control(
    robot,
    inference_runtime_params_config,
    inference_runtime_topics_config,
    shm_specs,             # dict[str, ShmArraySpec]
    lock,                  # RLock
    control_iter_cond,     # Condition(lock)
    inference_ready_cond,  # Condition(lock)
    stop_event,            # Event
    episode_complete_event,# Event
    num_control_iters,     # Value('i', 0, lock=False)
    inference_ready_flag,  # Value(c_bool, False, lock=False)
):
    ...

def start_inference(...): ...   # same signature plus policy_yaml_path
```

`shm_specs` are *not* live `SharedMemory` objects — they are `ShmArraySpec` descriptors. Each child re-`attach`es by name via [`attach_shared_ndarray`](../data_manager/utils/shared_memory_utils.py).

## Lifecycle

### Control loop

[`control_loop.py:start_control`](control_loop.py):

1. `ray.init(address="auto", namespace="online_rl", log_to_driver=True)` — re-attach to the cluster from the child.
2. Load `RuntimeParams` and the topics config.
3. Instantiate `ControllerInterface` and `SharedMemoryInterface.attach_from_specs(...)`.
4. `controller_interface.start_state_readers()` — opens cameras, starts the `rclpy` executor in a background thread.
5. Outer `while True:`:
   - If `episode >= 0`: `shm_manager.signal_episode_complete()`.
   - `shm_manager.wait_for_inference_ready()` — blocks until inference has warmed up.
   - `shm_manager.clear_episode_complete()`; `shm_manager.reset()`; `init_robot_position()`; `shm_manager.init_action_chunk_obs_history(controller_interface.read_state())`.
   - Inner `for t in range(1200):`:
     - Stop check.
     - `read_state()` → `obs_data`.
     - `action = shm_manager.atomic_write_obs_and_increment_get_action(obs=obs_data, action_chunk_size=...)`.
     - `if t > 100: controller_interface.publish_action(action, prev_joint)`.
     - Sleep to maintain `DT`.
6. `finally:` `shm_manager.cleanup(); controller_interface.shutdown()`.

### Inference loop

[`inference_loop.py:start_inference`](inference_loop.py):

1. `ray.init(...)`.
2. `device = cuda if available`; `cudnn.benchmark = True`; `set_float32_matmul_precision("high")`.
3. `policy = build_policy(...).to(device); policy.eval()`.
4. `policy.warmup()` inside `torch.no_grad`.
5. Build `SharedMemoryInterface.attach_from_specs(...)` and `DataNormalizationInterface(robot, runtime_params.read_stats_file())`.
6. Outer `while True:`:
   - `shm_manager.set_inference_ready()`.
   - Inner `while True:`:
     - `result = shm_manager.wait_for_min_actions(35)`.
     - If `"stop"`: return.
     - If `"episode_complete"`: break.
     - Else: `set_inference_not_ready()`; `atomic_read_for_inference()` → `input_data`.
     - Inside `inference_mode + autocast(bfloat16)`: `next_actions = policy.guided_inference(input_data, normalizer, 35, runtime_params.action_chunk_size)`.
     - `shm_manager.write_action_chunk_n_update_iter_val(next_actions, input_data["num_control_iters"])`.
7. `finally:` `shm_manager.cleanup()`.

## Related docs

- [docs/rtc_shared_memory.md § Sequence — one full handshake](../../../../../docs/rtc_shared_memory.md#sequence--one-full-handshake)
- [docs/walkthroughs/02_trace_one_step.md § RTC](../../../../../docs/walkthroughs/02_trace_one_step.md#rtc-one-control-loop-iteration--one-inference-loop-iteration)
