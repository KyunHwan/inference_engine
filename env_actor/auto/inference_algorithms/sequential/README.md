# sequential

**Parent:** [inference_algorithms](../README.md)

Single-process, synchronous inference. The control loop and the policy forward pass run in the same thread; `predict()` blocks the loop while it runs. Use this for debugging or when latency doesn't matter.

## Table of contents

- [Purpose](#purpose)
- [Key files](#key-files)
- [Control-loop logic](#control-loop-logic)
- [Extension points](#extension-points)
- [Related docs](#related-docs)

## Purpose

Sequential is the simpler of the two algorithms. There is one process, one thread of execution (plus background threads for cameras and the rclpy executor), and the data flow is a function call chain. No shared memory, no condition variables. If something is wrong with your policy, your bridges, or your configs, Sequential will surface the bug with a clean stack trace; RTC may swallow it in a child process.

## Key files

| File | Purpose |
|---|---|
| [`sequential_actor.py`](sequential_actor.py) | `SequentialActor` — Ray remote with `num_gpus=1`. Its `start()` is the control loop. |
| [`data_manager/data_manager_interface.py`](data_manager/data_manager_interface.py) | `DataManagerInterface` — robot factory. |
| [`data_manager/robots/igris_b/data_manager_bridge.py`](data_manager/robots/igris_b/data_manager_bridge.py) | IGRIS_B data manager — proprio + image history buffers, action chunk caching, current-action indexing. |

## Control-loop logic

[`SequentialActor.start`](sequential_actor.py):

1. `controller_interface.start_state_readers()` — open cameras, start rclpy executor, wait for subscriptions.
2. `policy.warmup()` inside `torch.no_grad`. Failures are caught and logged.
3. Outer `while True:`:
   - `prev_joint = controller_interface.init_robot_position()`.
   - `data_manager_interface.init_inference_obs_state_buffer(controller_interface.read_state())`.
   - `next_t = time.perf_counter()`.
   - Inner `for t in range(9000):`:
     - `obs_data = controller_interface.read_state()`.
     - Skip the step if `"proprio"` is missing.
     - `data_manager_interface.update_state_history(obs_data)`.
     - If `t % policy_update_period == 0`:
       - `obs = data_manager_interface.serve_raw_obs_state()`.
       - `chunk = policy.predict(obs, data_normalization_interface)`.
       - `data_manager_interface.buffer_action_chunk(chunk, t)`.
     - `action = data_manager_interface.get_current_action(t)`.
     - `prev_joint, _ = controller_interface.publish_action(action, prev_joint)`.
     - Sleep to maintain `DT`.

The inner loop runs `9000` steps (= 7.5 minutes at 20 Hz) before resetting via the outer loop.

## Extension points

- **New robot** — add a per-robot `data_manager_bridge.py` under [data_manager/robots/](data_manager/data_manager_interface.py) and an `elif robot == "your_robot":` in `DataManagerInterface`. Walkthrough: [docs/walkthroughs/04_add_a_new_robot.md § Phase 3 Sequential](../../../../docs/walkthroughs/04_add_a_new_robot.md#phase-3-data-manager-bridges-sequential--rtc).
- **Tune `policy_update_period`** — in the runtime params JSON; controls how often `predict()` fires.
- **Reduce the episode length** — change the `for t in range(9000):` literal in `sequential_actor.py` if you want quicker resets in dev.

## Related docs

- [docs/walkthroughs/02_trace_one_step.md § Sequential](../../../../docs/walkthroughs/02_trace_one_step.md#sequential-one-full-iteration) — line-by-line trace.
- [docs/architecture.md § RTC vs Sequential, detailed](../../../../docs/architecture.md#rtc-vs-sequential-detailed).
- [docs/concepts.md § Sequential vs RTC](../../../../docs/concepts.md#sequential-vs-rtc).
