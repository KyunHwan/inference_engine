# sequential/data_manager

**Parent:** [sequential](../README.md)

In-process observation history and action buffering. A robot-agnostic interface forwards every call to a robot-specific bridge.

## Files

| File | Purpose |
|---|---|
| [`data_manager_interface.py`](data_manager_interface.py) | `DataManagerInterface` — robot factory. Imports the matching bridge based on a string. |

## Subdirectories

| Directory | Purpose |
|---|---|
| [`robots/igris_b/`](robots/igris_b/README.md) | IGRIS_B `DataManagerBridge` — proprio history + image history + cached action chunk + per-step indexing. |
| `robots/igris_c/` | Stub. |

## Contracts

The bridge implements (called by [`SequentialActor`](../sequential_actor.py)):

| Method | Purpose |
|---|---|
| `init_inference_obs_state_buffer(init_data)` | Bootstrap proprio + image history with the first observation |
| `update_state_history(obs_data)` | FIFO-shift proprio (newest at `[0]`), replace image at `[0]` |
| `serve_raw_obs_state()` | Return `{"proprio": (H, D) copy, "head/left/right": (N, 3, H, W)}` |
| `buffer_action_chunk(chunk, t)` | Cache `chunk` and `t`; ndim=3 chunks are squeezed to 2D |
| `get_current_action(t)` | Return `chunk[clip(t - last_t, 0, K-1)]` |
| `serve_init_action()` | (Optional) bootstrap an init action vector if your RTC path needs one |

`serve_raw_obs_state` returns a *copy* of proprio (because the next step will FIFO-shift the buffer) but image arrays by reference. The policy's `predict` slices `[0:1]` / `[-1:]` immediately, so the reference-vs-copy distinction doesn't bite — but if you write a policy that retains the reference across steps you may see torn reads.

## Related docs

- [docs/walkthroughs/02_trace_one_step.md § Sequential](../../../../../docs/walkthroughs/02_trace_one_step.md#sequential-one-full-iteration)
- [docs/walkthroughs/04_add_a_new_robot.md § Phase 3 Sequential](../../../../../docs/walkthroughs/04_add_a_new_robot.md#phase-3-data-manager-bridges-sequential--rtc)
