# rtc/data_manager/robots/igris_b

**Parent:** [data_manager](../../README.md)

IGRIS_B implementation of `SharedMemoryManager`. Owns every atomic operation on the RTC shared-memory regions plus the delay estimator used by action inpainting.

## Table of contents

- [Files](#files)
- [What it does](#what-it-does)
- [IGRIS_B-specific details](#igris_b-specific-details)
- [Related docs](#related-docs)

## Files

| File | Purpose |
|---|---|
| [`shm_manager_bridge.py`](shm_manager_bridge.py) | `SharedMemoryManager` class. Attaches to SHM regions, holds sync primitives, exposes the atomic read/write API. |

## What it does

The class is instantiated via the `attach_from_specs` classmethod, which is called by both child processes in [`control_loop.py`](../../../actors/control_loop.py) and [`inference_loop.py`](../../../actors/inference_loop.py). Each attached instance:

- Stores references to the SHM handles and to the synchronization primitives created by the parent.
- Holds a `MaxDeque(buffer_len=5)` initialized with `5` (the initial delay estimate).
- Provides atomic operations under a shared `RLock`:
  - Read: `atomic_read_for_inference` (returns deep copies + builds `prev_action`).
  - Write: `atomic_write_obs_and_increment_get_action`, `write_action_chunk_n_update_iter_val`.
  - Bootstrap: `init_action_chunk`, `bootstrap_obs_history`, `init_action_chunk_obs_history`.
  - Episode handshake: `signal_episode_complete`, `is_episode_complete`, `clear_episode_complete`.
- Cleans up: closes its SHM views (only the creator unlinks).

## IGRIS_B-specific details

### Initial action chunk layout

`init_action_chunk` (and `init_action_chunk_obs_history`) builds an `init_vec` from the joint and finger init constants in [init_params.py](../../../../../../runtime_settings_configs/robots/igris_b/init_params.py):

```python
init_vec = np.asarray(
    INIT_JOINT_LIST[6:]   # right-arm joints (the policy expects right after left in some channels...)
    + INIT_JOINT_LIST[:6] # left-arm joints
    + INIT_HAND_LIST[:6]  # left hand fingers
    + INIT_HAND_LIST[6:], # right hand fingers
    dtype=np.float32,
)
init_vec[:12] *= np.pi / 180.0   # joints to radians
init_vec[12:] *= 0.03            # finger scale factor
```

This reshuffling reflects a mismatch between `INIT_JOINT_LIST` ordering (right then left) and the policy's expected action layout (left then right). When you add a new robot, re-derive this — don't blindly copy.

### Delay estimation

After `write_action_chunk_n_update_iter_val`, the bridge does `self._delay_queue.add(self._num_control_iters.value)` — pushing the (now-decremented) counter into the queue. On the next inference call `atomic_read_for_inference` reports `est_delay = self._delay_queue.max()`. This is a conservative latency hedge: even if inference *usually* takes 4 steps, if it once took 9 in the last 5 calls the inpainting weights will protect that case.

### FIFO proprio history

`atomic_write_obs_and_increment_get_action` shifts the proprio history *only if* `proprio_history_size > 1`. For IGRIS_B's `proprio_history_size=50` setting, the array slides forward by one each step.

## Related docs

- [docs/rtc_shared_memory.md](../../../../../../../docs/rtc_shared_memory.md) — the canonical doc on what each method does and what it protects.
- [docs/walkthroughs/02_trace_one_step.md § RTC](../../../../../../../docs/walkthroughs/02_trace_one_step.md#rtc-one-control-loop-iteration--one-inference-loop-iteration) — sequence per step.
- [docs/walkthroughs/04_add_a_new_robot.md § Phase 3 RTC](../../../../../../../docs/walkthroughs/04_add_a_new_robot.md#phase-3-data-manager-bridges-sequential--rtc) — what to change for a new robot.
