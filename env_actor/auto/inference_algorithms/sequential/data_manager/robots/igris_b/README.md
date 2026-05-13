# sequential/data_manager/robots/igris_b

**Parent:** [data_manager](../../README.md)

IGRIS_B implementation of `DataManagerBridge`. Owns proprio + image history buffers and the cached action chunk.

## Files

| File | Purpose |
|---|---|
| [`data_manager_bridge.py`](data_manager_bridge.py) | `DataManagerBridge` class — implements every method in the bridge contract (see [parent README](../../README.md#contracts)). |

## State

| Attribute | Type | Notes |
|---|---|---|
| `robot_proprio_history` | `np.ndarray (proprio_history_size, state_dim)` | float32; newest at index 0; FIFO-shifted each step (if `size > 1`) |
| `img_obs_history` | `dict[str, np.ndarray (num_img_obs, 3, H, W)]` | One entry per camera in `runtime_params.camera_names` |
| `image_frame_counter` | `int` | Used with `img_obs_every` to subsample image history |
| `last_action_chunk` | `np.ndarray (action_chunk_size, action_dim)` | Most recent denormalized chunk from the policy |
| `last_policy_step` | `int` | `t` at which `last_action_chunk` was buffered |

## Behavior notes

- **`init_inference_obs_state_buffer`** uses `np.repeat(init_data["proprio"][np.newaxis, ...], num_robot_obs, axis=0)` to fill the entire history with the bootstrap value. Same for each camera.
- **`update_state_history`** has a guarded FIFO shift: if `proprio_history_size == 1`, no shift; if `num_img_obs == 1`, no image shift; the values just overwrite index 0. This matches the IGRIS_B JSON defaults (`num_img_obs=1`, `proprio_history_size=50`).
- **`buffer_action_chunk`** squeezes a leading batch dim and converts to numpy if needed. If the policy returns `(50, 24)` it's stored as-is; if it returns `(1, 50, 24)` it's squeezed.
- **`get_current_action`** uses `offset = current_step - last_policy_step` then `idx = clip(offset, 0, K-1)`. On a fresh chunk this is index 0; on the step just before the next inference it's at most index `policy_update_period - 1`. Since `policy_update_period (50) == action_chunk_size (50)` on IGRIS_B, the chunk is fully consumed at the boundary.
- **`serve_init_action`** does the same reshuffle as the RTC bridge (`INIT_JOINT_LIST[6:] + [:6]`, finger reshuffle, deg→rad and finger scale). Currently only called if a Sequential codepath needs an init action; the inference actor itself doesn't use it.

## Related docs

- [docs/api.md § Key data shapes](../../../../../../../docs/api.md#key-data-shapes-igris_b-defaults)
- [docs/walkthroughs/02_trace_one_step.md § Sequential](../../../../../../../docs/walkthroughs/02_trace_one_step.md#sequential-one-full-iteration) — steps 2, 4, 6, 7 cover this bridge.
