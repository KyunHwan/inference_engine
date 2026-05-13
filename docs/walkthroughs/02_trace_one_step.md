# Walkthrough 02 — Trace one control step

Two parallel traces of a single ~50 ms iteration of the control loop — Sequential and RTC side by side. Each row: the function, the file/line, the inputs/outputs (with shapes and dtypes), and what changed in shared state.

## Table of contents

- [Setup assumed by this walkthrough](#setup-assumed-by-this-walkthrough)
- [Sequential: one full iteration](#sequential-one-full-iteration)
- [RTC: one control-loop iteration + one inference-loop iteration](#rtc-one-control-loop-iteration--one-inference-loop-iteration)
- [What changes between the two](#what-changes-between-the-two)

## Setup assumed by this walkthrough

- Robot: IGRIS_B (so `state_dim=24`, `action_dim=24`, three cameras at 320×240, chunk size 50, 20 Hz).
- The actor has already initialized: `build_policy` completed, `start_state_readers()` filled the rclpy subscription cache, `init_robot_position()` returned `prev_joint = INIT_JOINT`, and the observation history is bootstrapped.
- We are now at iteration `t`. For Sequential, `t` is a multiple of `policy_update_period=50` so inference will fire this step. For RTC, the control loop is already running and the inference loop is blocked on `wait_for_min_actions(35)`.

## Sequential: one full iteration

| Step | File:line | Call | Input shape/dtype | Output shape/dtype | Notes |
|---|---|---|---|---|---|
| 1 | [sequential_actor.py:111](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L111) | `obs_data = controller_interface.read_state()` | — | `dict` with `proprio:(24,) float32`, `head/left/right:(3,240,320) uint8` | Cameras + rclpy joint subscriptions; `proprio` packed from 4 6-d topic slices in `_obs_dict_to_np_array`. |
| 2 | [sequential_actor.py:118](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L118) | `data_manager_interface.update_state_history(obs_data)` | dict above | — | FIFO-shift `robot_proprio_history`: new value at `[0]`, old `[0]` slides to `[1]`. Images replace `[0]` directly (`num_img_obs=1`). |
| 3 | [sequential_actor.py:121](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L121) | `if (t % policy_update_period) == 0:` — yes | — | — | We're firing inference this step. |
| 4 | [sequential_actor.py:123](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L123) | `obs = data_manager_interface.serve_raw_obs_state()` | — | dict with `proprio:(50,24) float32`, `head/left/right:(1,3,240,320) uint8` | Copies the proprio history, returns image arrays by reference. |
| 5 | [sequential_actor.py:127](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L127) | `denormalized_policy_output = policy.predict(obs, data_normalization_interface)` | dict above | `(50, 24) float32 np.ndarray` | For `OpenPiPolicy`: takes `obs["proprio"][0:1]` and `obs[cam][-1:]`, batches to `(1, ...)`, calls `OpenPiBatchedWrapper.predict(batched_obs, noise=None)`, strips batch dim. The wrapper handles normalization internally. |
| 6 | [sequential_actor.py:130](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L130) | `data_manager_interface.buffer_action_chunk(policy_output, t)` | `(50,24) float32`, `int` | — | Stores `last_action_chunk = policy_output`, `last_policy_step = t`. |
| 7 | [sequential_actor.py:133](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L133) | `action = data_manager_interface.get_current_action(t)` | `int` | `(24,) float32` | `idx = clip(t - last_policy_step, 0, 49)`. On a fresh chunk this is index 0. |
| 8 | [sequential_actor.py:136](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L136) | `smoothed_joints, fingers = controller_interface.publish_action(action, prev_joint)` | `(24,) float32`, `(12,) float32` | `(12,) float32, (12,) float32` | Slices `action` into 4 6-d vectors, concatenates joints as `[right, left]`, clips to `±max_delta`, publishes both `JointState` and `Float32MultiArray`. |
| 9 | [sequential_actor.py:142](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L142) | `prev_joint = smoothed_joints` | — | — | Carry over for next step's slew-rate computation. |
| 10 | [sequential_actor.py:145-148](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L145-L148) | `next_t += DT; time.sleep(...)` | — | — | Sleep until `next_t = previous_next_t + DT` (50 ms past previous deadline). |

On steps where `t % policy_update_period != 0`, steps 4–6 are skipped and step 7 returns `last_action_chunk[clip(t - last_policy_step, 0, 49)]`.

## RTC: one control-loop iteration + one inference-loop iteration

The two loops are running in two different processes. We trace one iteration of each.

### Control loop (at iteration `t`)

| Step | File:line | Call | Input | Output | Notes |
|---|---|---|---|---|---|
| C1 | [control_loop.py:120](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L120) | `if shm_manager.stop_event_is_set(): return` | — | — | Quick exit check. |
| C2 | [control_loop.py:125](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L125) | `obs_data = controller_interface.read_state()` | — | `dict` (24-d proprio + 3 cameras) | Same as Sequential step 1. |
| C3 | [control_loop.py:131](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L131) | `action = shm_manager.atomic_write_obs_and_increment_get_action(obs=obs_data, action_chunk_size=50)` | dict, 50 | `(24,) float32` | Under `RLock`: FIFO-shift `proprio` SHM, copy `obs[cam]` into SHM; increment `num_control_iters`; index `action_idx = clip(num_control_iters - 1, 0, 49)`; copy `action[action_idx]`; call `control_iter_cond.notify_all()` to wake inference. |
| C4 | [control_loop.py:133](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L133) | `if t > 100:` (warmup gate) | — | — | First 5 s of an episode: no publish. After that, publish. |
| C5 | [control_loop.py:135](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L135) | `smoothed_joints, fingers = controller_interface.publish_action(action, prev_joint)` | as Sequential step 8 | as Sequential step 8 | |
| C6 | [control_loop.py:141-144](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L141-L144) | `next_t += DT; time.sleep(...)` | — | — | Same precise timing as Sequential. |

### Inference loop (one iteration)

| Step | File:line | Call | Input | Output | Notes |
|---|---|---|---|---|---|
| I1 | [inference_loop.py:102](../../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L102) | `result = shm_manager.wait_for_min_actions(35)` | — | `str` | Blocks on `control_iter_cond` until `num_control_iters >= 35` (or `stop_event` / `episode_complete_event` set). Returns `"min_actions"`, `"stop"`, or `"episode_complete"`. |
| I2 | [inference_loop.py:113](../../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L113) | `shm_manager.set_inference_not_ready()` | — | — | Flips `inference_ready_flag` to False, notifies `inference_ready_cond`. |
| I3 | [inference_loop.py:117](../../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L117) | `input_data = shm_manager.atomic_read_for_inference()` | — | dict — see below | Under `RLock`: snapshot copies of SHM arrays + build `prev_action` (the un-executed tail of the current `action`, zero-padded). |
| I4 | [inference_loop.py:119-124](../../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L119-L124) | `with torch.inference_mode(), torch.autocast("cuda", dtype=bfloat16): next_actions = policy.guided_inference(input_data, normalizer, 35, 50)` | dict above | `(50, 24) float32` | The forward pass that does the real work. Inside the policy: `_run_inference` → backbone + noise processor + noise actor + OpenPI flow-matching → action chunk → inpainting blend with `prev_action`. |
| I5 | [inference_loop.py:126-128](../../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L126-L128) | `shm_manager.write_action_chunk_n_update_iter_val(next_actions, input_data["num_control_iters"])` | `(50,24)`, `int` | — | Under `RLock`: copy `next_actions` into SHM `action`; subtract `executed` from `num_control_iters` (so the control loop's `action_idx` lines back up with the start of the new chunk); push the new `num_control_iters` into `MaxDeque` for the next `est_delay`. |

`input_data` from `atomic_read_for_inference`:

| Key | Shape | Dtype | Origin |
|---|---|---|---|
| `proprio` | `(50, 24)` | float32 | copy of SHM `proprio` |
| `head` | `(1, 3, 240, 320)` | uint8 | copy of SHM `head` |
| `left` | `(1, 3, 240, 320)` | uint8 | copy of SHM `left` |
| `right` | `(1, 3, 240, 320)` | uint8 | copy of SHM `right` |
| `action` | `(50, 24)` | float32 | copy of SHM `action` (current chunk) |
| `prev_action` | `(50, 24)` | float32 | first `(50 - num_control_iters)` rows = `action[num_control_iters:]`; the rest = `0` |
| `num_control_iters` | scalar | int | counter value |
| `est_delay` | scalar | int | `MaxDeque.max()` |

## What changes between the two

| Aspect | Sequential | RTC |
|---|---|---|
| Process count | 1 (the Ray actor) | 3 (actor + control child + inference child) |
| Policy method called | `predict` | `guided_inference` |
| When does inference fire? | Every `policy_update_period = 50` control steps | Whenever `num_control_iters ≥ min_num_actions_executed = 35` since last call |
| Control loop while inference runs | Blocked | Continues at 20 Hz |
| How is the action selected? | `chunk[clip(t - last_policy_step, 0, 49)]` | `action[clip(num_control_iters - 1, 0, 49)]` from SHM |
| How are chunks stitched? | Hard switch on chunk reset (`last_policy_step` updated) | Action inpainting via `compute_guided_prefix_weights` |
| Episode length | `for t in range(9000):` (`~7.5 min` at 20 Hz) | `for t in range(1200):` (`60 s` at 20 Hz) per episode, outer `while True:` |
| Warmup period before publishing | None (publishes from step 0) | 100 steps (`if t > 100`) so the inference loop has time to write a real chunk |
| `prev_joint` carry | Local variable in `start()` | Local variable in `start_control` |

Both algorithms share the policy class (and so the same `_run_inference` body in DSRL or `_wrapper.predict` in OpenPI). The differences are entirely in the scheduling layer.
