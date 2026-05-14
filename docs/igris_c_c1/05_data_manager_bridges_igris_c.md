# 05 — Data manager bridges for IGRIS_C

**What this covers.** The two IGRIS_C data-manager implementations on this branch: the **sequential** one (single-threaded, easy to debug) and the **RTC shared-memory** one (dual-process, production). For each, what state it maintains, how observations and actions flow through it, and how it is selected by the factory.
**Who this is for.** Anyone changing the observation pipeline, action buffering, RTC shared-memory layout, or adding a new robot.

This document mirrors `docs/rtc_shared_memory.md` (the algorithm-side architectural reference) and adds the IGRIS_C-specific details — shapes, init constants, and the file locations under `robots/igris_c/`.

## Table of contents

- [Two algorithms, two bridges](#two-algorithms-two-bridges)
- [Sequential `DataManagerBridge`](#sequential-datamanagerbridge)
- [RTC `SharedMemoryManager`](#rtc-sharedmemorymanager)
- [How the data manager is selected by the factory](#how-the-data-manager-is-selected-by-the-factory)
- [`rtc/` vs `rtc_local/` — what differs](#rtc-vs-rtc_local--what-differs)

## Two algorithms, two bridges

| Inference algorithm | IGRIS_C data manager | Lives at |
|---|---|---|
| Sequential | `DataManagerBridge` | [`env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_c/data_manager_bridge.py`](../../env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_c/data_manager_bridge.py) |
| RTC (Ray) | `SharedMemoryManager` | [`env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_c/shm_manager_bridge.py`](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_c/shm_manager_bridge.py) |
| RTC (local, no-Ray) | `SharedMemoryManager` (a parallel copy) | [`env_actor/auto/inference_algorithms/rtc_local/data_manager/robots/igris_c/shm_manager_bridge.py`](../../env_actor/auto/inference_algorithms/rtc_local/data_manager/robots/igris_c/shm_manager_bridge.py) |

The sequential algorithm runs everything in one Python process — observation history, policy forward pass, action buffering, robot publishing — and the data manager is plain numpy state. The RTC algorithm splits the work across two processes (a control loop and an inference loop) and the data manager is a `SharedMemoryManager` that owns the numpy views into multiprocessing-backed `SharedMemory` blocks plus all the synchronization primitives.

## Sequential `DataManagerBridge`

Source: [`sequential/data_manager/robots/igris_c/data_manager_bridge.py`](../../env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_c/data_manager_bridge.py).

Constructed once per actor in [`sequential_actor.SequentialActor.__init__`](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L72) via the factory `DataManagerInterface` ([`sequential/data_manager/data_manager_interface.py:6-7`](../../env_actor/auto/inference_algorithms/sequential/data_manager/data_manager_interface.py#L6)).

### State

```python
self.runtime_params      = runtime_params
self.num_robot_obs       = runtime_params.proprio_history_size   # 50
self.num_image_obs       = runtime_params.num_img_obs            # 1
self.num_queries         = runtime_params.action_chunk_size      # 50
self.state_dim           = runtime_params.proprio_state_dim      # 86
self.action_dim          = runtime_params.action_dim             # 17
self.camera_names        = runtime_params.camera_names           # ["head","left","right"]

self.img_obs_history     = None    # set in init_inference_obs_state_buffer
self.robot_proprio_history = None  # set in init_inference_obs_state_buffer
self.image_frame_counter = 0

self.last_action_chunk = None      # last policy output, shape (50, 17)
self.last_policy_step  = -1
```

### Observation history flow

Per control step, the [sequential actor loop](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L107) calls:

```python
data_manager_interface.update_state_history(obs_data)
```

[Implementation:](../../env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_c/data_manager_bridge.py#L36)

```python
def update_state_history(self, obs_data):
    if self.runtime_params.proprio_history_size > 1:
        self.robot_proprio_history[1:] = self.robot_proprio_history[:-1]   # shift older
    self.robot_proprio_history[0] = obs_data['proprio']                    # write newest

    for cam_name in self.camera_names:
        if self.runtime_params.img_obs_every <= 1 or \
                (self.image_frame_counter % self.runtime_params.img_obs_every == 0):
            if self.runtime_params.num_img_obs > 1:
                self.img_obs_history[cam_name][1:] = self.img_obs_history[cam_name][:-1]
        self.img_obs_history[cam_name][0] = obs_data[cam_name]

    self.image_frame_counter += 1
```

In English: it is a circular ring buffer of depth `proprio_history_size` for proprio and `num_img_obs` for each camera, with new samples written at index 0 and older samples shifted by one slot. The `img_obs_every` gate lets you down-sample image inputs without affecting the proprio rate (e.g. set to `5` if your policy wants images every 5 control steps).

### Normalization pathway

The sequential data manager does **not** normalize internally. It serves raw observations via `serve_raw_obs_state()` ([line 85](../../env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_c/data_manager_bridge.py#L85)); the **policy** then calls `data_normalization_interface.normalize_state(...)` inside its `predict()` method.

See [06_normalization_igris_c.md](06_normalization_igris_c.md) for the IGRIS_C normalization math. The relevant call sites:
- [`sequential_actor.py:127`](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L127) — `policy.predict(obs, self.data_normalization_interface)`.
- The policy is expected to call `normalize_state` itself; the `DataNormalizationInterface` factory at [`nom_stats_manager/data_normalization_interface.py`](../../env_actor/nom_stats_manager/data_normalization_interface.py) routes to the IGRIS_C `DataNormalizationBridge`.

### Action buffering and selection

After the policy emits a chunk:

```python
self.data_manager_interface.buffer_action_chunk(denormalized_policy_output, t)
```

[Implementation:](../../env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_c/data_manager_bridge.py#L50)

```python
def buffer_action_chunk(self, policy_output, current_step):
    self.last_action_chunk = (
        policy_output.squeeze(0).cpu().numpy()
        if policy_output.ndim == 3
        else policy_output.cpu().numpy()
    )
    self.last_policy_step = current_step
```

The `squeeze(0)` handles `(1, chunk, action_dim)` from a batched policy; falls back to `(chunk, action_dim)` for non-batched.

Per-step action selection: [`get_current_action(t)`](../../env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_c/data_manager_bridge.py#L58):

```python
offset = current_step - self.last_policy_step
idx    = int(np.clip(offset, 0, self.last_action_chunk.shape[0] - 1))
return self.last_action_chunk[idx]
```

This is **simple linear advancement** through the chunk — at step `t`, return action `t - last_policy_step`. Since the sequential algorithm calls the policy every `policy_update_period` steps (50 by default), and the chunk size is also 50, the chunk is consumed exactly between policy updates.

### Episode bootstrap

```python
def init_inference_obs_state_buffer(self, init_data):
    self.image_frame_counter = 0
    self.last_policy_step    = -1
    self.img_obs_history     = {
        cam: np.repeat(init_data[cam][np.newaxis, ...], self.num_image_obs, axis=0)
        for cam in self.camera_names
    }
    self.robot_proprio_history = np.repeat(
        init_data['proprio'][np.newaxis, ...], self.num_robot_obs, axis=0,
    )
```

Repeats the first observation across the history dimension so the policy never sees a partial history. Called once per episode in [`sequential_actor.start()`](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L103).

### Initial-action seed

```python
def serve_init_action(self):
    return np.tile(INIT_ACTION_17[None, :], (self.runtime_params.action_chunk_size, 1))
```

Tiles the 17-D `INIT_ACTION_17` (from `init_params.py`) into a `(50, 17)` chunk. The sequential actor doesn't actually call this — the **RTC** bridge does, as the init seed for the shared-memory action block. But the sequential data manager also exposes it for parity with the IGRIS_B interface.

### Differences from IGRIS_B sequential bridge

[`sequential/data_manager/robots/igris_b/data_manager_bridge.py`](../../env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_b/data_manager_bridge.py) is **algorithmically identical** — the file is line-for-line the same except for:

| Aspect | IGRIS_B | IGRIS_C |
|---|---|---|
| `serve_init_action()` body | Builds `INIT_JOINT_LIST[6:] + [:6] + INIT_HAND_LIST[:6] + [6:]`, converts joints to rad, scales fingers by 0.03, then tiles | `np.tile(INIT_ACTION_17, ...)` |
| Imports | `INIT_JOINT_LIST, INIT_HAND_LIST, INIT_JOINT, IGRIS_B_STATE_KEYS` | `INIT_ACTION_17` only |
| Docstring | "IGRIS_B data manager — handles ALL data processing" | "IGRIS_C — Algorithm is identical to igris_b's bridge; only the action-init shape differs" |

If you need to add a third robot, this file is the right template — change the imports and `serve_init_action()` body; everything else copies.

## RTC `SharedMemoryManager`

Source: [`rtc/data_manager/robots/igris_c/shm_manager_bridge.py`](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_c/shm_manager_bridge.py). Mirror copy at [`rtc_local/data_manager/robots/igris_c/shm_manager_bridge.py`](../../env_actor/auto/inference_algorithms/rtc_local/data_manager/robots/igris_c/shm_manager_bridge.py).

The IGRIS_C variant is **a direct port of the IGRIS_B variant**; the only change is the init-action vector. The docstring at [`shm_manager_bridge.py:12-14`](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_c/shm_manager_bridge.py#L12) makes this explicit:

> This file is the igris_c port of the igris_b shm_manager_bridge: same algorithm, only the action-init shape differs (action_dim=17 init vector built from `INIT_ACTION_17` vs igris_b's 24-D `INIT_JOINT_LIST + INIT_HAND_LIST` concat).

Read [`docs/rtc_shared_memory.md`](../rtc_shared_memory.md) first for the conceptual design. This document adds the IGRIS_C specifics.

### Shared-memory layout

Created by the parent process (the `@ray.remote` `RTCActor` or the local `RTCLocalActor`) in [`rtc_actor.RTCActor.start()`](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L54) (or [`rtc_local_actor.RTCLocalActor.start()`](../../env_actor/auto/inference_algorithms/rtc_local/rtc_local_actor.py#L53) — same code). For IGRIS_C with the default config:

| SHM key | Shape | dtype | Bytes (≈) | Source dim |
|---|---|---|---|---|
| `proprio` | `(50, 86)` | `float32` | 16,800 | `proprio_history_size × proprio_state_dim` |
| `head` | `(1, 3, 224, 224)` | `uint8` | 150,528 | `num_img_obs × 3 × H × W` |
| `left` | `(1, 3, 224, 224)` | `uint8` | 150,528 | same |
| `right` | `(1, 3, 224, 224)` | `uint8` | 150,528 | same |
| `action` | `(50, 17)` | `float32` | 3,400 | `action_chunk_size × action_dim` |

Total ≈ 470 KB per Ray actor instance.

The OS-level shared memory name is generated by `multiprocessing.shared_memory.SharedMemory` and passed to the child processes via a `ShmArraySpec` (defined in [`shared_memory_utils.py:7-18`](../../env_actor/auto/inference_algorithms/rtc/data_manager/utils/shared_memory_utils.py#L7)). Child processes attach by **name**, not by inheriting the handle — see [`attach_shared_ndarray`](../../env_actor/auto/inference_algorithms/rtc/data_manager/utils/shared_memory_utils.py#L43). The child also calls `resource_tracker.unregister(...)` for each block so only the parent ever unlinks them on shutdown — this is the standard fix for `_posixshmem` double-unlink warnings.

### Synchronization primitives

All created in the parent process inside [`rtc_actor.start()` lines 80-86](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L80):

```python
lock                   = ctx.RLock()
control_iter_cond      = ctx.Condition(lock)   # signals: num_control_iters changed
inference_ready_cond   = ctx.Condition(lock)   # signals: inference_ready_flag changed
stop_event             = ctx.Event()           # process-wide stop signal
episode_complete_event = ctx.Event()           # control side announces end-of-episode
num_control_iters      = ctx.Value('i', 0, lock=False)
inference_ready_flag   = ctx.Value(c_bool, False, lock=False)
```

`ctx` is `mp.get_context("spawn")` — see "Why `spawn`?" in [01_quickstart_igris_c.md](01_quickstart_igris_c.md). The control loop and inference loop are both spawned as `ctx.Process(...)` children.

### Two-process protocol

The IGRIS_C `SharedMemoryManager` is the contract between the two:

```
┌───────────────────────────────────────────┐    ┌──────────────────────────────────────────┐
│  CONTROL LOOP (rtc/actors/control_loop.py)│    │ INFERENCE LOOP (rtc/actors/inference_loop)│
│                                           │    │                                           │
│  start_state_readers()                    │    │  build_policy() + warmup                  │
│  wait_for_inference_ready()  ─────────────┼────┼──────────►  set_inference_ready()         │
│  clear_episode_complete()                 │    │                                           │
│  init_action_chunk_obs_history(...)       │    │                                           │
│                                           │    │                                           │
│  for t in 0..episode_length:              │    │   while inner-loop:                       │
│    obs = controller.read_state()          │    │     wait_for_min_actions(35)              │
│    action = atomic_write_obs_and_         ├────┼─►  (returns when num_control_iters ≥ 35) │
│              increment_get_action(obs,...)│    │     set_inference_not_ready()             │
│    if t > 100:                            │    │     in = atomic_read_for_inference()      │
│      controller.publish_action(action,...)│    │     out = policy.guided_inference(in,...) │
│                                           │    │     write_action_chunk_n_update_iter_val( │
│                                           │    │             out, num_control_iters)      │
└───────────────────────────────────────────┘    └──────────────────────────────────────────┘
```

The arrows are the wait/notify points on the two Conditions. The two `Value`s carry the actual integer/bool state; the Conditions just signal that the value changed. All read/write to the SHM arrays is wrapped in `with self._lock:` ([several occurrences](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_c/shm_manager_bridge.py#L172)).

### Method-by-method

| Method | Caller | What it does |
|---|---|---|
| `wait_for_min_actions(min_actions)` | inference loop | Blocks on `control_iter_cond` until `num_control_iters ≥ min_actions` or stop/episode-complete. Returns `'min_actions' | 'episode_complete' | 'stop'`. See [line 119](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_c/shm_manager_bridge.py#L119). |
| `wait_for_inference_ready()` | control loop | Blocks on `inference_ready_cond` until the inference side flips `inference_ready_flag` (after warmup). Returns `False` if stop event fires. |
| `set_inference_ready()` / `set_inference_not_ready()` | inference loop | Flip the flag and notify_all. |
| `signal_episode_complete()` / `clear_episode_complete()` | control loop | Set/clear `episode_complete_event` and notify. |
| `notify_step()` | control loop | Wake any inference-side `wait_for_min_actions` waiter. |
| `atomic_read_for_inference()` | inference loop | Take the lock, copy proprio/cams/action from SHM into a fresh dict, build `prev_action` (tail of the action chunk that hasn't been executed yet), include `est_delay = max(delay_queue)`. See [line 170](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_c/shm_manager_bridge.py#L170). |
| `atomic_write_obs_and_increment_get_action(obs, chunk_size)` | control loop | Increment `num_control_iters`, shift the proprio ring, write new obs, compute `idx = clip(num_control_iters-1, 0, chunk_size-1)`, return `action[idx].copy()`. See [line 187](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_c/shm_manager_bridge.py#L187). |
| `write_action_chunk_n_update_iter_val(chunk, executed)` | inference loop | Squeeze leading batch dim, convert torch→np if needed, copy into SHM `action`, **decrement `num_control_iters` by `executed`** (how many actions the control side ran during inference), record this delay in `delay_queue`. See [line 209](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_c/shm_manager_bridge.py#L209). |
| `init_action_chunk()` | control loop | Tiles `INIT_ACTION_17` across the SHM action rows. **IGRIS_C-specific bit.** See [line 219](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_c/shm_manager_bridge.py#L219). |
| `bootstrap_obs_history(obs_history)` | control loop | Repeats the first proprio across the proprio ring; copies camera frames in directly. |
| `init_action_chunk_obs_history(obs_history)` | control loop | Combines `bootstrap_obs_history` + `init_action_chunk`. Called once at the top of each episode. |
| `reset()` | control loop | Zero `num_control_iters`, reset `delay_queue` to `[15]`. |
| `cleanup()` | both | Close SHM handles. Only the creator (parent) unlinks. |
| `signal_stop()` | external | Set `stop_event` and notify both Conditions so all waiters wake up. |

### Init seed: the IGRIS_C-specific bit

The **only** IGRIS_C-specific difference vs. IGRIS_B is at lines [222-224](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_c/shm_manager_bridge.py#L222) (inside `init_action_chunk`) and [265-266](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_c/shm_manager_bridge.py#L265) (inside `init_action_chunk_obs_history`):

```python
np.copyto(
    self._shm_array_dict['action'],
    np.tile(INIT_ACTION_17, (self._shm_array_dict['action'].shape[0], 1)),
)
```

`INIT_ACTION_17` is imported from [`runtime_settings_configs/robots/igris_c/init_params.py`](../../env_actor/runtime_settings_configs/robots/igris_c/init_params.py#L64) — the 17-D init action (left arm + right arm + left hand 0 + right hand 0 + waist yaw).

### Delay queue (the 15)

`self._delay_queue = MaxDeque(buffer_len=5)` with initial `add(15)`. The 15 is the **estimated delay floor** the inference side reports to guided inference until the first real measurement arrives. After every inference cycle, `num_control_iters` (the number of actions the control side ran while inference was busy) is pushed onto the deque; `est_delay = max(last 5 values)` is what the policy sees as "how far ahead I need to plan."

On `main`, this initial value was `5` for IGRIS_B — see [02 § IGRIS_B parity](02_changes_vs_main.md#env_actorautoinference_algorithmsrtcdata_managerrobotsigris_bshm_manager_bridgepy-modified). The branch bumps both IGRIS_B and IGRIS_C to `15` to match measured IGRIS_C latencies.

### Why `torch.multiprocessing.set_start_method("spawn")` matters

Both entry points call `torch.multiprocessing.set_start_method("spawn")` before importing the actor — see [`run_inference.py:88`](../../run_inference.py#L88) and [`run_inference_local.py:71`](../../run_inference_local.py#L71). The RTC actor's `start()` then explicitly grabs `ctx = mp.get_context("spawn")` ([line 47](../../env_actor/auto/inference_algorithms/rtc/rtc_actor.py#L47)) for the SHM block, lock, Conditions, Events, Values, and child Processes.

If you `fork` instead, the child inherits a half-initialized CUDA context from the parent (because the parent imports torch+CUDA before forking). The child's next `torch.cuda.*` call raises `Cannot re-initialize CUDA in forked subprocess`. `spawn` starts a fresh interpreter per child — same SHM blocks (because they are OS-level, not Python-level), but a clean CUDA context.

### Pinned tensors and tensor → SHM coordination

The `write_action_chunk_n_update_iter_val` method handles `torch.Tensor → np.ndarray` conversion explicitly ([lines 213-215](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_c/shm_manager_bridge.py#L213)):

```python
if isinstance(action_chunk, torch.Tensor):
    action_chunk = action_chunk.cpu().numpy()
np.copyto(self._shm_array_dict['action'], action_chunk.astype(np.float32, copy=False), casting='no')
```

The `np.copyto(..., casting='no')` is **strict**: it raises if the source dtype doesn't match the destination. This is intentional — silent dtype coercion would mask a policy producing `float64` outputs onto a `float32` SHM block. Make sure your policy's `predict` returns `float32`.

The bridge does **not** use `torch.tensor.pin_memory()` — observations and actions move between CPU SHM and the GPU only inside the inference process, which controls its own pinning via the policy's loader. See `docs/rtc_shared_memory.md` for the higher-level discussion.

## How the data manager is selected by the factory

### Sequential

[`sequential/data_manager/data_manager_interface.py:1-8`](../../env_actor/auto/inference_algorithms/sequential/data_manager/data_manager_interface.py#L1):

```python
class DataManagerInterface:
    def __init__(self, runtime_params, robot):
        self.data_manager_bridge = None
        if robot == "igris_b":
            from .robots.igris_b.data_manager_bridge import DataManagerBridge
        elif robot == "igris_c":
            from .robots.igris_c.data_manager_bridge import DataManagerBridge
        self.data_manager_bridge = DataManagerBridge(runtime_params=runtime_params)
```

Constructed in `SequentialActor.__init__` ([line 72](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L72)).

### RTC

[`rtc/data_manager/shm_manager_interface.py:17-32`](../../env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py#L17):

```python
class SharedMemoryInterface:
    def __init__(self, robot, shm_specs, lock, control_iter_cond, ...):
        if robot == "igris_b":
            from .robots.igris_b.shm_manager_bridge import SharedMemoryManager
        elif robot == "igris_c":
            from .robots.igris_c.shm_manager_bridge import SharedMemoryManager
        self.shm_manager = SharedMemoryManager.attach_from_specs(shm_specs=shm_specs, ...)
```

Constructed in both `control_loop.start_control` ([line 57](../../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L57)) and `inference_loop.start_inference` ([line 82](../../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L82)) inside their respective child processes — each child attaches independently by name. The parent (`RTCActor`) creates the SHM blocks but does **not** instantiate a `SharedMemoryManager` itself; it only passes the specs.

Same pattern in the local variant — see [`rtc_local/data_manager/shm_manager_interface.py:17-32`](../../env_actor/auto/inference_algorithms/rtc_local/data_manager/shm_manager_interface.py#L17).

## `rtc/` vs `rtc_local/` — what differs

Both subtrees implement the same algorithm with the same SHM layout. The only differences are:

| File pair | What differs in the `_local` copy |
|---|---|
| `rtc_local_actor.py` vs `rtc_actor.py` | No `@ray.remote` decorator; instantiated directly by `run_inference_local.py`. |
| `rtc_local/actors/control_loop.py` vs `rtc/actors/control_loop.py` | No `import ray` / `ray.init()` block at the top. Outer per-episode loop collapsed into a single-episode flow (see [02](02_changes_vs_main.md#env_actorautoinference_algorithmsrtc_local-and-the-12-other-files-under-rtc_local-added--13-files)). |
| `rtc_local/actors/inference_loop.py` vs `rtc/actors/inference_loop.py` | No `import ray` / `ray.init()`. |
| `rtc_local/data_manager/**` vs `rtc/data_manager/**` | **Byte-identical files** (`shm_manager_interface.py`, `shm_manager_bridge.py` per robot, `utils/*`). Maintained as separate copies. |

`TODO:` factor the data-manager subtrees into a single shared module so they cannot drift. The `rtc/` and `rtc_local/` actor and `actors/*` files have legitimate small differences (ray vs. no-ray), but the data-manager copies have none.

---

← Back to index: [README.md](README.md) · Next → [06_normalization_igris_c.md](06_normalization_igris_c.md)
