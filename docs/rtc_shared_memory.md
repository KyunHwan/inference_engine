# RTC shared memory — deep dive

How the RTC algorithm coordinates two processes through OS shared memory. Read [concepts.md § Shared memory](concepts.md#shared-memory) first; this doc is the implementation-level companion.

## Table of contents

- [Why shared memory, not pipes or queues?](#why-shared-memory-not-pipes-or-queues)
- [Shared-memory layout](#shared-memory-layout)
- [Synchronization primitives](#synchronization-primitives)
- [Sequence — one full handshake](#sequence--one-full-handshake)
- [Process lifecycle](#process-lifecycle)
- [Why spawn, not fork](#why-spawn-not-fork)
- [Failure modes](#failure-modes)
- [Clean shutdown](#clean-shutdown)

## Why shared memory, not pipes or queues?

Per control step we move:

- 1 proprio vector: `24 × 4 bytes = 96 B`
- 3 camera frames: `3 × 3 × 240 × 320 × 1 = 691 KB`

Per inference call we move:

- A `(50, 24) float32` action chunk: `4.7 KB`

At 20 Hz the camera bandwidth is ~14 MB/s. `multiprocessing.Queue` would pickle every send and unpickle every receive — for 700 KB camera blobs that is real CPU. `Pipe` is the same (it uses pickle by default).

`multiprocessing.shared_memory.SharedMemory` is an OS-level RAM region two processes can `mmap`. Wrap it in a numpy array and read/write are **zero-copy** — the kernel just hands you the same physical pages. Pair it with locks/conditions for synchronization and you have an IPC primitive that scales to camera-frame rates with negligible CPU overhead.

## Shared-memory layout

Created in [`RTCActor.start`](../env_actor/auto/inference_algorithms/rtc/rtc_actor.py) using [`create_shared_ndarray`](../env_actor/auto/inference_algorithms/rtc/data_manager/utils/shared_memory_utils.py). Each block has a `ShmArraySpec(name, shape, dtype_str)` that the children use to re-attach.

| Region | Shape (IGRIS_B) | Dtype | Bytes | Writer | Reader | What it holds |
|---|---|---|---|---|---|---|
| `proprio` | `(50, 24)` | float32 | 4.7 KB | control | inference | FIFO buffer; newest proprio at `[0]` |
| `head` | `(1, 3, 240, 320)` | uint8 | 230 KB | control | inference | Latest head camera frame |
| `left` | `(1, 3, 240, 320)` | uint8 | 230 KB | control | inference | Latest left camera frame |
| `right` | `(1, 3, 240, 320)` | uint8 | 230 KB | control | inference | Latest right camera frame |
| `action` | `(50, 24)` | float32 | 4.7 KB | inference | control | Current action chunk |

Plus the synchronization primitives (themselves multiprocessing objects, passed by reference through `mp.spawn`):

- `lock: RLock`
- `control_iter_cond: Condition(lock)` and `inference_ready_cond: Condition(lock)`
- `stop_event: Event`, `episode_complete_event: Event`
- `num_control_iters: Value('i', 0, lock=False)`
- `inference_ready_flag: Value(c_bool, False, lock=False)`

Note that the `Value`s are `lock=False`: we don't want their built-in lock because we already hold `lock` externally. All mutations to these counters happen inside `with self._lock:` blocks.

## Synchronization primitives

Each primitive has one job. Mixing them is a source of bugs.

| Primitive | Protects / Signals | Acquired by | Notified by |
|---|---|---|---|
| `lock` (`RLock`) | All shared-memory reads/writes; counter mutations | Anyone doing an atomic op | (re-entrant, no notify) |
| `control_iter_cond` | "the control loop has stepped" / "the episode ended" / "stop" | Inference loop, in `wait_for_min_actions` | Control loop after each step (`notify_step`), or on episode-complete / stop |
| `inference_ready_cond` | "the inference loop is initialized and ready" | Control loop, in `wait_for_inference_ready` | Inference loop, in `set_inference_ready` / `set_inference_not_ready` |
| `stop_event` (`Event`) | Shutdown requested | All waiters check it in their `wait_for(...)` predicate | Anyone, via `signal_stop` (also notifies both conditions) |
| `episode_complete_event` (`Event`) | Episode finished | Inference's `wait_for_min_actions` checks it | Control's `signal_episode_complete` (sets + notifies `control_iter_cond`) |
| `num_control_iters` (`Value[int]`) | How many control steps since last inference | Read everywhere | Control's atomic-write op increments; inference's atomic-write op decrements by `executed` |
| `inference_ready_flag` (`Value[bool]`) | Initial-warmup handshake | Read by `wait_for_inference_ready` | Set/cleared by inference around each inference call |

## Sequence — one full handshake

The most important sequence to internalize: how a single inference cycle happens from "control just stepped" through "new chunk is in shared memory".

```
Time ─────────────────────────────────────────────────────────────────────────────→

Control                                       Inference
──────────                                    ──────────
read_state()                                  (waiting on wait_for_min_actions)
with lock:
  shift proprio history
  copy obs[head/left/right]
  num_control_iters += 1
  action = action[action_idx].copy()
  control_iter_cond.notify_all()  ────────►   wakes up, checks predicate
publish(action)                               (predicate: num_control_iters >= 35
                                                ? yes; return 'min_actions')
                                              set_inference_not_ready()
                                              with lock:                       [snapshot phase]
                                                copy proprio/head/left/right
                                                build prev_action
                                                est_delay = MaxDeque.max()
                                              (release lock; data is now local)
read_state()                                  with torch.inference_mode +
with lock:                                      autocast(bfloat16):            [compute phase]
  ... (same write+notify)                       guided_inference(input)
publish(action[i+1])                          (this can take 100s of ms)
read_state()
with lock:
  num_control_iters += 1
  ...                                         with lock:                       [commit phase]
publish(action[i+2])                            copy next_actions → SHM.action
                                                num_control_iters -= executed
                                                MaxDeque.add(new value)
                                              set_inference_ready()
                                              (loop back to wait_for_min_actions)
```

Key invariants:

1. **`num_control_iters` is the action index.** After the lock is taken in the control loop, `action[clip(num_control_iters - 1, 0, chunk_size-1)]` is the action to publish this step.
2. **When inference commits, it subtracts `executed`.** That re-anchors the counter to the start of the new chunk. The number of steps that actually went out during the inference call is the diff between the value at snapshot-phase and commit-phase.
3. **Reading from SHM always copies.** `atomic_read_for_inference` does `arr.copy()` for each region. The control loop is free to overwrite while inference computes.
4. **`prev_action` only holds the un-executed tail.** `prev_action[:k]` = `action[num_control_iters:]`; `prev_action[k:]` = zeros. This is what gets blended into the new chunk via [`compute_guided_prefix_weights`](../env_actor/inference_engine_utils/action_inpainting.py).

## Process lifecycle

```
ray.init                                      (driver process)
 │
RTCActor.options(...).remote(...)             (ray creates the actor process)
 │
actor.start.remote()                          (start() runs in the actor process)
 │
 ├── create SHM regions (5)                   ← parent is sole creator
 ├── create sync primitives
 ├── ctx = mp.get_context("spawn")
 ├── ctx.Process(target=start_inference) ──►  [InferenceLoop child]
 ├── ctx.Process(target=start_control)   ──►  [ControlLoop child]
 │                                              ↓ each child attaches to SHM by name
 │                                              ↓ each child resource_tracker.unregister
 │                                              ↓   (parent unlinks on shutdown)
 │
 ├── while any(p.is_alive() for p in procs):
 │     join with 0.5s timeout
 │     if a child exited != 0:
 │         stop_event.set()
 │         control_iter_cond.notify_all()
 │
 └── finally:
       stop_event.set()
       notify both conditions
       p.join(5) → p.terminate(); p.join(3)   ← reap children
       for each shm: close(); unlink(); resource_tracker.unregister()
```

The Ray actor process is **not** running the control loop itself. It is a supervisor for the two children. This matters when reading stack traces: a crash in the inference loop shows up in the *inference child's* stdout, not the actor's.

## Why spawn, not fork

Two reasons:

1. **CUDA.** The actor process is on the GPU worker and has already initialized CUDA (it imported `torch` and called `policy.warmup` indirectly). `fork` would copy that CUDA state into the children, which is not supported and would raise `Cannot re-initialize CUDA in forked subprocess` the moment the child tried to use the GPU.

2. **rclpy.** ROS2's Python bindings allocate native state at `rclpy.init()`. Forking that state across processes leads to memory corruption.

`spawn` re-imports modules in the child, including `torch` and `rclpy`, and they initialize cleanly. The cost is startup time (~1–2 s per child), which we pay once.

## Failure modes

### Orphaned `/dev/shm/` entries

If the parent process is `kill -9`'d (or its `finally:` cleanup is bypassed), the SHM blocks live on in `/dev/shm/`. They won't crash the next run (each run creates new uniquely-named blocks), but they consume RAM-backed disk. To clean up:

```bash
ls /dev/shm/         # look for psm_* / wnsm_* entries
rm /dev/shm/psm_*    # only if no other Python process is using them
```

`/dev/shm/` is a tmpfs — a reboot clears it.

### `resource_tracker: There appear to be N leaked shared_memory objects`

Python's `multiprocessing.resource_tracker` is paranoid. When a child process attaches to an SHM block, its tracker records the block; if the child dies without unlinking, the tracker warns at process exit.

We work around this by calling `resource_tracker.unregister(shm._name, "shared_memory")` in each child immediately after attaching, so only the parent's tracker is bookkeeping the block (and only the parent unlinks). If you still see the warning, an exception probably bypassed `cleanup()`. Search the child's stdout for an exception around the warning's timestamp.

### Deadlock: `wait_for_inference_ready` never returns

Either the inference process never reached `set_inference_ready()`, or `inference_ready_flag` was reset and not raised again. Diagnosis:

```bash
# In the Ray dashboard, find the inference child's logs.
# Look for the last "Warming up CUDA kernels..." or "Signaling inference ready..." line.
```

If you don't see "Signaling inference ready...", the inference process is stuck before that — likely in `build_policy` (e.g., loading a slow checkpoint) or `policy.warmup` (which can take 30+ s on first compile).

### Deadlock: `wait_for_min_actions` never returns

Either the control loop crashed without notifying, or it's somehow getting past `notify_step` without incrementing `num_control_iters`. The cleanup path in `start_control`'s `finally:` calls `shm_manager.cleanup()` which doesn't `signal_stop()`. If the control loop dies, the parent's join loop will eventually notice (`exitcode != 0`) and set `stop_event`, which wakes the inference loop with `result == 'stop'`. So you won't hang forever; you'll hang for up to 0.5 s × N (the join-loop timeout).

### `np.copyto(... casting='no')` raises `TypeError`

The bridge uses `casting='no'` to enforce that the SHM array's dtype and the incoming array's dtype are identical. If you change `proprio_state_dim` in the JSON but forget to regenerate `RuntimeParams` or restart, you can get a shape mismatch. Restart and re-read the JSON.

## Clean shutdown

The cleanup contract:

1. **Anyone may set `stop_event`** — the parent's `finally:`, a child's exception handler, or `Ctrl+C` propagated through the Ray actor.
2. **Anyone setting `stop_event` must also `notify_all()` both conditions** so any blocked waiter wakes up and sees `stop_event.is_set() == True`.
3. **`cleanup()` closes the local view** of each SHM block.
4. **Only the creator (`is_creator=True`) unlinks** the block. The default `attach_from_specs` creates non-creator managers; this is correct for children.
5. **The creator also `resource_tracker.unregister`s** to suppress the leak warning that would otherwise fire on parent exit.

If you ever add a new exit path, walk through this list and verify each step happens. The most common bug is "I exited the loop but didn't `notify_all()`, so a peer is still blocked on `wait_for(...)`".

## Code references

- [shared_memory_utils.py](../env_actor/auto/inference_algorithms/rtc/data_manager/utils/shared_memory_utils.py) — `ShmArraySpec`, `create_shared_ndarray`, `attach_shared_ndarray`.
- [max_deque.py](../env_actor/auto/inference_algorithms/rtc/data_manager/utils/max_deque.py) — `MaxDeque` for `est_delay`.
- [shm_manager_interface.py](../env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py) — robot factory.
- [shm_manager_bridge.py](../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py) — IGRIS_B implementation of all the atomic ops above.
- [rtc_actor.py](../env_actor/auto/inference_algorithms/rtc/rtc_actor.py) — parent process; SHM creation and child supervision.
- [control_loop.py](../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py) — control child.
- [inference_loop.py](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py) — inference child.
