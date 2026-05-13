# rtc/data_manager

**Parent:** [rtc](../README.md)

The shared-memory layer that lets the control and inference children communicate. A robot-agnostic interface forwards every call to a robot-specific bridge.

## Table of contents

- [Files](#files)
- [Contracts](#contracts)
- [Subdirectories](#subdirectories)
- [Related docs](#related-docs)

## Files

| File | Purpose |
|---|---|
| [`shm_manager_interface.py`](shm_manager_interface.py) | `SharedMemoryInterface` — robot factory. Picks the per-robot `SharedMemoryManager` based on a string. |

## Contracts

The interface forwards every call to the bridge — see [docs/api.md § SharedMemoryInterface (RTC only)](../../../../../docs/api.md#sharedmemoryinterface-rtc-only) for the surface.

Notable invariants enforced inside the bridges:

- All SHM mutations happen inside `with self._lock:` blocks.
- `num_control_iters` is the action index. Control increments it on each step; inference decrements it by `executed` when committing a new chunk.
- `atomic_read_for_inference` returns *copies* — the control loop can keep writing while inference computes.
- `prev_action` is the un-executed tail of `action`, zero-padded to `action_chunk_size`.

## Subdirectories

| Directory | Purpose |
|---|---|
| [`utils/`](utils/README.md) | `ShmArraySpec`, `create_shared_ndarray`, `attach_shared_ndarray`, `MaxDeque`. |
| [`robots/igris_b/`](robots/igris_b/README.md) | IGRIS_B `SharedMemoryManager` — all atomic ops + lifecycle. |
| `robots/igris_c/` | Stub — `NotImplementedError` until specs land. |

## Related docs

- [docs/rtc_shared_memory.md](../../../../../docs/rtc_shared_memory.md) — full layout, sync primitives, sequence diagram, failure modes.
- [docs/api.md § SharedMemoryInterface (RTC only)](../../../../../docs/api.md#sharedmemoryinterface-rtc-only) — method-by-method API reference.
