# rtc/data_manager/utils

**Parent:** [data_manager](../README.md)

Algorithm-agnostic utilities used by the RTC shared-memory layer.

## Files

| File | What it provides |
|---|---|
| [`shared_memory_utils.py`](shared_memory_utils.py) | `ShmArraySpec` dataclass (name + shape + dtype string); `create_shared_ndarray(shape, dtype, zero=True)` (parent process); `attach_shared_ndarray(spec_dict, unregister=True)` (child processes). |
| [`max_deque.py`](max_deque.py) | `MaxDeque(buffer_len=5)` — sliding-window max over recent inference latencies. `add(int)`, `max() -> int`, `clear()`. Used by the IGRIS_B `SharedMemoryManager` to compute `est_delay`. |

## How they're used

`RTCActor.start` calls `create_shared_ndarray` once per region (parent process). The returned `ShmArraySpec` is passed by reference into both children via `mp.Process(args=...)`. Each child then calls `attach_shared_ndarray(specs)` to map the same OS-level block into its own address space.

`attach_shared_ndarray` does `resource_tracker.unregister(...)` so only the parent's tracker bookkeeps the block — the parent unlinks on cleanup; the children just `close()` their views.

`MaxDeque.max()` is used inside `atomic_read_for_inference` as `est_delay`. A larger delay extends the "keep old" zone in [`compute_guided_prefix_weights`](../../../../../inference_engine_utils/action_inpainting.py), so action inpainting hedges more toward the previous chunk.

## Related docs

- [docs/rtc_shared_memory.md § Shared-memory layout](../../../../../../docs/rtc_shared_memory.md#shared-memory-layout)
- [docs/glossary.md § MaxDeque](../../../../../../docs/glossary.md#maxdeque)
