# inference_algorithms

**Parent:** [auto](../README.md)

The two implementations: [`sequential/`](sequential/README.md) and [`rtc/`](rtc/README.md). Both are Ray remote actors that share the same policy, controller, and normalization interfaces; they differ only in *when* and *how* inference fires relative to control.

## Table of contents

- [Algorithm comparison](#algorithm-comparison)
- [Sequential algorithm](#sequential-algorithm)
- [RTC algorithm](#rtc-algorithm)
- [Shared interfaces (used by both)](#shared-interfaces-used-by-both)
- [Design decisions](#design-decisions)
- [Extension points](#extension-points)
- [Related docs](#related-docs)

## Algorithm comparison

| Aspect | Sequential | RTC (Real-Time Control) |
|---|---|---|
| Architecture | Single-threaded, synchronous | Dual-process with shared memory |
| Policy method | `predict()` | `guided_inference()` (with action inpainting) |
| Ray actor | `SequentialActor` (`@ray.remote(num_gpus=1)`) | `RTCActor` (`@ray.remote(num_gpus=1, num_cpus=3)`) |
| Resource request | `resources={"inference_pc": 1}` | `resources={"inference_pc": 1}` |
| Inference trigger | Every `policy_update_period` control steps | When `num_control_iters ≥ min_num_actions_executed (= 35)` |
| Action transitions | Hard switch between chunks | Smooth blending via inpainting weights |
| Use case | Debugging, testing | Real-time robot deployment |

## Sequential algorithm

**Location**: [sequential/](sequential/README.md)

Single-threaded control loop. The policy runs synchronously — the control loop blocks during inference.

```
sequential/
├── sequential_actor.py                  # Ray remote actor (entrypoint)
└── data_manager/
    ├── data_manager_interface.py        # Factory for robot-specific data managers
    └── robots/
        ├── igris_b/data_manager_bridge.py
        └── igris_c/                     # Stub
```

**Key files**:

- [`sequential_actor.py`](sequential/sequential_actor.py) — Ray actor whose `start()` runs the outer-`while True:` / inner-`for t in range(9000):` loop. Policy fires every `policy_update_period` steps.
- [`data_manager_interface.py`](sequential/data_manager/data_manager_interface.py) — Factory that creates the robot-specific data manager.

**Control-loop logic** (one iteration):

1. Controller `read_state` → raw observations.
2. Data manager `update_state_history` → FIFO-shift proprio, replace image.
3. Every N steps: `policy.predict()` → action chunk; `data_manager.buffer_action_chunk(chunk, t)`.
4. `action = data_manager.get_current_action(t)` → index into the cached chunk.
5. Controller `publish_action(action, prev_joint)` with slew-rate limiting.

## RTC algorithm

**Location**: [rtc/](rtc/README.md)

Dual-process: a control loop (20 Hz) and an inference loop run concurrently via shared memory. The control loop never blocks; the robot keeps executing previous-chunk actions while a new chunk is being computed.

```
rtc/
├── rtc_actor.py                         # Parent actor — creates SHM + spawns 2 child processes
├── actors/
│   ├── control_loop.py                  # Process 1: 20 Hz robot control
│   └── inference_loop.py                # Process 2: policy inference
└── data_manager/
    ├── shm_manager_interface.py         # Factory for robot-specific SHM managers
    ├── utils/                           # SHM creation + max-deque helpers
    └── robots/
        ├── igris_b/shm_manager_bridge.py
        └── igris_c/                     # Stub
```

**Synchronization primitives** (created in [`RTCActor.start`](rtc/rtc_actor.py)):

- `RLock` — guards atomic reads/writes
- `Condition` × 2 — `control_iter_cond` (control → inference), `inference_ready_cond` (inference → control)
- `Event` × 2 — `stop_event` (shutdown), `episode_complete_event` (episode boundary)
- `Value` × 2 — `num_control_iters` (counter), `inference_ready_flag` (bool)

**Shared-memory layout (IGRIS_B defaults)**:

| Key | Shape | Dtype | Writer |
|---|---|---|---|
| `proprio` | `(50, 24)` | float32 | control |
| `head` | `(1, 3, 240, 320)` | uint8 | control |
| `left` | `(1, 3, 240, 320)` | uint8 | control |
| `right` | `(1, 3, 240, 320)` | uint8 | control |
| `action` | `(50, 24)` | float32 | inference |

Deep dive: [docs/rtc_shared_memory.md](../../../docs/rtc_shared_memory.md).

## Shared interfaces (used by both)

```
┌─────────────────────────────────────────────────────┐
│  ALGORITHM LAYER (Robot-Agnostic)                   │
│  SequentialActor / RTCActor — control orchestration │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│  INTERFACE LAYER                                    │
│  build_policy()        → Policy inference           │
│  ControllerInterface   → Robot I/O                  │
│  DataNormalization     → State/action transforms    │
│  DataManager           → History + action buffering │
└─────────────────────────────────────────────────────┘
```

## Design decisions

- **No temporal ensemble.** Action selection uses simple indexing (`action = chunk[clip(t - last_t, 0, K-1)]` or `chunk[num_control_iters - 1]`) rather than weighted averaging across overlapping chunks. RTC blends only at chunk-boundary transitions via action inpainting.
- **No human-in-the-loop gating.** Actors start immediately when `start.remote()` is called. RTC adds a `t > 100` warmup gate before publishing, but there is no operator confirmation.
- **Factory pattern for robot bridges.** Robot-specific code is isolated; algorithms are robot-agnostic.
- **Duck-typing.** Bridge classes satisfy interfaces structurally (no inheritance).

## Extension points

To add a new algorithm:

1. Create a directory under `inference_algorithms/your_algorithm/`.
2. Implement a Ray remote actor class with a `start()` method.
3. Create robot-specific data manager bridges under `data_manager/robots/`.
4. The actor must:
   - Load the policy via [`build_policy()`](../../policy/utils/loader.py).
   - Create a [`ControllerInterface`](../../robot_io_interface/controller_interface.py) for the target robot.
   - Create a [`DataNormalizationInterface`](../../nom_stats_manager/data_normalization_interface.py).
   - Run a control loop that reads state, runs inference, and publishes actions.
5. Update [run_inference.py](../../../run_inference.py) — add the algorithm name to `--inference_algorithm` choices and add the actor creation branch.

## Related docs

- [docs/architecture.md](../../../docs/architecture.md)
- [docs/rtc_shared_memory.md](../../../docs/rtc_shared_memory.md)
- [docs/walkthroughs/02_trace_one_step.md](../../../docs/walkthroughs/02_trace_one_step.md)
