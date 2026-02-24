# Inference Algorithms

Two inference algorithms that orchestrate the end-to-end control loop. Both are implemented as Ray remote actors and share the same policy, controller, and normalization interfaces.

## Algorithm Comparison

| Aspect | Sequential | RTC (Real-Time Control) |
|---|---|---|
| **Architecture** | Single-threaded, synchronous | Dual-process with shared memory |
| **Policy method** | `predict()` | `guided_inference()` (with action inpainting) |
| **Ray actor** | `SequentialActor` | `RTCActor` |
| **Resources** | CPU only | 3 CPUs + 1 GPU |
| **Use case** | Debugging, testing | Real-time robot deployment |
| **Action transitions** | Hard switch between chunks | Smooth blending via inpainting weights |

## Sequential Algorithm

**Location:** `sequential/`

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

**Key files:**
- `sequential_actor.py` — Ray actor that calls `start()` to begin the loop. Loads policy, creates controller, runs at configured HZ. Policy inference happens every `policy_update_period` steps.
- `data_manager_interface.py` — Factory that creates robot-specific data managers based on the `robot` string.

**Control loop logic:**
1. Controller reads state (proprio + images)
2. Data manager updates observation history buffers
3. Every N steps: `policy.predict(obs)` returns an action chunk
4. Data manager buffers the chunk and returns the current action via simple indexing (`action = chunk[offset]`)
5. Controller publishes the action with slew-rate limiting

## RTC Algorithm

**Location:** `rtc/`

Dual-process architecture where control and inference run concurrently, communicating via shared memory. This allows the robot to continue executing actions from the previous chunk while the policy computes the next one.

```
rtc/
├── rtc_actor.py                         # Ray remote actor (creates both processes)
├── actors/
│   ├── control_loop.py                  # Process 1: 20 Hz robot control
│   └── inference_loop.py               # Process 2: policy inference
└── data_manager/
    ├── shm_manager_interface.py         # Factory for shared memory managers
    ├── utils/                           # Shared memory utilities
    └── robots/
        ├── igris_b/shm_manager_bridge.py
        └── igris_c/                     # Stub
```

**Key files:**
- `rtc_actor.py` — Creates both processes and the shared memory manager. Calls `start()` to launch.
- `control_loop.py` — Runs at the configured HZ. Reads robot state, writes to shared memory, reads actions from shared memory, publishes to robot.
- `inference_loop.py` — Waits for a minimum number of actions to be executed, then calls `policy.guided_inference()`. Writes the new action chunk to shared memory.
- `shm_manager_bridge.py` — Manages shared memory arrays and synchronization primitives.

**Synchronization primitives:**
- `RLock` — protects shared memory reads/writes
- `Condition` (2) — `control_iter_cond` signals the inference loop; `inference_ready_cond` signals the control loop
- `Event` (2) — `stop_event` for shutdown; `episode_complete_event` for episode boundaries
- `Value` — `num_control_iters` counter; `inference_ready_flag` boolean

**Shared memory layout** (IGRIS_B defaults):

| Key | Shape | Dtype |
|---|---|---|
| `proprio` | `(1, 24)` | float32 |
| `head` | `(1, 3, 240, 320)` | uint8 |
| `left` | `(1, 3, 240, 320)` | uint8 |
| `right` | `(1, 3, 240, 320)` | uint8 |
| `action` | `(50, 24)` | float32 |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  ALGORITHM LAYER (Robot-Agnostic)                   │
│  SequentialActor / RTCActor - Control orchestration  │
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

## Design Decisions

- **No temporal ensemble** — action selection uses simple indexing (`action = chunk[offset]`) rather than weighted averaging across overlapping chunks
- **No human-in-the-loop gating** — actors start immediately when `start()` is called
- **Factory pattern for robot bridges** — robot-specific code is isolated; algorithms are robot-agnostic
- **Duck-typing** — bridge classes satisfy interfaces via structural subtyping, not inheritance

## Adding a New Algorithm

1. Create a directory under `inference_algorithms/your_algorithm/`
2. Implement a Ray remote actor class with a `start()` method
3. Create robot-specific data manager bridges under `data_manager/robots/`
4. The actor must:
   - Load the policy via `build_policy()`
   - Create a `ControllerInterface` for the target robot
   - Create a `DataNormalizationInterface` for normalization
   - Run a control loop that reads state, runs inference, and publishes actions
5. Update `run_inference.py` to add the new algorithm name to `--inference_algorithm` choices
