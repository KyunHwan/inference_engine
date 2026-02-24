# env_actor

The core inference runtime package. Orchestrates policy inference, robot communication, data normalization, and real-time control for physical robot deployment.

## Architecture

```
run_inference.py
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  auto/inference_algorithms/                              │
│  RTCActor (dual-process) or SequentialActor (sync)       │
│  ─ loads policy, creates controller, runs control loop   │
└──────────┬───────────────┬───────────────┬───────────────┘
           │               │               │
     ┌─────▼─────┐  ┌─────▼─────┐  ┌──────▼──────┐
     │  policy/   │  │ robot_io  │  │ nom_stats   │
     │  load &    │  │ interface │  │ manager     │
     │  predict   │  │ ROS2 I/O  │  │ normalize   │
     └───────────┘  └───────────┘  └─────────────┘
```

## Subdirectories

| Directory | Purpose |
|---|---|
| `policy/` | Policy protocol definition, registry, YAML-based loader, and concrete policy implementations |
| `auto/` | Inference algorithm implementations (Sequential and RTC) |
| `robot_io_interface/` | Hardware abstraction — ROS2 communication, camera capture, action publishing |
| `runtime_settings_configs/` | Per-robot JSON/Python configuration (timing, dimensions, ROS2 topics) |
| `nom_stats_manager/` | Data normalization and denormalization using pre-computed statistics |
| `inference_engine_utils/` | Shared utilities — action inpainting weight computation |

## How the Pieces Connect

Starting from `run_inference.py`:

1. **Ray actor creation** — `RTCActor` or `SequentialActor` is created as a Ray remote actor with GPU resources
2. **Inside the actor's `start()` method:**
   - `build_policy(yaml_path)` loads the policy from YAML config → model factory → checkpoint → registry
   - `ControllerInterface(robot=...)` creates the robot-specific ROS2 bridge
   - `DataNormalizationInterface(robot=...)` creates the normalization bridge using pre-computed stats
   - A data manager (per-algorithm, per-robot) handles observation history and action buffering
3. **Control loop** runs at the configured frequency (default 20 Hz):
   - Controller reads robot state (proprioception + camera images)
   - Data manager updates observation history buffers
   - Every `policy_update_period` steps: policy runs inference
   - Actions are denormalized and published with slew-rate limiting

## Extension Points

- **New policy**: Add to `policy/policies/` — implement the `Policy` protocol and register
- **New robot**: Add bridges in `robot_io_interface/robots/`, data managers in `auto/inference_algorithms/*/data_manager/robots/`, normalization in `nom_stats_manager/robots/`, and configs in `runtime_settings_configs/robots/`
- **New algorithm**: Add to `auto/inference_algorithms/` — create a Ray actor with a `start()` method
