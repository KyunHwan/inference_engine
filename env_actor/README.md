# env_actor

**Parent:** [inference_engine root](../README.md)

The core inference runtime package. Orchestrates policy inference, robot I/O, data normalization, and the real-time control loop for physical-robot deployment.

## Table of contents

- [Purpose](#purpose)
- [Subdirectories](#subdirectories)
- [How the pieces connect](#how-the-pieces-connect)
- [Extension points](#extension-points)
- [Related docs](#related-docs)

## Purpose

Everything that runs at inference time lives here. The training framework lives in [trainer/](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/README.md) (a read-only submodule); this directory consumes its outputs (checkpoints, model YAMLs, registry).

## Subdirectories

| Directory | Purpose |
|---|---|
| [policy/](policy/README.md) | `Policy` protocol, registry, YAML loader, concrete policy implementations |
| [auto/](auto/README.md) | Inference algorithms — Sequential (single-threaded) and RTC (dual-process) |
| [robot_io_interface/](robot_io_interface/README.md) | Robot hardware abstraction — ROS2 publishers/subscribers, camera capture, slew-rate-limited action publishing |
| [runtime_settings_configs/](runtime_settings_configs/README.md) | Per-robot JSON + Python configuration (timing, dimensions, ROS2 topics, initial joint pose) |
| [nom_stats_manager/](nom_stats_manager/README.md) | Z-score normalization for proprioception, scale-to-[0,1] for images, denormalization for actions |
| [inference_engine_utils/](inference_engine_utils/README.md) | Shared utilities — action inpainting weight computation |

## How the pieces connect

Starting from [run_inference.py](../run_inference.py):

1. **Ray actor creation** — `RTCActor` or `SequentialActor` is created as a Ray remote actor with `resources={"inference_pc": 1}`.
2. **Inside the actor's `start()` method**:
   - `build_policy(yaml_path)` ([policy/utils/loader.py](policy/utils/loader.py)) loads the policy: YAML → model factory → checkpoint → registry.
   - `ControllerInterface(robot=...)` ([robot_io_interface/controller_interface.py](robot_io_interface/controller_interface.py)) wires up the robot-specific ROS2 bridge.
   - `DataNormalizationInterface(robot=...)` ([nom_stats_manager/data_normalization_interface.py](nom_stats_manager/data_normalization_interface.py)) wires up the normalization bridge using stats loaded by `RuntimeParams.read_stats_file()`.
   - A data manager (per algorithm × robot) owns observation history and action buffering.
3. **Control loop** runs at the configured `HZ` (default 20):
   - Controller reads robot state (proprio + camera images).
   - Data manager updates observation history.
   - Every `policy_update_period` steps (Sequential) or whenever `num_control_iters ≥ min_num_actions_executed` (RTC): policy runs inference.
   - Actions are denormalized and published with slew-rate limiting.

Full data flow per step: [docs/architecture.md § Data flow per control step](../docs/architecture.md#data-flow-per-control-step).

## Extension points

- **New policy** → [policy/policies/](policy/policies/README.md). Walkthrough: [docs/walkthroughs/03_add_a_new_policy.md](../docs/walkthroughs/03_add_a_new_policy.md).
- **New robot** → bridges in [robot_io_interface/robots/](robot_io_interface/README.md), data managers in [auto/inference_algorithms/*/data_manager/robots/](auto/inference_algorithms/README.md), normalization in [nom_stats_manager/robots/](nom_stats_manager/README.md), configs in [runtime_settings_configs/robots/](runtime_settings_configs/README.md). Walkthrough: [docs/walkthroughs/04_add_a_new_robot.md](../docs/walkthroughs/04_add_a_new_robot.md).
- **New algorithm** → [auto/inference_algorithms/](auto/inference_algorithms/README.md). Reference patterns: `RTCActor`, `SequentialActor`.

## Related docs

- [docs/onboarding.md](../docs/onboarding.md) — read first if you are new.
- [docs/concepts.md](../docs/concepts.md) — mental model.
- [docs/architecture.md](../docs/architecture.md) — system topology and data flow.
- [docs/api.md](../docs/api.md) — public interfaces.
- [docs/glossary.md](../docs/glossary.md) — vocabulary.
