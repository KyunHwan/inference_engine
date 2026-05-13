# auto

**Parent:** [env_actor](../README.md)

Inference algorithm implementations. Each algorithm is a Ray actor that orchestrates the full control loop — reading robot state, running policy inference, publishing actions.

## Table of contents

- [Purpose](#purpose)
- [Structure](#structure)
- [Algorithms](#algorithms)
- [Extension points](#extension-points)
- [Related docs](#related-docs)

## Purpose

`auto/` is the "robot-agnostic control loop" layer. Algorithms here own:

- The Ray actor class (`@ray.remote`).
- The loop that drives `read_state → buffer → policy → buffer → publish_action`.
- Algorithm-specific concurrency (single-threaded vs dual-process).

Robot-specific I/O and normalization sit *under* this layer — they live in [robot_io_interface/](../robot_io_interface/README.md) and [nom_stats_manager/](../nom_stats_manager/README.md). The factory pattern keeps algorithms robot-agnostic.

## Structure

```
auto/
└── inference_algorithms/
    ├── rtc/          # Real-Time Control (dual-process, shared memory)
    └── sequential/   # Single-threaded inference
```

## Algorithms

| Algorithm | Actor | Policy method | Use case |
|---|---|---|---|
| [Sequential](inference_algorithms/sequential/README.md) | `SequentialActor` | `predict()` | Debugging, testing, no-latency-budget scenarios |
| [RTC](inference_algorithms/rtc/README.md) | `RTCActor` | `guided_inference()` | Real-time robot deployment |

Detailed comparison: [docs/architecture.md § RTC vs Sequential, detailed](../../docs/architecture.md#rtc-vs-sequential-detailed).

## Extension points

To add a new algorithm:

1. Create a subdirectory under [inference_algorithms/](inference_algorithms/README.md).
2. Implement a Ray remote actor with a `start()` method.
3. Add robot-specific data manager bridges under your algorithm's `data_manager/robots/<robot>/`.
4. Wire it into [run_inference.py](../../run_inference.py) — add the algorithm name to `--inference_algorithm` choices and add the actor creation branch.

Reference patterns: [`SequentialActor.start`](inference_algorithms/sequential/sequential_actor.py) for the synchronous case, [`RTCActor.start`](inference_algorithms/rtc/rtc_actor.py) for the multi-process case.

## Related docs

- [docs/architecture.md](../../docs/architecture.md) — system topology and process model.
- [docs/concepts.md § Sequential vs RTC](../../docs/concepts.md#sequential-vs-rtc) — the latency-vs-complexity trade-off.
- [docs/walkthroughs/02_trace_one_step.md](../../docs/walkthroughs/02_trace_one_step.md) — both algorithms, one step each, side by side.
