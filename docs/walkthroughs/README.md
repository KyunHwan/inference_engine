# Walkthroughs

Task-oriented end-to-end guides. Each one names its prerequisites and the time it should take.

## Table of contents

- [01 — First run](#01--first-run)
- [02 — Trace one control step](#02--trace-one-control-step)
- [03 — Add a new policy](#03--add-a-new-policy)
- [04 — Add a new robot](#04--add-a-new-robot)

## 01 — First run

[01_first_run.md](01_first_run.md) · ~45 minutes (longer if hardware setup is new)

Install dependencies, start Ray, launch `run_inference.py`. Includes annotated success-log output and "dry run / no robot" instructions for the cases where you don't have IGRIS_B in front of you.

**Prereqs**: dependencies installed ([onboarding § Day 1](../onboarding.md#day-1-clone-install-read)). GPU+ROS2 only if you want to actually invoke `predict`; otherwise read along.

## 02 — Trace one control step

[02_trace_one_step.md](02_trace_one_step.md) · ~60 minutes

Walks a single 50 ms iteration through every function it touches — for both Sequential and RTC, with every intermediate tensor's shape and dtype. End in a side-by-side diff so the differences pop out.

**Prereqs**: you've read [concepts.md](../concepts.md) and finished [01_first_run.md](01_first_run.md).

## 03 — Add a new policy

[03_add_a_new_policy.md](03_add_a_new_policy.md) · ~90 minutes

Copy `openpi_policy`, rename it to `my_policy`, change one method, register it, smoke-test. Explains the `Policy` protocol, the registry decorator, the YAML structure, and how `build_policy` finds your code.

**Prereqs**: you can run [01_first_run.md](01_first_run.md) to the warmup stage.

## 04 — Add a new robot

[04_add_a_new_robot.md](04_add_a_new_robot.md) · 1–2 working days (excluding hardware bring-up)

The longest walkthrough. Fill in the IGRIS_C stubs: runtime configs, controller bridge, two data manager bridges, normalization bridge, factory updates, argparse update. Ends with a verification checklist.

**Prereqs**: hardware specs for the new robot (joint count, DOF, ROS2 topics, cameras). If you don't have them yet, read [robots/igris_c/README.md](../../env_actor/robot_io_interface/robots/igris_c/README.md) for the spec list.
