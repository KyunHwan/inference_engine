# Documentation

The complete documentation set for the inference engine. Start at the top if you are new; jump by task if you know what you need.

## Table of contents

- [Start here](#start-here)
- [By task](#by-task)
- [By module](#by-module)
- [Walkthroughs](#walkthroughs)
- [Reference](#reference)
- [Submodule documentation](#submodule-documentation)
- [Conventions](#conventions)

## Start here

If this is your first time in the repo, read in this order:

1. [Onboarding](onboarding.md) — day-by-day plan for your first week. Includes commands, expected outputs, and pointers to deeper material.
2. [Concepts](concepts.md) — vocabulary and mental model. VLA, action chunks, RTC, flow matching, shared memory, Ray actors, spawn vs fork, ROS2 primer.
3. [First run](walkthroughs/01_first_run.md) — end-to-end "Hello, robot".
4. [Trace one step](walkthroughs/02_trace_one_step.md) — line-by-line inspection of one control-loop iteration.
5. [Architecture](architecture.md) — system topology, process model, shutdown lifecycle.

## By task

| I want to… | Read |
|---|---|
| Spend my first day in the repo | [onboarding.md](onboarding.md) |
| Build a mental model | [concepts.md](concepts.md) |
| Run my first inference | [walkthroughs/01_first_run.md](walkthroughs/01_first_run.md) |
| Trace one control step | [walkthroughs/02_trace_one_step.md](walkthroughs/02_trace_one_step.md) |
| Add a new policy | [walkthroughs/03_add_a_new_policy.md](walkthroughs/03_add_a_new_policy.md) |
| Add a new robot | [walkthroughs/04_add_a_new_robot.md](walkthroughs/04_add_a_new_robot.md) |
| Understand the RTC shared-memory design | [rtc_shared_memory.md](rtc_shared_memory.md) |
| Modify configuration safely | [configuration_cookbook.md](configuration_cookbook.md) |
| Debug a crash | [troubleshooting.md](troubleshooting.md) |
| Find an answer to a common question | [faq.md](faq.md) |
| Look up a function or shape | [api.md](api.md) |
| Look up a term | [glossary.md](glossary.md) |
| Set up dev environment / commit conventions | [development.md](development.md) |

## By module

Folder READMEs live next to the code they document. Open them when you need to know what a directory contains and what it depends on.

| Module | README |
|---|---|
| Top-level package | [../env_actor/README.md](../env_actor/README.md) |
| Inference algorithms (RTC + Sequential) | [../env_actor/auto/README.md](../env_actor/auto/README.md) · [../env_actor/auto/inference_algorithms/README.md](../env_actor/auto/inference_algorithms/README.md) |
| RTC algorithm | [../env_actor/auto/inference_algorithms/rtc/README.md](../env_actor/auto/inference_algorithms/rtc/README.md) · [actors](../env_actor/auto/inference_algorithms/rtc/actors/README.md) · [data_manager](../env_actor/auto/inference_algorithms/rtc/data_manager/README.md) · [utils](../env_actor/auto/inference_algorithms/rtc/data_manager/utils/README.md) · [IGRIS_B bridge](../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/README.md) |
| Sequential algorithm | [../env_actor/auto/inference_algorithms/sequential/README.md](../env_actor/auto/inference_algorithms/sequential/README.md) · [data_manager](../env_actor/auto/inference_algorithms/sequential/data_manager/README.md) · [IGRIS_B bridge](../env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_b/README.md) |
| Policy package | [../env_actor/policy/README.md](../env_actor/policy/README.md) · [policies](../env_actor/policy/policies/README.md) · [registry](../env_actor/policy/registry/README.md) · [templates](../env_actor/policy/templates/README.md) · [utils](../env_actor/policy/utils/README.md) |
| Concrete policies | [openpi_policy](../env_actor/policy/policies/openpi_policy/README.md) · [dsrl_openpi_policy](../env_actor/policy/policies/dsrl_openpi_policy/README.md) |
| Robot I/O | [../env_actor/robot_io_interface/README.md](../env_actor/robot_io_interface/README.md) · [IGRIS_B](../env_actor/robot_io_interface/robots/igris_b/README.md) · [IGRIS_C](../env_actor/robot_io_interface/robots/igris_c/README.md) |
| Runtime configs | [../env_actor/runtime_settings_configs/README.md](../env_actor/runtime_settings_configs/README.md) · [IGRIS_B](../env_actor/runtime_settings_configs/robots/igris_b/README.md) · [IGRIS_C](../env_actor/runtime_settings_configs/robots/igris_c/README.md) |
| Normalization | [../env_actor/nom_stats_manager/README.md](../env_actor/nom_stats_manager/README.md) · [IGRIS_B](../env_actor/nom_stats_manager/robots/igris_b/README.md) |
| Shared utilities | [../env_actor/inference_engine_utils/README.md](../env_actor/inference_engine_utils/README.md) |

## Walkthroughs

[walkthroughs/](walkthroughs/README.md) contains task-oriented end-to-end guides. Each names its prerequisites and estimated time.

## Reference

- [api.md](api.md) — public interfaces (Policy protocol, `build_policy`, `ControllerInterface`, `DataNormalizationInterface`, `RuntimeParams`, action inpainting helpers). Shapes table is at the top.
- [glossary.md](glossary.md) — alphabetized definitions for every domain-specific term.
- [development.md](development.md) — dev environment, naming conventions, design patterns, submodule update workflow.

## Submodule documentation

The `trainer/` directory is a git submodule with its own documentation set. It is **read-only** from this repo — never edit it here; make changes upstream in the [trainer repo](https://github.com/KyunHwan/trainer) and bump the pinned SHA.

Pinned at commit `3ca051a256c9068f77b556df98f538d9a6185ccf`:

- [trainer/README.md](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/README.md) — overview and architecture
- [trainer/docs/README.md](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/docs/README.md) — documentation hub
- [docs/01_getting_started.md](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/docs/01_getting_started.md)
- [docs/04_concepts.md](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/docs/04_concepts.md) — Trainer's "what every term means" (overlaps somewhat with our concepts.md but is training-side focused)
- [docs/05_configuration.md](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/docs/05_configuration.md)
- [docs/07_extending.md](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/docs/07_extending.md)
- [docs/10_troubleshooting.md](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/docs/10_troubleshooting.md)
- [docs/12_glossary.md](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/docs/12_glossary.md)

`trainer/policy_constructor/` is a nested git submodule, pinned at commit `00663cc10c91d7614c1a0ea3d68629c38767b167`:

- [policy_constructor/README.md](https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/README.md)
- [docs/INDEX.md](https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/docs/INDEX.md)
- [docs/QUICKSTART.md](https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/docs/QUICKSTART.md)
- [docs/MENTAL_MODEL.md](https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/docs/MENTAL_MODEL.md)
- [docs/GLOSSARY.md](https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/docs/GLOSSARY.md)
- [docs/TROUBLESHOOTING.md](https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/docs/TROUBLESHOOTING.md)

## Conventions

- Folder READMEs follow a fixed shape: a one-line summary, a `**Parent:** [link]` line, then Purpose → Key files → Contracts → How it plugs in → Extension points → Cross-links.
- Every `docs/*.md` file has a TOC at the top.
- Code references use repo-relative links: `[loader.py](../env_actor/policy/utils/loader.py)`.
- Submodule references use absolute GitHub URLs pinned to the SHA recorded in `.gitmodules`.
- Diagrams are ASCII by default. Some pages add Mermaid versions inside `<details>` blocks for visual readers.
- A `TODO(verify)` marker means the author could not confirm a value from code and the reader should treat it as a hypothesis.
