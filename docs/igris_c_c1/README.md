# IGRIS_C / c1 — Documentation Hub

Branch-scoped documentation for the **`igris_c/c1`** branch of `inference_engine`. This folder is the first concrete implementation of the IGRIS_C robot platform — it turns the hardware-spec stub left in [`env_actor/robot_io_interface/robots/igris_c/README.md`](../../env_actor/robot_io_interface/robots/igris_c/README.md) on `main` into a runnable bridge, runtime config, data managers, and normalization layer.

If you are a junior engineer who just cloned the repo and was told "run inference on IGRIS_C," start at [01_quickstart_igris_c.md](01_quickstart_igris_c.md).

## Table of contents

- [What this folder is](#what-this-folder-is)
- [Why `igris_c/c1` exists](#why-igris_cc1-exists)
- [How `igris_c/c1` differs from `main`](#how-igris_cc1-differs-from-main)
- [How to read these docs](#how-to-read-these-docs)
- [Document index](#document-index)
- [Where to look in the existing docs](#where-to-look-in-the-existing-docs)
- [Submodule references](#submodule-references)
- [Conventions used here](#conventions-used-here)

## What this folder is

A self-contained doc set that lets a new engineer go from `git clone` to running inference on the IGRIS_C robot on the `igris_c/c1` branch, **without** modifying anything in the repo. Every file in this folder was written against the code on this branch — every fact is traceable to a source file and line. Where a project-specific value is brittle (an IP, a `/dev/...` path, a hostname, a checkpoint path), it is either labelled "current value on this branch — verify against your environment" or replaced with `<PLACEHOLDER>` and a `TODO`.

This folder lives at [`docs/igris_c_c1/`](.) deliberately: the name encodes the branch so future `igris_c/c2`, `igris_c/c3`, etc. can sit alongside it without overwriting. Nothing else in `docs/` was changed.

## Why `igris_c/c1` exists

`main` ships only the IGRIS_B robot. The IGRIS_C subtree on `main` is a checklist plus stubs:

- [`env_actor/robot_io_interface/robots/igris_c/README.md`](../../env_actor/robot_io_interface/robots/igris_c/README.md) lists the hardware specs IGRIS_C needs before it can run.
- [`env_actor/runtime_settings_configs/robots/igris_c/init_params.py`](../../env_actor/runtime_settings_configs/robots/igris_c/init_params.py) on `main` is empty (`INIT_JOINT_LIST = []`, `IGRIS_C_STATE_KEYS = []`).
- The sequential `DataManagerBridge` and the matching RTC `SharedMemoryManager` for IGRIS_C on `main` either do not exist or `raise NotImplementedError`.

`igris_c/c1` is the first branch that **fills in** that checklist. It:

- Implements an end-to-end Cyclone DDS controller bridge for the IGRIS_C body (31 joints) + hands (12 finger motors), with `cyclonedds-python` directly — bypassing `igris_c_sdk` because the SDK was built against a fork of `igris_c_msgs.idl` whose type hashes do not match the NUC firmware (see the docstring in [`controller_bridge.py:1-22`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py)).
- Adds runtime configuration (`init_params.py`, `inference_runtime_params.{json,py}`, `inference_runtime_topics.json`, plus two Cyclone DDS XML files).
- Adds matching IGRIS_C data managers under both `sequential/` and `rtc/`, plus brand-new `rtc_local/` and `sequential_local/` non-Ray variants.
- Adds an IGRIS_C normalization bridge that handles the new 86-dimensional proprioceptive state and 17-dimensional action vector.
- Introduces `run_inference_local.py`, a no-Ray entry point that mirrors `run_inference.py`.

Full file-by-file accounting is in [02_changes_vs_main.md](02_changes_vs_main.md).

## How `igris_c/c1` differs from `main`

At a glance, between branch tips (`git diff main..igris_c/c1`):

| Area | What changed |
|---|---|
| **Robot I/O** | Implemented [`env_actor/robot_io_interface/robots/igris_c/controller_bridge.py`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py) (Cyclone DDS); added a legacy SDK fallback, a DDS messages module, and a standalone bridge monitor. |
| **Runtime config** | Filled in [`init_params.py`](../../env_actor/runtime_settings_configs/robots/igris_c/init_params.py); added `inference_runtime_params.{json,py}`, `inference_runtime_topics.json`, and `dds/dds_state.xml` + `dds/dds_camera.xml`. |
| **Data managers** | Implemented the IGRIS_C sequential `DataManagerBridge` and added the RTC `SharedMemoryManager` for IGRIS_C under both `rtc/` and `rtc_local/`. |
| **Normalization** | New [`nom_stats_manager/robots/igris_c/data_normalization_manager.py`](../../env_actor/nom_stats_manager/robots/igris_c/data_normalization_manager.py). |
| **Local (no-Ray) runtime** | New `rtc_local/` and `sequential_local/` subtrees, plus `run_inference_local.py`. |
| **Policy YAML** | `openpi_batched.yaml` `action_dim: 24 → 17` (to match IGRIS_C action vector). |
| **IGRIS_B parity** | The IGRIS_B RTC `shm_manager_bridge.py` initial delay-queue value moved from `5` to `15` (affects guided inference timing on both robots — same change applied to IGRIS_C). |

See [02_changes_vs_main.md](02_changes_vs_main.md) for the per-file narrative.

## How to read these docs

A junior engineer with no prior knowledge of the repo, in this order:

1. [01_quickstart_igris_c.md](01_quickstart_igris_c.md) — first run, prerequisites, commands you actually type. Stop reading and try things at the first error.
2. [02_changes_vs_main.md](02_changes_vs_main.md) — what code is new on this branch and why. Skim sections that match the files you opened.
3. [03_robot_io_interface_igris_c.md](03_robot_io_interface_igris_c.md) — how the bridge talks to the robot over DDS. Read this before changing any topic name or action layout.
4. [04_runtime_configuration_igris_c.md](04_runtime_configuration_igris_c.md) — every field in the JSON/Python config and what depends on it.
5. [05_data_manager_bridges_igris_c.md](05_data_manager_bridges_igris_c.md) — observation history and shared-memory layout (both sequential and RTC).
6. [06_normalization_igris_c.md](06_normalization_igris_c.md) — stats file format and the normalize/denormalize math.
7. [07_factory_registration_igris_c.md](07_factory_registration_igris_c.md) — the dispatch graph from `run_inference.py` to the IGRIS_C-specific code.
8. [08_running_and_extending_igris_c.md](08_running_and_extending_igris_c.md) — CLI reference, day-to-day operation, "if I want to change X" recipes.
9. [09_troubleshooting_igris_c.md](09_troubleshooting_igris_c.md) — symptom → cause → fix.
10. [10_glossary_and_references.md](10_glossary_and_references.md) — vocabulary and outbound links.

Already comfortable with IGRIS_B? Jump straight to [02_changes_vs_main.md](02_changes_vs_main.md) and then the comparison table in [03_robot_io_interface_igris_c.md](03_robot_io_interface_igris_c.md).

## Document index

| # | File | One-line |
|---|---|---|
| 0 | [README.md](README.md) | This hub. |
| 1 | [01_quickstart_igris_c.md](01_quickstart_igris_c.md) | Zero-to-running on IGRIS_C. |
| 2 | [02_changes_vs_main.md](02_changes_vs_main.md) | File-by-file diff narrative. |
| 3 | [03_robot_io_interface_igris_c.md](03_robot_io_interface_igris_c.md) | The DDS controller bridge. |
| 4 | [04_runtime_configuration_igris_c.md](04_runtime_configuration_igris_c.md) | Runtime config (JSON + Python + DDS XML). |
| 5 | [05_data_manager_bridges_igris_c.md](05_data_manager_bridges_igris_c.md) | Sequential + RTC data managers. |
| 6 | [06_normalization_igris_c.md](06_normalization_igris_c.md) | Normalization bridge math. |
| 7 | [07_factory_registration_igris_c.md](07_factory_registration_igris_c.md) | Every `if robot == "igris_c"` site. |
| 8 | [08_running_and_extending_igris_c.md](08_running_and_extending_igris_c.md) | Day-to-day operation and extension. |
| 9 | [09_troubleshooting_igris_c.md](09_troubleshooting_igris_c.md) | Failure modes and fixes. |
| 10 | [10_glossary_and_references.md](10_glossary_and_references.md) | Glossary and outbound links. |

## Where to look in the existing docs

This folder **does not restate** what the existing docs already explain correctly. Use these first:

- Project root: [`README.md`](../../README.md) — prerequisites, install steps, CLI summary, the architecture diagram. Treat its IGRIS_B-specific defaults as IGRIS_B defaults, not IGRIS_C defaults — IGRIS_C uses different config files (see [04_runtime_configuration_igris_c.md](04_runtime_configuration_igris_c.md)).
- [`docs/README.md`](../README.md) — top-level doc index.
- [`docs/concepts.md`](../concepts.md) — VLA, action chunks, RTC, flow matching, shared memory, Ray actors, spawn vs fork, ROS2 primer. Read this once before touching the bridge.
- [`docs/architecture.md`](../architecture.md) — system topology and process model. The IGRIS_C bridge slots into the same `ControllerInterface` factory; the diagrams transfer.
- [`docs/rtc_shared_memory.md`](../rtc_shared_memory.md) — shared-memory design. The IGRIS_C RTC bridge uses the same primitives; only the shapes differ. See [05_data_manager_bridges_igris_c.md § RTC](05_data_manager_bridges_igris_c.md#rtc-sharedmemorymanager) for the IGRIS_C shapes.
- [`docs/walkthroughs/04_add_a_new_robot.md`](../walkthroughs/04_add_a_new_robot.md) — the recipe IGRIS_C/c1 followed. Useful if you want to add IGRIS_D later.
- [`docs/api.md`](../api.md) — public interfaces (`ControllerInterface`, `DataNormalizationInterface`, `RuntimeParams`, `build_policy`).
- [`docs/troubleshooting.md`](../troubleshooting.md) — generic crash table. IGRIS_C-specific entries are in [09_troubleshooting_igris_c.md](09_troubleshooting_igris_c.md).
- [`docs/glossary.md`](../glossary.md) — vocabulary. IGRIS_C-specific terms are added in [10_glossary_and_references.md](10_glossary_and_references.md).
- IGRIS_C subtree READMEs:
  - [`env_actor/robot_io_interface/robots/igris_c/README.md`](../../env_actor/robot_io_interface/robots/igris_c/README.md) — the original hardware-spec stub. Historical context only. **Note:** it predates the implementation in this branch; some statements there (e.g. "Implementation Stub Only") no longer apply on `igris_c/c1`. The implementation checklist is satisfied by the code in this branch.
  - [`env_actor/runtime_settings_configs/robots/igris_c/README.md`](../../env_actor/runtime_settings_configs/robots/igris_c/README.md) — also predates the implementation. It says `inference_runtime_params.json` is "Missing"; on `igris_c/c1` it exists. The "What needs to be created" checklist is done.
  - [`env_actor/robot_io_interface/robots/igris_b/README.md`](../../env_actor/robot_io_interface/robots/igris_b/README.md) — the reference template the implementation followed.

Whenever the existing docs and this folder disagree, this folder is the branch-specific source of truth for `igris_c/c1`. Discrepancies with existing docs are recorded in [09_troubleshooting_igris_c.md § Doc/code inconsistencies](09_troubleshooting_igris_c.md#docscode-inconsistencies-on-this-branch).

## Submodule references

`trainer/` is pinned at the same SHA on both `main` and `igris_c/c1` (as of `git diff main..igris_c/c1`):

```
trainer @ 3ca051a256c9068f77b556df98f538d9a6185ccf
```

Verify with `git ls-tree igris_c/c1 -- trainer`. Note that **inside** the branch's history there is a commit (`dd4e8a9 updated trainer`) that bumps trainer from `6cefd4715c33a731f6b8f57b80423780d8fb1f50` → `3ca051a256c9068f77b556df98f538d9a6185ccf`. The four upstream commits this brought in:

```
3ca051a updated documentation
c71ff08 removed unnecessary folders
4f4b4b9 fixing inconsistency
ca5de96 documented policy_constructor
```

In other words, the branch and main are now in lock-step on the trainer pointer; the branch went through an interim where it was behind, and the bump landed in `dd4e8a9` so the two tips would agree.

Outbound links into the submodule (pinned at the SHA above):
- [trainer/README.md](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/README.md)
- [trainer/docs/README.md](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/docs/README.md)
- [trainer/docs/05_configuration.md](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/docs/05_configuration.md) (trainer-side config — the source of the `dataset_stats.pkl` referenced by IGRIS_C `inference_runtime_params.json`)

The nested `trainer/policy_constructor` submodule is pinned at `00663cc10c91d7614c1a0ea3d68629c38767b167` per the root `README.md` acknowledgement section.

## Conventions used here

- Every file starts with what it covers and who it is for, then a TOC if it has more than two `##` sections.
- Every file ends with a "Next →" pointer and a "← Back to index" pointer to this README.
- Relative links only, except for `https://github.com/...` URLs that point into the `trainer/` submodule (these are pinned to the SHA recorded above).
- Code references use `file:line` form: e.g. [`controller_bridge.py:386`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L386).
- Project-specific values that may not match your machine are flagged "current value on this branch — verify against your environment" or shown as `<PLACEHOLDER>` with a `TODO:` note.
- Code blocks always declare their language: ` ```bash `, ` ```python `, ` ```json `, ` ```yaml `, ` ```xml `, ` ```mermaid `.
- Diagrams are ASCII or Mermaid. No images.

---

Next → [01_quickstart_igris_c.md](01_quickstart_igris_c.md)
