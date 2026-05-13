# Development guide

## Table of contents

- [Environment setup](#environment-setup)
- [Project structure conventions](#project-structure-conventions)
- [Design patterns](#design-patterns)
- [Editing the trainer submodule](#editing-the-trainer-submodule)
- [What you should never edit from this repo](#what-you-should-never-edit-from-this-repo)
- [Adding components checklists](#adding-components-checklists)
- [Ray cluster management](#ray-cluster-management)
- [Debugging tips](#debugging-tips)
- [Style and review conventions](#style-and-review-conventions)

## Environment setup

```bash
git clone --recurse-submodules <repo-url>
cd inference_engine

bash uv_setup.sh         # uv + .venv (Python 3.12.3)
source .venv/bin/activate
bash env_setup.sh        # PyTorch 2.9.0+cu130, Ray, Transformers, etc.
bash openpi_transformer_lib_patch.sh    # only if you use OpenPI/Pi0.5

git submodule update --init --recursive   # if you cloned without --recurse-submodules
```

Use `uv pip install <pkg>` for new dependencies; do not use bare `pip` (it will install into the system Python and your venv will diverge from `env_setup.sh`).

## Project structure conventions

### Directory organization

- **Robot-specific code** lives under `robots/<robot_name>/` subdirectories.
- **Interfaces / factories** live at the module root (`controller_interface.py`, `data_normalization_interface.py`, etc.).
- **Protocols** live in `policy/templates/`.
- **Configuration** is YAML for model + policy definitions, JSON for runtime parameters.

### Naming

| Suffix | Means |
|---|---|
| `*_bridge.py` | Robot-specific implementation behind an interface (one per (algorithm, robot) pair) |
| `*_interface.py` | Factory + delegation entry point (`ControllerInterface`, `DataNormalizationInterface`, `SharedMemoryInterface`) |
| `*_actor.py` | Ray remote actor class |
| `RuntimeParams` (in `inference_runtime_params.py`) | Per-robot runtime config class |

## Design patterns

- **Factory** — a string `robot` argument selects which bridge to import. See [controller_interface.py](../env_actor/robot_io_interface/controller_interface.py), [data_manager_interface.py](../env_actor/auto/inference_algorithms/sequential/data_manager/data_manager_interface.py), [shm_manager_interface.py](../env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py), [data_normalization_interface.py](../env_actor/nom_stats_manager/data_normalization_interface.py).
- **Protocol (structural subtyping)** — `Policy` ([policy.py](../env_actor/policy/templates/policy.py)) is a `Protocol`, not a base class. You don't inherit; you implement the methods.
- **Registry** — `@POLICY_REGISTRY.register("name")` ([core.py](../env_actor/policy/registry/core.py)) maps a string to a class. `build_policy` looks up by string.
- **Bridge** — every robot has its own implementation file under `robots/<name>/`. Interfaces forward calls to bridges.

## Editing the trainer submodule

`trainer/` is a git submodule pointing at [KyunHwan/trainer](https://github.com/KyunHwan/trainer). It is pinned to commit `3ca051a256c9068f77b556df98f538d9a6185ccf` by [.gitmodules](../.gitmodules) + the SHA recorded in `git submodule status`.

**To change something inside `trainer/`:**

1. Make the change in the **trainer repo**, not here. Push it upstream.

   ```bash
   cd trainer
   git checkout main
   git pull
   # edit files...
   git add -A
   git commit -m "..."
   git push
   cd ..
   ```

2. Bump the pinned SHA in this repo:

   ```bash
   git -C trainer log -1 --format=%H   # confirm the new SHA
   git add trainer                      # stages the new submodule SHA
   git commit -m "Bump trainer submodule"
   git push
   ```

3. Anyone else's working copy will need:

   ```bash
   git pull
   git submodule update --init --recursive
   ```

The same procedure applies to `trainer/policy_constructor/`. Updates flow inside-out: edit `policy_constructor`, push, bump in `trainer`, push trainer, bump in `inference_engine`, push.

## What you should never edit from this repo

- Anything inside `trainer/` (including `trainer/policy_constructor/`). Treat it as read-only. Documentation, tests, source — all of it. Changes go upstream.
- `.venv/` — that's pip's territory. Use `uv pip install` to manage it.
- The `transformers` package files that [openpi_transformer_lib_patch.sh](../openpi_transformer_lib_patch.sh) overwrites — those come from `trainer/...`. If you want to change them, change them in the trainer repo and re-run the patch.

## Adding components checklists

### New policy

- [ ] `env_actor/policy/policies/{name}/{name}.py` with `@POLICY_REGISTRY.register("{name}")`
- [ ] Implement all `Policy` protocol methods (`__init__`, `predict`, `guided_inference`, `warmup`, `freeze_all_model_params`)
- [ ] `env_actor/policy/policies/{name}/{name}.yaml` with `policy.type: {name}`
- [ ] Component YAML(s) under `env_actor/policy/policies/{name}/components/`
- [ ] Smoke test: `python run_inference.py --robot igris_b -P env_actor/policy/policies/{name}/{name}.yaml`

Full walkthrough: [walkthroughs/03_add_a_new_policy.md](walkthroughs/03_add_a_new_policy.md).

### New robot

- [ ] `env_actor/runtime_settings_configs/robots/{name}/init_params.py` — `INIT_JOINT`, `INIT_HAND_LIST`, `<NAME>_STATE_KEYS`
- [ ] `env_actor/runtime_settings_configs/robots/{name}/inference_runtime_params.json`
- [ ] `env_actor/runtime_settings_configs/robots/{name}/inference_runtime_topics.json`
- [ ] `env_actor/runtime_settings_configs/robots/{name}/inference_runtime_params.py` — `RuntimeParams` class
- [ ] `env_actor/robot_io_interface/robots/{name}/controller_bridge.py`
- [ ] `env_actor/auto/inference_algorithms/sequential/data_manager/robots/{name}/data_manager_bridge.py`
- [ ] `env_actor/auto/inference_algorithms/rtc/data_manager/robots/{name}/shm_manager_bridge.py`
- [ ] `env_actor/nom_stats_manager/robots/{name}/data_normalization_manager.py`
- [ ] Update `if/elif` blocks in:
  - [controller_interface.py](../env_actor/robot_io_interface/controller_interface.py)
  - [data_manager_interface.py](../env_actor/auto/inference_algorithms/sequential/data_manager/data_manager_interface.py)
  - [shm_manager_interface.py](../env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py)
  - [data_normalization_interface.py](../env_actor/nom_stats_manager/data_normalization_interface.py)
  - [sequential_actor.py](../env_actor/auto/inference_algorithms/sequential/sequential_actor.py) (runtime params import)
  - [rtc_actor.py](../env_actor/auto/inference_algorithms/rtc/rtc_actor.py) (runtime params import)
  - [control_loop.py](../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py)
  - [inference_loop.py](../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py)
- [ ] Add `"{name}"` to `choices` in [run_inference.py](../run_inference.py) argparse

Full walkthrough: [walkthroughs/04_add_a_new_robot.md](walkthroughs/04_add_a_new_robot.md).

### New inference algorithm

- [ ] `env_actor/auto/inference_algorithms/{name}/`
- [ ] A Ray remote actor class with a `start()` method
- [ ] Robot-specific data manager bridges under `data_manager/robots/<robot>/`
- [ ] Add the algorithm name to `--inference_algorithm` choices in `run_inference.py`
- [ ] Wire actor creation in `run_inference.py`'s `start_inference()`

## Ray cluster management

```bash
bash start_ray.sh         # multi-machine (edit hostnames first)
ray status                # show cluster + per-node resources
ray stop                  # tear down on this machine
```

Single-machine smoke test (skip `start_ray.sh`):

```bash
ray start --head --port=6379
ray start --address=127.0.0.1:6379 --resources='{"inference_pc": 1}'
```

Dashboard: `http://<HEAD_IP>:8265`. With `--dashboard-host=0.0.0.0` in `start_ray.sh` it's reachable from other hosts on the same network.

## Debugging tips

### Running without Ray

The `SequentialActor` class can be instantiated directly (it's just a regular class with `@ray.remote` on top). For a script that bypasses Ray entirely, copy `SequentialActor.start()` into a function that takes the same constructor args.

You can also `python -c "from env_actor.policy.utils.loader import build_policy; p = build_policy('path/to/yaml'); print(p)"` to test policy loading in isolation.

### Common environment issues

```bash
# CUDA
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
python -c "import torch; print(torch.cuda.is_available())"

# ROS2
source /opt/ros/<distro>/setup.bash
ros2 topic list
ros2 topic echo /igris_b/<robot_id>/joint_states --once
```

If CUDA isn't visible, see [troubleshooting.md § CUDA](troubleshooting.md#cuda).

### Multiprocessing errors

If you call inference from your own script, set `torch.multiprocessing.set_start_method("spawn")` before any CUDA operation. The entrypoint already does it; standalone scripts often forget.

## Style and review conventions

- **Markdown only** for documentation in this repo. No `.rst`, no Sphinx config.
- **Link, don't copy**. If a fact lives in [glossary.md](glossary.md), link to it instead of restating it. The single source of truth for shapes is [api.md § Key data shapes](api.md#key-data-shapes-igris_b-defaults).
- **Tables for parameter lists**, fenced blocks for commands and YAML/JSON.
- **Define jargon on first use** within a file. Long-form on first occurrence, abbreviation after. "Vision–Language–Action (VLA) policy. The VLA…"
- **Diagrams are ASCII** in the primary form; Mermaid can be added inside `<details>` blocks but never replacing the ASCII version.
- **No emoji.** Add them only if the user explicitly asks.
- **No comments that explain WHAT.** Variable names already do that. Comments are for non-obvious WHY: invariants, gotchas, citation of a specific bug.
- **Keep folder READMEs short.** They should mostly link. The deep material lives in `docs/*`.
- **A change to a public interface bumps the API doc.** Don't merge a signature change without updating [api.md](api.md).
