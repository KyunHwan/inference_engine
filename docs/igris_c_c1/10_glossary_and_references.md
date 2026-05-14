# 10 — Glossary and outbound references

**What this covers.** Every IGRIS_C-specific or branch-specific term used in this folder, defined once. Plus outbound links to the existing repo docs and the relevant `trainer/` submodule docs at the pinned SHA.
**Who this is for.** Anyone who hit an unfamiliar acronym and needs the one-line answer.

For broader inference-engine vocabulary (VLA, action chunk, RTC, action inpainting, etc.), see [`docs/glossary.md`](../glossary.md). This file extends that with IGRIS_C-specific terms.

## Table of contents

- [IGRIS_C-specific glossary](#igris_c-specific-glossary)
- [Generic terms used here (definitions cross-linked)](#generic-terms-used-here-definitions-cross-linked)
- [References to existing repo docs](#references-to-existing-repo-docs)
- [References to IGRIS_B reference files](#references-to-igris_b-reference-files)
- [References to the `trainer/` submodule](#references-to-the-trainer-submodule)
- [References to the `policy_constructor` submodule](#references-to-the-policy_constructor-submodule)

## IGRIS_C-specific glossary

| Term | Definition |
|---|---|
| **IGRIS_C** | The robot platform documented by this branch. Bipedal humanoid (3 waist + 12 leg + 14 arm + 2 neck = 31 body joints; 12 finger motors). Communicates via Cyclone DDS rather than ROS2 (the IGRIS_B convention). |
| **NUC** | The on-robot Intel NUC that runs the firmware speaking DDS to the host. Listens on `rt/lowcmd` / `rt/handcmd`; publishes `rt/lowstate` / `rt/handstate`. |
| **Jetson** | The on-robot NVIDIA Jetson that publishes camera frames on `igris_c/sensor/eyes_stereo`, `.../left_hand`, `.../right_hand`. |
| **PJS** | Parallel Joint Space — kinematic_mode where motor commands and reads are in joint coordinates (after parallel-mechanism transforms). The IGRIS_C bridge sets `kinematic_mode=KinematicMode.PJS` on every `LowCmd` ([`controller_bridge.py:480`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L480)). |
| **MS** | Motor Space — the raw per-motor encoder coordinates. IGRIS_C uses PJS for commands but reads `tau_est` from `MotorState` (which is MS-side) — same convention as the recorder. |
| **action_dim=17** | The IGRIS_C action vector size: 7 left arm + 7 right arm + 1 left hand + 1 right hand + 1 waist yaw. The bridge **lifts** this to a full 31-D body command by holding the other 16 joints at `HOME_POSE_RAD`. |
| **proprio_state_dim=86** | The IGRIS_C proprioceptive observation size: 31 body joint positions + 12 hand joint positions + 31 body torques + 12 hand torques (current proxy). The bridge concatenates these by position. See [03 § State packing](03_robot_io_interface_igris_c.md#state-packing-layout-the-86-d-proprio-vector). |
| **`HOME_POSE_RAD`** | The 31-D home pose IGRIS_C boots into. Mostly zeros; two non-zero entries at indices 16 (`+0.13` rad) and 23 (`-0.13` rad) for Shoulder_Roll_L/R. Defined in [`init_params.py:24-29`](../../env_actor/runtime_settings_configs/robots/igris_c/init_params.py#L24). |
| **`INIT_ACTION_17`** | The 17-D action vector used to fill the action chunk at episode start. Defined in [`init_params.py:64-68`](../../env_actor/runtime_settings_configs/robots/igris_c/init_params.py#L64). |
| **`LEFT_ARM_IDS` / `RIGHT_ARM_IDS` / `WAIST_YAW_ID`** | Index lists into the 31-D body vector that the bridge writes action values into. `LEFT_ARM_IDS = [15..21]`, `RIGHT_ARM_IDS = [22..28]`, `WAIST_YAW_ID = 0`. |
| **`HAND_LEFT_IDS` / `HAND_RIGHT_IDS`** | Index lists into the **hand** DDS topic (not body). `[11..16]` and `[21..26]` respectively. These IDs overlap numerically with body joints but live on a separate DDS topic (`rt/handstate` / `rt/handcmd`). |
| **DDS Domain 0** | The state/cmd domain on this branch. Carries `rt/lowstate`, `rt/handstate`, `rt/lowcmd`, `rt/handcmd`. Pinned to NIC `enp11s0` via `dds/dds_state.xml`. |
| **DDS Domain 1** | The camera domain on this branch. Carries `igris_c/sensor/{eyes_stereo, left_hand, right_hand}`. Same NIC `enp11s0`. Some operator setups bridge cameras to Domain 10 over WiFi — see the comment block in `dds/dds_camera.xml`. |
| **`CYCLONEDDS_URI`** | Environment variable libcyclonedds reads to configure participants. The IGRIS_C bridge sets this to the **contents** of `dds_state.xml` (not the path) at construction time. |
| **`BRIDGE_HEALTH_LOG`** | Environment variable that, when set, makes the bridge spawn a daemon thread that prints per-topic Hz + sample age every period_sec. See [03 § Health monitoring](03_robot_io_interface_igris_c.md#health-monitoring). |
| **Slew-rate limit / `max_delta`** | Per-step joint delta cap. `max_delta = np.deg2rad(max_delta_deg)`, currently 5° / step. At 20 Hz, this caps each joint to ~100°/s. Applied in `publish_action` ([`controller_bridge.py:396-398`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L396)). |
| **Init ramp** | The `init_robot_position` routine: linearly interpolates from the live `body_q` to a hard-coded 31-D `start_position` over 100 steps × 50 ms (5 s total). See [`controller_bridge.py:340, 412-489`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L340). |
| **stereo eye crop** | The head camera publishes a side-by-side stereo image. `read_state` rotates 180° and takes the left half (`frame[:, :w//2, :]`) — [`controller_bridge.py:378-381`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L378). |
| **`rtc_local` / `sequential_local`** | The no-Ray mirror subtrees added on this branch. They use `multiprocessing.spawn` directly. Functionally identical to `rtc/` and `sequential/` minus the Ray imports. See [02 § rtc_local](02_changes_vs_main.md#env_actorautoinference_algorithmsrtc_local-and-the-12-other-files-under-rtc_local-added--13-files). |
| **`run_bridge_monitor`** | Standalone script that brings up the bridge without the policy stack and prints a rolling state summary. Use it to verify DDS connectivity before running inference. See [03 § Bridge-only test](03_robot_io_interface_igris_c.md#bridge-only-test). |

## Generic terms used here (definitions cross-linked)

These terms are defined comprehensively in the existing root docs. This table is just so you can jump straight to the canonical definition.

| Term | Canonical definition |
|---|---|
| Action chunk | [`docs/concepts.md`](../concepts.md), [`docs/glossary.md`](../glossary.md) |
| Action inpainting | [`docs/concepts.md`](../concepts.md), referenced paper in root `README.md` acknowledgements |
| RTC (Real-Time Control) | [`docs/concepts.md`](../concepts.md), [`docs/rtc_shared_memory.md`](../rtc_shared_memory.md) |
| Sequential algorithm | [`docs/architecture.md`](../architecture.md), [`docs/walkthroughs/02_trace_one_step.md`](../walkthroughs/02_trace_one_step.md) |
| Policy protocol | [`docs/api.md § Policy protocol`](../api.md), [`env_actor/policy/README.md`](../../env_actor/policy/README.md) |
| `build_policy()` | [`docs/api.md`](../api.md), [`env_actor/policy/utils/README.md`](../../env_actor/policy/utils/README.md) |
| `POLICY_REGISTRY` | [`env_actor/policy/registry/README.md`](../../env_actor/policy/registry/README.md) |
| `ControllerInterface` | [`docs/api.md`](../api.md), [`env_actor/robot_io_interface/README.md`](../../env_actor/robot_io_interface/README.md) |
| `DataNormalizationInterface` | [`docs/api.md`](../api.md), [`env_actor/nom_stats_manager/README.md`](../../env_actor/nom_stats_manager/README.md) |
| `RuntimeParams` | [`docs/api.md`](../api.md), [`env_actor/runtime_settings_configs/README.md`](../../env_actor/runtime_settings_configs/README.md) |
| `@ray.remote` / actor lifecycle | [`docs/architecture.md`](../architecture.md), [`docs/concepts.md`](../concepts.md) |
| `inference_pc` custom resource | [`docs/architecture.md`](../architecture.md), root [`README.md § Ray cluster setup`](../../README.md#ray-cluster-setup) |
| `torch.compile`, `torch.autocast`, `torch.inference_mode`, `cudnn.benchmark` | [`docs/concepts.md`](../concepts.md) (CUDA-optimization section) |
| `torch.multiprocessing.set_start_method("spawn")` | [`docs/concepts.md`](../concepts.md), [`docs/troubleshooting.md`](../troubleshooting.md) (CUDA section) |
| `multiprocessing.shared_memory` | [`docs/rtc_shared_memory.md`](../rtc_shared_memory.md) |
| Flow matching | [`docs/concepts.md`](../concepts.md), `trainer/` docs (see below) |
| VLA (Vision-Language-Action) | [`docs/concepts.md`](../concepts.md), [`docs/glossary.md`](../glossary.md) |

## References to existing repo docs

Relative links from this folder. Read these for the conceptual / framework-wide content that this branch did not change.

- Top-level entry point — [`../../README.md`](../../README.md)
- Docs hub — [`../README.md`](../README.md)
- Concepts and mental model — [`../concepts.md`](../concepts.md)
- Architecture overview — [`../architecture.md`](../architecture.md)
- Public API reference — [`../api.md`](../api.md)
- Dev setup & conventions — [`../development.md`](../development.md)
- Onboarding — [`../onboarding.md`](../onboarding.md)
- Generic glossary — [`../glossary.md`](../glossary.md)
- Configuration cookbook — [`../configuration_cookbook.md`](../configuration_cookbook.md)
- RTC shared memory deep-dive — [`../rtc_shared_memory.md`](../rtc_shared_memory.md)
- Generic troubleshooting — [`../troubleshooting.md`](../troubleshooting.md)
- FAQ — [`../faq.md`](../faq.md)
- Walkthroughs hub — [`../walkthroughs/README.md`](../walkthroughs/README.md)
  - First run — [`../walkthroughs/01_first_run.md`](../walkthroughs/01_first_run.md)
  - Trace one step — [`../walkthroughs/02_trace_one_step.md`](../walkthroughs/02_trace_one_step.md)
  - Add a new policy — [`../walkthroughs/03_add_a_new_policy.md`](../walkthroughs/03_add_a_new_policy.md)
  - **Add a new robot** — [`../walkthroughs/04_add_a_new_robot.md`](../walkthroughs/04_add_a_new_robot.md) — this is the procedure `igris_c/c1` followed
- Module READMEs (selection):
  - [`../../env_actor/README.md`](../../env_actor/README.md)
  - [`../../env_actor/policy/README.md`](../../env_actor/policy/README.md)
  - [`../../env_actor/auto/README.md`](../../env_actor/auto/README.md)
  - [`../../env_actor/auto/inference_algorithms/README.md`](../../env_actor/auto/inference_algorithms/README.md)
  - [`../../env_actor/auto/inference_algorithms/rtc/README.md`](../../env_actor/auto/inference_algorithms/rtc/README.md)
  - [`../../env_actor/auto/inference_algorithms/rtc/actors/README.md`](../../env_actor/auto/inference_algorithms/rtc/actors/README.md)
  - [`../../env_actor/auto/inference_algorithms/rtc/data_manager/README.md`](../../env_actor/auto/inference_algorithms/rtc/data_manager/README.md)
  - [`../../env_actor/auto/inference_algorithms/sequential/README.md`](../../env_actor/auto/inference_algorithms/sequential/README.md)
  - [`../../env_actor/auto/inference_algorithms/sequential/data_manager/README.md`](../../env_actor/auto/inference_algorithms/sequential/data_manager/README.md)
  - [`../../env_actor/robot_io_interface/README.md`](../../env_actor/robot_io_interface/README.md)
  - [`../../env_actor/runtime_settings_configs/README.md`](../../env_actor/runtime_settings_configs/README.md)
  - [`../../env_actor/nom_stats_manager/README.md`](../../env_actor/nom_stats_manager/README.md)
  - [`../../env_actor/inference_engine_utils/README.md`](../../env_actor/inference_engine_utils/README.md)
- IGRIS_C subtree READMEs (note: some are stale — see [09 § Docs/code inconsistencies](09_troubleshooting_igris_c.md#docscode-inconsistencies-on-this-branch)):
  - [`../../env_actor/robot_io_interface/robots/igris_c/README.md`](../../env_actor/robot_io_interface/robots/igris_c/README.md)
  - [`../../env_actor/runtime_settings_configs/robots/igris_c/README.md`](../../env_actor/runtime_settings_configs/robots/igris_c/README.md)

## References to IGRIS_B reference files

When porting features between robots, these IGRIS_B files are the reference templates:

| Concern | IGRIS_B file (link) |
|---|---|
| Controller bridge | [`env_actor/robot_io_interface/robots/igris_b/controller_bridge.py`](../../env_actor/robot_io_interface/robots/igris_b/controller_bridge.py) |
| Sequential data manager | [`env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_b/data_manager_bridge.py`](../../env_actor/auto/inference_algorithms/sequential/data_manager/robots/igris_b/data_manager_bridge.py) |
| RTC shm manager | [`env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py`](../../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py) |
| Normalization bridge | [`env_actor/nom_stats_manager/robots/igris_b/data_normalization_manager.py`](../../env_actor/nom_stats_manager/robots/igris_b/data_normalization_manager.py) |
| `init_params.py` | [`env_actor/runtime_settings_configs/robots/igris_b/init_params.py`](../../env_actor/runtime_settings_configs/robots/igris_b/init_params.py) |
| `inference_runtime_params.py` | [`env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.py`](../../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.py) |
| `inference_runtime_params.json` | [`env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json`](../../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json) |
| `inference_runtime_topics.json` | [`env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_topics.json`](../../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_topics.json) |

## References to the `trainer/` submodule

Pinned at `3ca051a256c9068f77b556df98f538d9a6185ccf` on `igris_c/c1` (same as `main`). All links pin the SHA.

| Doc | Link |
|---|---|
| Trainer overview | https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/README.md |
| Trainer docs hub | https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/docs/README.md |
| Getting started | https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/docs/01_getting_started.md |
| Concepts | https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/docs/04_concepts.md |
| Configuration | https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/docs/05_configuration.md |
| Extending | https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/docs/07_extending.md |
| Troubleshooting | https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/docs/10_troubleshooting.md |
| Glossary | https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/docs/12_glossary.md |

The IGRIS_C inference engine consumes the trainer's `dataset_stats.pkl` (see [06](06_normalization_igris_c.md)) and is currently configured to use OpenPI / Pi-0.5 checkpoints trained by the trainer. The trainer's configuration docs explain how those stats are generated.

## References to the `policy_constructor` submodule

`trainer/policy_constructor/` is a nested submodule pinned at `00663cc10c91d7614c1a0ea3d68629c38767b167` (per root [`README.md`](../../README.md) acknowledgements).

| Doc | Link |
|---|---|
| Policy constructor README | https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/README.md |
| Docs index | https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/docs/INDEX.md |
| Quickstart | https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/docs/QUICKSTART.md |
| Mental model | https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/docs/MENTAL_MODEL.md |
| Glossary | https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/docs/GLOSSARY.md |
| Troubleshooting | https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/docs/TROUBLESHOOTING.md |

`env_setup.sh` installs Depth Anything v3 from `trainer/policy_constructor/.../depth_anything_3` as an editable package. If you need to know more about the model construction graph the policy uses, the `policy_constructor` docs are the right starting point.

---

← Back to index: [README.md](README.md) · This is the last file in the set.
