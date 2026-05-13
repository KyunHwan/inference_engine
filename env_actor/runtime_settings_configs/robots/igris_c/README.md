# runtime_settings_configs/robots/igris_c

**Parent:** [runtime_settings_configs](../../README.md)

Placeholder configuration directory for the IGRIS_C robot. Currently contains only [`init_params.py`](init_params.py), which is itself a stub.

## Table of contents

- [Status](#status)
- [What needs to be created](#what-needs-to-be-created)
- [Step-by-step](#step-by-step)
- [Related docs](#related-docs)

## Status

| File | State |
|---|---|
| `init_params.py` | Stub — `INIT_JOINT_LIST = []`, `IGRIS_C_STATE_KEYS = []`, etc. with `TODO` comments. |
| `inference_runtime_params.json` | **Missing.** Required before any IGRIS_C run can start. |
| `inference_runtime_topics.json` | **Missing.** Required. |
| `inference_runtime_params.py` | **Missing.** Required (a `RuntimeParams` class). |

Currently, `python run_inference.py --robot igris_c` will fail at the very first step: `sequential_actor.py` (or `rtc_actor.py`) tries to `from env_actor.runtime_settings_configs.robots.igris_c.inference_runtime_params import RuntimeParams` and the file isn't there.

## What needs to be created

In order:

1. **Fill in `init_params.py`** with real values:

   ```python
   INIT_JOINT_LIST = [...]   # In degrees, length = arm joint count × sides
   INIT_HAND_LIST  = [...]   # Initial finger targets
   INIT_JOINT = np.array(INIT_JOINT_LIST, dtype=np.float32) * np.pi / 180.0
   IGRIS_C_STATE_KEYS = ["/observation/joint_pos/left", ...]
   ```

2. **Create `inference_runtime_params.json`** modeled after [igris_b/inference_runtime_params.json](../igris_b/inference_runtime_params.json). At minimum set `proprio_state_dim`, `action_dim`, and `camera_names` to match your hardware.

3. **Create `inference_runtime_topics.json`** modeled after [igris_b/inference_runtime_topics.json](../igris_b/inference_runtime_topics.json). Update `robot_id`, every topic name (likely `/igris_c/<robot_id>/...`), and the `slice` ranges to match your messages.

4. **Create `inference_runtime_params.py`** — usually identical to [igris_b/inference_runtime_params.py](../igris_b/inference_runtime_params.py); just copy it.

## Step-by-step

The full walkthrough that orchestrates these four files plus all the matching bridges lives in [docs/walkthroughs/04_add_a_new_robot.md § Phase 1](../../../../docs/walkthroughs/04_add_a_new_robot.md#phase-1-runtime-configs).

## Related docs

- [docs/walkthroughs/04_add_a_new_robot.md](../../../../docs/walkthroughs/04_add_a_new_robot.md) — full robot bring-up.
- [robots/igris_b/README.md](../igris_b/README.md) — your reference template.
- [robot_io_interface/robots/igris_c/README.md](../../../robot_io_interface/robots/igris_c/README.md) — the hardware spec checklist.
