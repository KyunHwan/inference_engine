# Configuration cookbook

"If I want to change X, edit Y." Each recipe lists the files to touch, the value to change, and what to retest.

## Table of contents

- [Change control frequency](#change-control-frequency)
- [Change image resolution](#change-image-resolution)
- [Change action chunk size](#change-action-chunk-size)
- [Change the OpenPI checkpoint path](#change-the-openpi-checkpoint-path)
- [Change the prompt](#change-the-prompt)
- [Add a fourth camera](#add-a-fourth-camera)
- [Tune image history (num_img_obs)](#tune-image-history-num_img_obs)
- [Switch the default robot](#switch-the-default-robot)
- [Switch the default algorithm (RTC ↔ Sequential)](#switch-the-default-algorithm-rtc--sequential)
- [Use a non-GPU machine for debugging](#use-a-non-gpu-machine-for-debugging)
- [Change normalization stats](#change-normalization-stats)
- [Change slew-rate limit](#change-slew-rate-limit)

## Change control frequency

**Default**: 20 Hz.

**Edit**: [inference_runtime_params.json](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json) → `"HZ": 30`.

**Also check**:

- [inference_runtime_topics.json](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_topics.json) → `"HZ"` (used by `GenericRecorder` for `data_time` warnings). Keep them aligned.
- The cameras' native FPS — `RBRSCamera.singleCamera` sets `CAP_PROP_FPS=60`, so 30 Hz is fine but 60 Hz isn't sustainable.
- `policy_update_period` is in *steps*. If you keep `policy_update_period: 50`, inference now fires every `50/30 ≈ 1.67 s` instead of `2.5 s`. Adjust if you want a constant inference cadence in seconds.

**Retest**: Sequential first. Watch `nvidia-smi` to ensure inference still completes before the next `policy_update_period`.

## Change image resolution

**Default**: 320 × 240.

**Edit**: [inference_runtime_params.json](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json) → `"mono_image_resize": {"width": <W>, "height": <H>}`.

**Also check**:

- The policy was trained at a specific resolution. Changing it usually means re-training, or accepting accuracy loss because the backbone was finetuned at the training size. Read the policy's component YAML.
- RTC's shared-memory blocks are sized at startup from these values, so the new resolution will take effect on the next launch — no migration needed.

**Retest**: confirm `read_state()["head"].shape == (3, H, W)`. The controller bridge does the resize in [controller_bridge.py:76-78](../env_actor/robot_io_interface/robots/igris_b/controller_bridge.py#L76-L78).

## Change action chunk size

**Default**: 50.

**Edit two places**:

- [inference_runtime_params.json](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json) → `"action_chunk_size": <N>`.
- The policy's component YAML — e.g. [openpi_batched.yaml](../env_actor/policy/policies/openpi_policy/components/openpi_batched.yaml) → `params.action_horizon: <N>`.

These two values must agree, or the policy will produce a chunk of one size and the data manager will try to index it as if it were another.

**Trade-offs** — see [concepts.md § Action chunks](concepts.md#action-chunks).

**Retest**: a fresh `policy.predict()` should return `(N, action_dim)`. Use [walkthroughs/01_first_run.md § Dry run](walkthroughs/01_first_run.md#dry-run--no-robot)'s scripted load to verify.

## Change the OpenPI checkpoint path

**Default**: hard-coded under `/home/robros/Projects/...` in [openpi_batched.yaml](../env_actor/policy/policies/openpi_policy/components/openpi_batched.yaml) and [components/openpi_model.yaml](../env_actor/policy/policies/dsrl_openpi_policy/components/openpi_model.yaml).

**Edit**: in the relevant component YAML, change `params.ckpt_dir` to your new absolute path.

**Note**: this is `ckpt_dir` (consumed by `OpenPiBatchedWrapper`), **not** `checkpoint_path` at the top-level policy YAML (that one is read by `build_policy` and applies to *all* components). For OpenPI, all weights live under `ckpt_dir`; the policy YAML's `checkpoint_path` is typically unset.

**Retest**: launch and confirm the policy loads from your new path. A wrong path raises `FileNotFoundError` or, more cryptically, an OpenPI internal exception.

## Change the prompt

**Default**: a sentence per policy YAML (`params.default_prompt`).

**Edit**: the component YAML's `params.default_prompt`. The wrapper uses this as the language conditioning unless the caller passes `"prompt"` in the observation dict.

If you want per-step prompts, add `obs["prompt"] = "..."` somewhere upstream (e.g. modify your policy class's `predict`).

**Retest**: behavior change is the test — same observations, different prompt, different actions.

## Add a fourth camera

**Edit four places**:

1. [inference_runtime_params.json](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json) → `"camera_names": ["head", "left", "right", "wrist"]`.
2. [controller_bridge.py](../env_actor/robot_io_interface/robots/igris_b/controller_bridge.py) → `_start_cam_recording` has hardcoded `if cam_name in ['head', 'right']` and `elif cam_name == 'left'`. Add an `elif cam_name == 'wrist': self.cams['wrist'] = RBRSCamera(device_id1="/dev/wrist_camera1", device_id2=None)`.
3. The policy's component YAML — if your policy uses an image processor with a fixed camera list (DSRL's [noise_processor.yaml](../env_actor/policy/policies/dsrl_openpi_policy/components/noise_processor.yaml) has `img_data_keys: ['head', 'left', 'right']`), add `"wrist"` there.
4. RTC's [rtc_actor.py](../env_actor/auto/inference_algorithms/rtc/rtc_actor.py) creates one SHM block per camera by name (`head_cam_shm`, `left_cam_shm`, `right_cam_shm`). If you add a fourth, add a block for it, add the spec to `shm_specs`, and make sure the shm_manager_bridge knows about it.

**Retest**: a fresh launch should print four camera initialization lines. `read_state()` should return a dict with four image keys.

## Tune image history (num_img_obs)

**Default**: 1 (only the latest frame).

**Edit**: [inference_runtime_params.json](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json) → `"num_img_obs": <N>`.

**Effect on shapes**: `obs[cam].shape` becomes `(N, 3, H, W)`. The data manager (Sequential) and the SHM region (RTC) will allocate `(N, 3, H, W)` accordingly on next launch.

**Caveat**: the OpenPI policy uses only the latest frame regardless (`obs[cam][-1:]` in `predict`). Histories > 1 only matter for policies that actually consume them — see the DSRL policy if you're writing one.

## Switch the default robot

**Edit**: [run_inference.py](../run_inference.py) — change the default for `--robot`. The current default is `igris_b`. To make `your_robot` the default, change the `default="igris_b"` argument.

The `choices` list also needs your robot present (you should have added it during [walkthroughs/04_add_a_new_robot.md](walkthroughs/04_add_a_new_robot.md)).

## Switch the default algorithm (RTC ↔ Sequential)

**Edit**: [run_inference.py](../run_inference.py) — change `default="rtc"` to `default="sequential"` on the `--inference_algorithm` argument.

For debugging sessions, prefer to just pass `--inference_algorithm sequential` explicitly so the default stays as the "real" one.

## Use a non-GPU machine for debugging

Three options:

1. **Sequential + CPU build of PyTorch** — `policy.to("cpu")` will be slow but functional. The actors check `torch.cuda.is_available()` and pick `cpu` if False. You'll still need Ray:

   ```bash
   ray start --head --port=6379
   # Don't add inference_pc — sequential_actor's @ray.remote(num_gpus=1) line still requires 1 GPU.
   ```

   Note: `@ray.remote(num_gpus=1)` will **fail to schedule** on a CPU-only cluster. Edit [sequential_actor.py:18](../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L18) to `@ray.remote` (no `num_gpus`) for local debugging. Don't commit that change.

2. **Skip Ray entirely** — see [01_first_run.md § A targeted dry test](walkthroughs/01_first_run.md#a-targeted-dry-test).

3. **Skip the policy** — write a stub that returns zeros. Useful for testing the control plane without GPU.

## Change normalization stats

**Default**: `norm_stats_file_path` in [inference_runtime_params.json](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json) points at a `dataset_stats.pkl` in the trainer's experiments directory.

**Edit**: the JSON's `norm_stats_file_path` to the absolute path of your pickle.

**The pickle must contain** (IGRIS_B normalizer expects these keys — see [data_normalization_manager.py](../env_actor/nom_stats_manager/robots/igris_b/data_normalization_manager.py)):

```python
{
    "observation.state":   {"mean": np.ndarray, "std": np.ndarray},   # arm joints
    "observation.current": {"mean": np.ndarray, "std": np.ndarray},   # additional state
    "action":              {"mean": np.ndarray, "std": np.ndarray},
}
```

If your pickle has different keys, edit the bridge — not the interface.

**Retest**: `RuntimeParams.read_stats_file()` returns the dict; if it prints `File not found at: ...` your path is wrong.

## Change slew-rate limit

**Default**: 5 degrees per control step.

**Edit**: [inference_runtime_params.json](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json) → `"max_delta_deg": <N>`.

The bridge does `np.deg2rad(max_delta_deg)` once at startup ([inference_runtime_params.py:10](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.py#L10)). Larger = faster motion (and more risk); smaller = sluggish.

At 20 Hz, `5 deg/step ≈ 100 deg/s`. For most manipulation that's already fast.

**Retest**: with the robot ready, watch a published joint command vs the reported `prev_joint` to confirm the clip is firing.
