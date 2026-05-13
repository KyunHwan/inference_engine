# Troubleshooting

Grouped by symptom. Each entry: what you see → what's happening → what to do.

## Table of contents

- [CUDA](#cuda)
- [Ray](#ray)
- [ROS2](#ros2)
- [Cameras](#cameras)
- [Shared memory](#shared-memory)
- [Policy loading](#policy-loading)
- [Dependency installation](#dependency-installation)
- [Submodules](#submodules)

## CUDA

### `Cannot re-initialize CUDA in forked subprocess`

`torch.multiprocessing.set_start_method("spawn")` was not called, or was called after CUDA had already been initialized.

The fix in normal use: nothing — [run_inference.py](../run_inference.py) does it for you at the top. If you wrote your own entrypoint, make sure your first non-trivial line is:

```python
import torch
try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass
```

…and that no CUDA operation runs before it. See [concepts.md](concepts.md#spawn-vs-fork).

### `torch.cuda.is_available()` returns `False`

```bash
nvidia-smi                  # do you see a GPU?
echo $CUDA_VISIBLE_DEVICES   # is it set to "" or "-1"?
python -c "import torch; print(torch.version.cuda)"  # was torch built against your CUDA?
```

If `nvidia-smi` works but PyTorch doesn't see the GPU, the most common cause is `CUDA_VISIBLE_DEVICES=""` left over from a benchmarking script. Unset it (`unset CUDA_VISIBLE_DEVICES`).

If `nvidia-smi` doesn't work, install or update your NVIDIA driver. The repo expects CUDA 13.0 wheels (`torch==2.9.0+cu130`).

### `Out of memory` during warmup

The OpenPI/Pi0.5 backbone is large (multi-GB). If your GPU is shared, free it:

```bash
nvidia-smi          # find offending PIDs
kill <pid>          # or fuser -k /dev/nvidia*
```

The inference engine doesn't currently support multi-GPU sharding for a single policy.

## Ray

### `ConnectionError: Could not find any running Ray instance`

You started inference before starting Ray. Run [start_ray.sh](../start_ray.sh) (or `ray start --head`) and try again.

### `Resource 'inference_pc' not available`

The worker that has the GPU did not register `inference_pc:1`. Two cases:

1. **Multi-machine**: the worker hostname doesn't match any case in [start_ray.sh](../start_ray.sh). Add a case for your hostname, or just run the worker command directly:

   ```bash
   ray start --address=<HEAD_IP>:6379 --resources='{"inference_pc": 1}'
   ```

2. **Single machine**: you only ran `ray start --head` and forgot the worker. The head node doesn't have `inference_pc:1` by default. Run a second:

   ```bash
   ray start --address=127.0.0.1:6379 --resources='{"inference_pc": 1}'
   ```

`ray status` shows what each node is offering.

### `ray.init` hangs forever

Likely a firewall problem. Ray needs ports 6379 (GCS), 10001 (client server), and a range for object manager / node manager. On the same machine make sure nothing else is binding 6379. Across machines, check that the worker can reach the head's `--node-ip-address`.

### Dashboard at `http://<head>:8265` not loading

`start_ray.sh` passes `--dashboard-host=0.0.0.0` on the head node so the dashboard is reachable from other hosts. If the page doesn't load, the dashboard probably failed to install. Try:

```bash
uv pip install "ray[default]"
```

`ray[default]` (with the extras) is needed for the dashboard.

## ROS2

### `rclpy: not found` / ROS2 not sourced

You see `ModuleNotFoundError: No module named 'rclpy'` (or `sensor_msgs`, `std_msgs`, `geometry_msgs`).

Source ROS2 before launching:

```bash
source /opt/ros/<distro>/setup.bash
python run_inference.py ...
```

For convenience, append the source line to `.venv/bin/activate` or to a wrapper script. This needs to happen in *every* shell that runs Python from this repo.

### Subscriptions never deliver data

`controller_bridge.py` waits in `_check_proprio_reading` and logs `Waiting for proprio data to come in...` repeatedly. Possibilities:

- The `robot_id` in [inference_runtime_topics.json](../env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_topics.json) doesn't match the actual robot's namespace. Default is `"packy"`; check with `ros2 topic list`.
- The publisher process isn't running on the same ROS2 domain. Check `ROS_DOMAIN_ID`.
- QoS mismatch — the bridge uses `Reliable`/`KeepLast(10)`. If your publisher uses `BestEffort`, change one side.

## Cameras

### `Camera with device ID ... could not be opened`

[`RBRSCamera.singleCamera`](../env_actor/robot_io_interface/robots/igris_b/utils/camera_utils.py) raises this if `cv2.VideoCapture(device, CAP_V4L2)` fails. Check:

```bash
ls /dev/head_camera1 /dev/left_camera2 /dev/right_camera1
```

If a path is missing, your udev rules haven't fired. Either:

- Plug the camera in.
- Add a udev rule that aliases the right `/dev/video*` to the expected name.
- Hardcode `/dev/videoN` in the bridge for development (don't commit that).

### Black frames / `image is None !!`

The camera opened but `cap.read()` returned `False`. Common causes:

- Another process is holding the camera (e.g., `cheese`, `mpv`, a previous run that didn't clean up).
- The camera is busy on its USB controller — try a different port.
- The requested resolution/FPS isn't supported. The bridge sets MJPG@1600x1200@60. Use `v4l2-ctl --list-formats-ext -d /dev/head_camera1` to confirm.

## Shared memory

### `FileNotFoundError: [Errno 2] No such file or directory: '/...'` referencing `/dev/shm/<random>`

A previous run was killed before it could unlink its shared memory blocks. Linux leaves them in `/dev/shm/`. Clean up:

```bash
ls /dev/shm/
# Look for entries that start with "psm_" or "wnsm_" — these are Python multiprocessing names.
rm /dev/shm/psm_*    # only if no other Python process is currently using them
```

You can also reboot — `/dev/shm/` is a tmpfs.

### `resource_tracker: There appear to be N leaked shared_memory objects to clean up at shutdown`

The parent process didn't get to `cleanup()`. The warning is benign (the OS will reclaim them when the tmpfs is unmounted or on reboot) but indicates a missed cleanup. The RTCActor `finally:` block already does its best; if you see this often, look for an exception in the children that's being swallowed.

### Deadlock / control loop stuck

`wait_for_inference_ready` or `wait_for_min_actions` is waiting on a condition that will never fire. Possible causes:

- The inference process died silently. Check its stdout — it's logged via Ray.
- The `stop_event` was set but the condition's `notify_all` didn't get called. The cleanup paths set + notify; if you added a new exit path, make sure it does both.

See [rtc_shared_memory.md § Failure modes](rtc_shared_memory.md#failure-modes).

## Policy loading

### `ValueError: model.component_config_paths must be a non-empty mapping`

Your policy YAML is missing the `model.component_config_paths` block. Compare to [openpi_policy.yaml](../env_actor/policy/policies/openpi_policy/openpi_policy.yaml).

### `KeyError: 'policy' registry has no key 'my_policy'`

`build_policy` tried to auto-import `env_actor.policy.policies.my_policy.my_policy` and the module didn't register itself. Check:

1. Does the file exist at that exact path?
2. Does it have `@POLICY_REGISTRY.register("my_policy")` decorating the class?
3. Is the import side-effect actually running? Add `print("registering my_policy")` at module top to verify.

### `FileNotFoundError: ... .pt`

`build_policy` is loading `checkpoint_path/<component>.pt` and that file doesn't exist. Check `checkpoint_path` in your policy YAML. The DSRL policy expects `backbone.pt`, `noise_processor.pt`, `noise_actor.pt` under its `params.checkpoint_path` — but **not** `openpi_model.pt`. OpenPI loads its own weights from `ckpt_dir` inside [components/openpi_model.yaml](../env_actor/policy/policies/dsrl_openpi_policy/components/openpi_model.yaml).

### `AttributeError` inside OpenPI / transformers

You forgot to run [openpi_transformer_lib_patch.sh](../openpi_transformer_lib_patch.sh) after installing dependencies. Re-run it and try again.

## Dependency installation

### `depth_anything_3` editable install fails

`env_setup.sh`'s last step is:

```bash
uv pip install -e ./trainer/policy_constructor/.../experiments/backbones/vision/externals/depth_anything_3
```

If you didn't clone with `--recurse-submodules`, that directory is empty. Run:

```bash
git submodule update --init --recursive
bash env_setup.sh   # re-run; uv will skip already-installed packages
```

### `torch==2.9.0+cu130` wheel not found

Your network can't reach `https://download.pytorch.org/whl/cu130`. Try a different network or override the index URL:

```bash
uv pip install torch==2.9.0 torchvision==0.24.0 --index-url <your-mirror>
```

If `cu130` is wrong for your driver, lower it (`cu124`, `cu121`), but be aware no one has verified the rest of the stack works against an older CUDA.

## Submodules

### `Permission denied (publickey)` on submodule init

The default URLs use SSH. Switch to HTTPS:

```bash
git submodule set-url trainer https://github.com/KyunHwan/trainer.git
git submodule update --init
cd trainer
git submodule set-url policy_constructor https://github.com/KyunHwan/policy_constructor.git
git submodule update --init
cd ..
```

### Submodule shows as "modified" in `git status` but you didn't touch it

You're inside the submodule and made commits or have local changes. `git status` from the outer repo reports the new SHA as a pending change.

- If you really meant to bump the submodule, run `git add trainer` and commit the new SHA from the outer repo.
- If you didn't, `cd trainer && git stash` or `git reset --hard <pinned-sha>`.

The currently pinned SHAs are in [.gitmodules](../.gitmodules) (URL only) plus `git submodule status` (SHA). At the time of writing: `trainer` = `3ca051a`, `policy_constructor` = `00663cc`.
