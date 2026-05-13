# FAQ

Common questions a new engineer asks in the first month. Each answer points at the relevant code or doc.

## Table of contents

- [What is the difference between `predict()` and `guided_inference()`?](#what-is-the-difference-between-predict-and-guided_inference)
- [Why is `action_dim = 24`?](#why-is-action_dim--24)
- [Do I need a real robot to read or run anything?](#do-i-need-a-real-robot-to-read-or-run-anything)
- [Can I run this on a single machine?](#can-i-run-this-on-a-single-machine)
- [Where do checkpoints come from?](#where-do-checkpoints-come-from)
- [How do I know my policy is actually running on GPU?](#how-do-i-know-my-policy-is-actually-running-on-gpu)
- [Why does the system use `torch.multiprocessing` instead of plain `multiprocessing`?](#why-does-the-system-use-torchmultiprocessing-instead-of-plain-multiprocessing)
- [What is the `inference_pc` resource?](#what-is-the-inference_pc-resource)
- [Can I add a new policy without modifying `run_inference.py`?](#can-i-add-a-new-policy-without-modifying-run_inferencepy)
- [Can I add a new robot without modifying `run_inference.py`?](#can-i-add-a-new-robot-without-modifying-run_inferencepy)
- [What does `torch.compile` do here and is it on by default?](#what-does-torchcompile-do-here-and-is-it-on-by-default)
- [Why is the RTC algorithm split into two processes rather than two threads?](#why-is-the-rtc-algorithm-split-into-two-processes-rather-than-two-threads)
- [How do I read the Ray dashboard?](#how-do-i-read-the-ray-dashboard)
- [Where do I find logs?](#where-do-i-find-logs)
- [What happens if the inference loop is slower than 20 Hz?](#what-happens-if-the-inference-loop-is-slower-than-20-hz)
- [Why does RTC wait 100 control steps before publishing?](#why-does-rtc-wait-100-control-steps-before-publishing)

## What is the difference between `predict()` and `guided_inference()`?

- `predict(obs, normalizer) → (chunk_size, action_dim)` — used by the **Sequential** algorithm. Standard single-shot inference, no action inpainting.
- `guided_inference(obs, normalizer, executed, chunk_size) → (chunk_size, action_dim)` — used by **RTC**. Takes the same observation plus `obs["prev_action"]` and `obs["est_delay"]`, runs the policy, then blends the new chunk with the previous one using [`compute_guided_prefix_weights`](../env_actor/inference_engine_utils/action_inpainting.py).

The wrapper in both shipped policies forwards `prev_action` and `est_delay` straight through to the inpainting function; the model itself doesn't know about RTC. See [concepts.md § Action inpainting](concepts.md#action-inpainting).

## Why is `action_dim = 24`?

IGRIS_B is a dual-arm robot. Each arm has 6 controllable joints; each hand has 6 controllable fingers. `24 = 6 + 6 + 6 + 6`. The action vector layout, per [controller_bridge.py:85-112](../env_actor/robot_io_interface/robots/igris_b/controller_bridge.py#L85-L112):

| Indices | Meaning |
|---|---|
| `[0:6]` | Left arm joints |
| `[6:12]` | Right arm joints |
| `[12:18]` | Left hand fingers |
| `[18:24]` | Right hand fingers |

The `publish_action` method then concatenates `[right_joints, left_joints]` for the ROS2 message (the on-robot joint order differs from the policy's output order).

## Do I need a real robot to read or run anything?

**To read code** — no. Nothing to install beyond a text editor.

**To run anything** — a GPU and ROS2 are required for the policy-loading and CUDA-warmup steps, but you can stop before camera initialization succeeds. So:

- No robot, no ROS2: you can install dependencies and read everything.
- GPU + ROS2 sourced, no robot: you can `build_policy()`, `policy.to("cuda")`, `policy.warmup()` — the run will fail when `RBRSCamera` can't open `/dev/head_camera1`. That's a reasonable stopping point for a "policy still loads after my change" sanity check.
- Robot + GPU + ROS2: full end-to-end.

See [walkthroughs/01_first_run.md](walkthroughs/01_first_run.md) for what each stage looks like.

## Can I run this on a single machine?

Yes — start both the head and the worker on the same host:

```bash
ray start --head --port=6379
ray start --address=127.0.0.1:6379 --resources='{"inference_pc": 1}'
```

[start_ray.sh](../start_ray.sh) is hostname-keyed; for single-machine work just bypass it and run those commands directly, or add a third case for your hostname.

## Where do checkpoints come from?

From [trainer/](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/README.md). The training runs save `.pt` files per component. Two paths matter:

- **DSRL components** (`backbone.pt`, `noise_processor.pt`, `noise_actor.pt`) — loaded by `DsrlOpenpiPolicy` from `params.checkpoint_path` in [dsrl_openpi_policy.yaml](../env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.yaml).
- **OpenPI weights** — loaded by `OpenPiBatchedWrapper.__init__` (inside the trainer submodule) from `params.ckpt_dir` in the component YAML.

This separation lets DSRL fine-tune on top of a frozen OpenPI backbone without needing the OpenPI weights in the DSRL checkpoint.

## How do I know my policy is actually running on GPU?

After the policy is built:

```python
next(policy.parameters()).device       # should be cuda:0
next(policy.parameters()).dtype        # bfloat16 or float32 depending on autocast
```

Both actors do `build_policy(...).to(self.device)` with `device = cuda if torch.cuda.is_available() else cpu`. While inference is running, `nvidia-smi` in another terminal should show the Python process and GPU utilization above zero.

## Why does the system use `torch.multiprocessing` instead of plain `multiprocessing`?

`torch.multiprocessing` is a drop-in wrapper that adds CUDA-tensor-safe sharing semantics (in case you ever pass a tensor across processes). The set-start-method call is needed regardless. See [concepts.md § Spawn vs fork](concepts.md#spawn-vs-fork).

## What is the `inference_pc` resource?

A Ray *custom resource* — a string label a worker advertises. Inference actors request `resources={"inference_pc": 1}` so Ray schedules them only on workers that advertise it. In our cluster the GPU machine declares `--resources='{"inference_pc": 1}'`; the head node does not. So inference actors always land on the GPU box.

It's not built into Ray — it's just a name we picked. If you renamed it everywhere, nothing would behave differently.

## Can I add a new policy without modifying `run_inference.py`?

Yes. Drop a new directory under `env_actor/policy/policies/your_policy/` with `your_policy.py` (decorated with `@POLICY_REGISTRY.register("your_policy")`) and `your_policy.yaml` (with `policy.type: your_policy`). Then run:

```bash
python run_inference.py --robot igris_b -P env_actor/policy/policies/your_policy/your_policy.yaml
```

`build_policy` will auto-import the module the first time it sees the new registry key (see [loader.py:71-72](../env_actor/policy/utils/loader.py#L71)).

Full walkthrough: [walkthroughs/03_add_a_new_policy.md](walkthroughs/03_add_a_new_policy.md).

## Can I add a new robot without modifying `run_inference.py`?

Honest answer: **you need to touch one line.** `run_inference.py` has `--robot ... choices=["igris_b", "igris_c"]`. You have to add `"your_robot"` to that list, or argparse will reject the argument.

After that, you create the four robot-specific files (controller bridge, runtime configs, two data managers, normalization bridge) and update four `if/elif` blocks in the corresponding interfaces. Full checklist: [walkthroughs/04_add_a_new_robot.md](walkthroughs/04_add_a_new_robot.md).

## What does `torch.compile` do here and is it on by default?

`torch.compile` (PyTorch 2.x) traces a `nn.Module`'s `forward` and emits an optimized fused kernel. The OpenPI wrapper's `warmup(batch_size=1)` triggers it, so it's on for the OpenPI/DSRL policies any time you actually run inference.

If you skip the `warmup()` call (it's wrapped in `try/except` in the actors so warmup failures don't crash startup), the model still works but uses uncompiled kernels.

## Why is the RTC algorithm split into two processes rather than two threads?

Python's GIL would serialize the inference forward pass with the control loop in a threaded implementation, defeating the point. By using two processes:

- The control loop's GIL is private — it can do its 20 Hz `read_state` / `publish_action` work without waiting for the inference forward pass.
- The inference process drives CUDA from its own thread of execution, with its own CUDA context.

The cost is shared-memory plumbing instead of shared-object references. See [rtc_shared_memory.md](rtc_shared_memory.md).

## How do I read the Ray dashboard?

Open `http://<HEAD_IP>:8265` in a browser. With `start_ray.sh`'s `--dashboard-host=0.0.0.0` you can reach it from any machine on the same network as the head.

What to look at:

- **Cluster** → confirms each node is alive and its declared resources (you should see `inference_pc: 1.0` on the GPU worker).
- **Jobs** / **Actors** — you'll see your single `RTCActor` (or `SequentialActor`).
- Click into the actor → **Logs** tab — that's where the `print(...)` from `start_inference` / `start_control` shows up.

## Where do I find logs?

- **`print()` from the actor and from the spawned children** — Ray captures stdout. View via the Ray dashboard's "Logs" tab, or in `~/ray_results/<run>/`, or in the terminal where you ran `run_inference.py` (because `log_to_driver=True` is set).
- **`ros2 topic echo`** — for raw ROS2 messages flowing in/out of the robot.
- **`nvidia-smi -l 1`** — for live GPU utilization.

## What happens if the inference loop is slower than 20 Hz?

That's the entire point of RTC. The control loop continues publishing buffered actions from the previous chunk regardless of how long inference takes. When the new chunk eventually lands in shared memory, the next control step reads it.

`MaxDeque` ([max_deque.py](../env_actor/auto/inference_algorithms/rtc/data_manager/utils/max_deque.py)) tracks the rolling max of "control steps since last inference," and that feeds `est_delay` into action inpainting so the blend covers the delay. If inference is consistently slower than the chunk size, the robot would run off the end of the buffer — action inpainting can't paper over hours of latency. In practice inference is sub-second.

## Why does RTC wait 100 control steps before publishing?

In [control_loop.py:133](../env_actor/auto/inference_algorithms/rtc/actors/control_loop.py#L133) the publish call is guarded by `if t > 100`. That's a 5-second warmup at 20 Hz before any joint commands are sent to the robot, giving the inference loop time to finish its first `guided_inference` call and write a real chunk into shared memory. Until then the buffer holds the init action vector ([`SharedMemoryManager.init_action_chunk_obs_history`](../env_actor/auto/inference_algorithms/rtc/data_manager/robots/igris_b/shm_manager_bridge.py)).
