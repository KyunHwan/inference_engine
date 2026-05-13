# Onboarding

A day-by-day plan for your first week. The goal is for you to be able to run an inference, trace one control step end-to-end, and start a non-trivial change by the end of Week 1.

## Table of contents

- [Before you start](#before-you-start)
- [Day 1: clone, install, read](#day-1-clone-install-read)
- [Day 2: skim the code](#day-2-skim-the-code)
- [Day 3: run something](#day-3-run-something)
- [Day 4–5: trace one step](#day-45-trace-one-step)
- [Week 2: extend](#week-2-extend)
- [When you get stuck](#when-you-get-stuck)
- [Quick reference card](#quick-reference-card)

## Before you start

You will need:

- A Linux workstation (Ubuntu tested).
- An NVIDIA GPU with CUDA 13.0 drivers, **only** if you want to run inference. For read-through and code reading, no GPU is needed.
- ROS2 installed, **only** if you want to run anything that touches `rclpy`. Without it the imports will fail. You can still read and understand the controller bridges.
- Real IGRIS_B hardware, **only** for `init_robot_position()` and end-to-end testing. Without hardware you can still run a "dry" smoke test that loads the policy, warms up CUDA, and crashes when it tries to talk to the cameras. That's fine — you just stop there.
- Permission to clone the trainer and policy_constructor repos. If you don't have an SSH key registered with GitHub, see [Day 1](#day-1-clone-install-read) for the HTTPS switch.

## Day 1: clone, install, read

### 1. Clone

```bash
git clone --recurse-submodules <repo-url> inference_engine
cd inference_engine
```

If the clone fails on the submodules with `Permission denied (publickey)`, you don't have an SSH key. Switch the submodule URLs to HTTPS:

```bash
git submodule set-url trainer https://github.com/KyunHwan/trainer.git
git submodule update --init
cd trainer
git submodule set-url policy_constructor https://github.com/KyunHwan/policy_constructor.git
git submodule update --init
cd ..
```

### 2. Install `uv` and create the venv

```bash
bash uv_setup.sh
```

Expected output ends with `uv` printing the install location, plus:

```
Using CPython 3.12.3
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
```

### 3. Activate

```bash
source .venv/bin/activate
```

Your prompt should now show `(inference_engine)` or similar.

### 4. Install dependencies

```bash
bash env_setup.sh
```

This downloads PyTorch 2.9.0 (CUDA 13.0), Ray, Transformers, and the rest. Expect 5–15 minutes on a fast link. It also does `uv pip install -e ./trainer/policy_constructor/.../depth_anything_3` at the end — if that step fails, see [troubleshooting.md](troubleshooting.md#depth_anything_3-editable-install-fails).

### 5. (Optional) Patch transformers for OpenPI

The OpenPI/Pi0.5 model uses a fork of `transformers`. Apply the patch:

```bash
bash openpi_transformer_lib_patch.sh
```

This copies a handful of `.py` files from `trainer/.../transformers_replace/` over the installed `transformers` package. Without it, OpenPI inference will throw `AttributeError` on some attention modules.

### 6. Verify CUDA

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

Expected:

```
2.9.0+cu130 True 1
```

(Or however many GPUs you have.) If `False`, you don't have the right driver or `CUDA_VISIBLE_DEVICES` is filtering everything out.

### 7. Read

Open these in this order. Take 90 minutes total. You're not memorizing — you're getting the *shape*.

1. [README.md](../README.md) — top-level "what is this and how do I run it".
2. [docs/concepts.md](concepts.md) — vocabulary. Especially "VLA", "action chunks", "RTC", "shared memory".
3. [docs/architecture.md](architecture.md) — the layered diagram and the data flow per step.
4. [docs/glossary.md](glossary.md) — skim. You'll come back here.

## Day 2: skim the code

Open these files in your editor side by side. For each, the goal is "two minutes per file, what does it do, what's the entry point?" — not "read every line."

| File | What to notice |
|---|---|
| [run_inference.py](../run_inference.py) | `torch.multiprocessing.set_start_method("spawn")` happens *before* anything; `ray.init` with namespace; one actor created with `options(resources={"inference_pc": 1})`, then `start.remote()`. |
| [env_actor/auto/inference_algorithms/sequential/sequential_actor.py](../env_actor/auto/inference_algorithms/sequential/sequential_actor.py) | A simple `while True:` outer loop and a `for t in range(9000):` inner loop. Inference fires every `policy_update_period` steps. Everything else is read → buffer → publish. |
| [env_actor/policy/templates/policy.py](../env_actor/policy/templates/policy.py) | The protocol you implement when adding a policy: `__init__`, `predict`, `guided_inference`, `warmup`, `freeze_all_model_params`. |
| [env_actor/policy/utils/loader.py](../env_actor/policy/utils/loader.py) | `build_policy(yaml_path)` is the function both algorithms call. Notice the auto-import on registry miss. |
| [env_actor/robot_io_interface/controller_interface.py](../env_actor/robot_io_interface/controller_interface.py) | Just a factory + delegation. Pick a robot string → import the bridge → forward every call. |
| [env_actor/robot_io_interface/robots/igris_b/controller_bridge.py](../env_actor/robot_io_interface/robots/igris_b/controller_bridge.py) | All the ROS2 plumbing: publishers, subscribers, slew-rate limiting in `publish_action`, the action-slice layout. |
| [env_actor/auto/inference_algorithms/rtc/rtc_actor.py](../env_actor/auto/inference_algorithms/rtc/rtc_actor.py) | Allocates shared memory, creates two `spawn`-context processes, joins them. |
| [env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py](../env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py) | A thin factory; all the interesting concurrency is in the per-robot bridge. |

When you finish, you should be able to draw the architecture diagram from memory at the layer level (algorithm → policy → robot I/O) and name one or two files per layer.

## Day 3: run something

Goal: get *some* code running on the GPU without a robot.

1. Start Ray on one machine. Edit [start_ray.sh](../start_ray.sh) so both the head and worker cases run on your hostname, or just run:

   ```bash
   ray start --head --port=6379
   ray start --address=127.0.0.1:6379 --resources='{"inference_pc": 1}'
   ```

   (Two terminals, or run the second in background.) See [`ray status`](#quick-reference-card) to verify.

2. Try the dry run:

   ```bash
   python run_inference.py --robot igris_b --inference_algorithm sequential
   ```

   What you can expect *without* the robot:

   - `build_policy` will load the YAML and try to load the OpenPI checkpoint from `ckpt_dir`. If the path is wrong it will throw `FileNotFoundError`. **This is expected when you don't have the checkpoints locally.** Either point `ckpt_dir` at a real path or stop here — you've already exercised the policy-loading path.
   - If you do have the checkpoint, the policy moves to GPU, `warmup()` runs, then it tries to open the cameras and dies because `/dev/head_camera1` doesn't exist. **This is also expected.** You've now exercised everything up to robot I/O.

3. Read the full first-run guide: [walkthroughs/01_first_run.md](walkthroughs/01_first_run.md). It includes the annotated success-log output.

## Day 4–5: trace one step

Open [walkthroughs/02_trace_one_step.md](walkthroughs/02_trace_one_step.md). Read it next to the source files. By the end you should be able to point at any tensor in the data flow and say what its shape, dtype, and producer are.

Then re-read [docs/rtc_shared_memory.md](rtc_shared_memory.md). Now that you've seen the control flow, the synchronization makes sense.

## Week 2: extend

Pick one of:

- [walkthroughs/03_add_a_new_policy.md](walkthroughs/03_add_a_new_policy.md) — make a copy of `openpi_policy`, rename it `my_policy`, change one method, register, run.
- [walkthroughs/04_add_a_new_robot.md](walkthroughs/04_add_a_new_robot.md) — flesh out the IGRIS_C stubs.
- A real ticket from your team.

The two walkthroughs are the worked examples; the per-folder READMEs are your reference while you implement.

## When you get stuck

In this order:

1. [troubleshooting.md](troubleshooting.md) — symptom → cause → fix.
2. [faq.md](faq.md) — common questions.
3. The relevant per-folder README (find it via [docs/README.md § By module](README.md#by-module)).
4. A senior teammate. When you ask, paste: the exact command, the full stack trace, what you expected vs what happened, and what you already tried.

## Quick reference card

```bash
# Clone + install
git clone --recurse-submodules <url>
cd inference_engine
bash uv_setup.sh && source .venv/bin/activate
bash env_setup.sh
bash openpi_transformer_lib_patch.sh   # only if you use OpenPI

# Ray
bash start_ray.sh                       # multi-machine
ray start --head --port=6379            # single machine, head
ray start --address=127.0.0.1:6379 \
  --resources='{"inference_pc": 1}'     # single machine, worker
ray status                              # check cluster
ray stop                                # tear down on this machine

# Run inference
python run_inference.py --robot igris_b                              # RTC (default)
python run_inference.py --robot igris_b --inference_algorithm sequential  # Sequential

# Submodule
git submodule update --init --recursive                              # initial
git -C trainer pull origin main && git add trainer && git commit ... # bump
git submodule set-url trainer https://github.com/.../trainer.git     # SSH → HTTPS

# CUDA / Python sanity
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```
