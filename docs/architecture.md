# Architecture

## System Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CLI ENTRYPOINT                                                          │
│  run_inference.py / run_inference_openpi.py                              │
│  - Parse args (robot, algorithm, config paths)                           │
│  - torch.multiprocessing.set_start_method("spawn")                       │
│  - ray.init(address="auto", namespace="inference_engine")                │
│  - Create Ray actor → call actor.start.remote()                          │
│  - signal.pause() until shutdown                                         │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │                                     │
    ┌───────────▼───────────┐           ┌─────────────▼─────────────┐
    │  SequentialActor       │           │  RTCActor                  │
    │  (Ray remote)          │           │  (Ray remote)              │
    │                        │           │                            │
    │  Single-threaded       │           │  Creates 2 processes:      │
    │  control loop          │           │  ┌────────────────────┐    │
    │                        │           │  │ ControlLoop (20Hz) │    │
    │  predict() every       │           │  │ read → publish     │    │
    │  policy_update_period  │           │  └────────┬───────────┘    │
    │  steps                 │           │           │ shared memory  │
    │                        │           │  ┌────────▼───────────┐    │
    │                        │           │  │ InferenceLoop      │    │
    │                        │           │  │ guided_inference() │    │
    │                        │           │  └────────────────────┘    │
    └───────────┬───────────┘           └─────────────┬─────────────┘
                │                                     │
                └──────────────────┬──────────────────┘
                                   │
    ┌──────────────────────────────▼──────────────────────────────────┐
    │  POLICY LAYER                                                   │
    │                                                                 │
    │  build_policy(yaml_path)                                        │
    │  ├─ load_policy_config()          ← YAML with defaults          │
    │  ├─ PolicyConstructorModelFactory  ← Build nn.Module components │
    │  ├─ torch.load(checkpoint)         ← Load weights               │
    │  └─ POLICY_REGISTRY.get(type)      ← Instantiate policy class   │
    │                                                                 │
    │  Policy Protocol:                                               │
    │  ├─ predict(obs, normalizer)            → (chunk_size, act_dim) │
    │  ├─ guided_inference(obs, ...)          → (chunk_size, act_dim) │
    │  ├─ warmup()                            → CUDA kernel selection │
    │  └─ freeze_all_model_params()           → Disable gradients     │
    └──────────────────────────────┬──────────────────────────────────┘
                                   │
    ┌──────────────────────────────▼──────────────────────────────────┐
    │  ROBOT I/O LAYER                                                │
    │                                                                 │
    │  ControllerInterface(robot="igris_b")                           │
    │  ├─ read_state()      → {proprio, head, left, right}           │
    │  ├─ publish_action()  → slew-rate limited ROS2 publish         │
    │  ├─ start_state_readers() → background camera + ROS2 threads   │
    │  └─ init_robot_position() → move to home pose                  │
    │                                                                 │
    │  DataNormalizationInterface(robot="igris_b")                    │
    │  ├─ normalize_state()      → z-score proprio, scale images     │
    │  └─ denormalize_action()   → inverse z-score                   │
    └─────────────────────────────────────────────────────────────────┘
```

## Execution Flow

### Startup

1. User runs `python run_inference.py --robot igris_b --inference_algorithm rtc`
2. `torch.multiprocessing.set_start_method("spawn")` is set (required for CUDA in subprocesses)
3. `ray.init(address="auto", namespace="inference_engine")` connects to an existing Ray cluster
4. An `RTCActor` (or `SequentialActor`) is created as a Ray remote actor with resource requirements
5. `actor.start.remote()` triggers the actor's initialization and control loop
6. The main process blocks on `signal.pause()` until interrupted

### Actor Initialization (inside `start()`)

1. `build_policy(policy_yaml_path)` loads the policy:
   - YAML is parsed, component paths resolved
   - `PolicyConstructorModelFactory` builds `nn.Module` components
   - Checkpoint state dicts loaded from `ckpt_dir`
   - Policy class resolved via `POLICY_REGISTRY`
2. `ControllerInterface(robot=...)` creates the robot I/O bridge
3. `DataNormalizationInterface(robot=...)` creates the normalization bridge
4. Data managers initialized (observation history buffers, action buffers)
5. Policy moved to GPU, `freeze_all_model_params()`, `warmup()`
6. CUDA optimizations enabled: `cudnn.benchmark = True`, `set_float32_matmul_precision("high")`

### Data Flow (per control step)

```
USB Cameras ──────────┐
                      ├──→ read_state() ──→ {proprio, head, left, right}
ROS2 Joint Topics ────┘                              │
                                                      ▼
                                              normalize_state()
                                              ├─ proprio: (x - μ) / (σ + ε)
                                              └─ images: x / 255.0
                                                      │
                                                      ▼
                                              policy.predict()  or
                                              policy.guided_inference()
                                                      │
                                                      ▼
                                              denormalize_action()
                                              └─ action * σ + μ
                                                      │
                                                      ▼
                                              publish_action()
                                              ├─ slew-rate limit (max_delta)
                                              ├─ ROS2 /target_joints
                                              └─ ROS2 /finger_target
```

## RTC vs Sequential: Detailed Comparison

### Sequential

```
Time ──────────────────────────────────────────────────→

Control: read → predict ─────────→ publish → read → [use cached action] → publish → ...
                 (blocks)           action[0]                                action[1]
```

The control loop blocks while the policy runs. No actions are published during inference.

### RTC (Real-Time Control)

```
Time ──────────────────────────────────────────────────→

Control:  read → publish → read → publish → read → publish → read → publish → ...
          action[i]        action[i+1]      action[i+2]      new_action[0]

Inference:        ╰── guided_inference() ──────────────────╯
                       (runs in parallel via shared memory)
```

The control loop never blocks. While inference runs, the control loop continues executing actions from the previous chunk. When the new chunk is ready, action inpainting ensures a smooth transition.

### Shared Memory Layout (RTC)

```
┌─────────────────────────────────────────────────┐
│  Shared Memory Region                           │
│                                                 │
│  proprio  [1 × 24]        float32   ← control  │
│  head     [1 × 3 × 240 × 320]  uint8  ← control  │
│  left     [1 × 3 × 240 × 320]  uint8  ← control  │
│  right    [1 × 3 × 240 × 320]  uint8  ← control  │
│  action   [50 × 24]       float32  ← inference │
│                                                 │
│  Sync: RLock + 2 Conditions + 2 Events + Values │
└─────────────────────────────────────────────────┘
```

- **Control loop writes**: proprio, head, left, right (latest observation)
- **Inference loop writes**: action (new action chunk)
- **Control loop reads**: action (current action to execute)
- **Inference loop reads**: proprio, head, left, right (observation for inference)

## Performance Optimizations

| Optimization | Where | Effect |
|---|---|---|
| `torch.backends.cudnn.benchmark = True` | Actor init | Auto-selects fastest CUDA kernels |
| `torch.set_float32_matmul_precision("high")` | Actor init | Hardware-accelerated matrix operations |
| `torch.inference_mode()` | Inference loop | Disables autograd bookkeeping |
| `torch.autocast("cuda", dtype=torch.bfloat16)` | Inference loop | Mixed-precision inference |
| `torch.compile` warmup | Policy `warmup()` | JIT-compiled model forward pass |
| `policy.freeze_all_model_params()` | Actor init | No gradient memory allocation |
| Shared memory (RTC) | Inter-process | Zero-copy data transfer |
| `multiprocessing.set_start_method("spawn")` | Entrypoint | Safe CUDA fork handling |
