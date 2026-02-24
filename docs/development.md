# Development Guide

## Environment Setup

```bash
# Clone with submodules
git clone --recurse-submodules <repo-url>
cd inference_engine

# Create virtual environment
bash uv_setup.sh
source .venv/bin/activate

# Install dependencies
bash env_setup.sh

# If submodule wasn't initialized:
git submodule update --init --recursive
```

The `uv` package manager is used instead of pip. After setup, use `uv pip install` to add new dependencies.

## Project Structure Conventions

### Directory Organization

- **Robot-specific code** goes under `robots/{robot_name}/` subdirectories
- **Interfaces/factories** live at the module root (e.g., `controller_interface.py`, `data_normalization_interface.py`)
- **Protocol definitions** live in `policy/templates/`
- **Configuration** is YAML for model/policy definitions, JSON for runtime parameters

### Naming Conventions

- Robot-specific implementations: `*_bridge.py` (e.g., `controller_bridge.py`, `data_manager_bridge.py`, `shm_manager_bridge.py`)
- Factory/interface files: `*_interface.py`
- Ray actors: `*_actor.py`
- Config classes: `RuntimeParams` in `inference_runtime_params.py`

### Design Patterns

- **Factory pattern**: String-based robot selection via `if/elif` blocks in interface files
- **Protocol (structural subtyping)**: The `Policy` interface uses Python's `Protocol` — no inheritance required
- **Registry**: Decorator-based registration via `@POLICY_REGISTRY.register("name")`
- **Bridge pattern**: Robot-specific implementations behind a common interface

## Git Submodule: trainer/

The `trainer/` directory is a git submodule pointing to a separate repository. It provides:
- `PolicyConstructorModelFactory` — builds `nn.Module` models from YAML
- `load_config()` — YAML configuration loading with defaults composition
- Model block definitions and graph construction

**Important:** Do not modify files inside `trainer/` directly. Changes to the training framework should be made in its own repository and pulled via submodule update.

```bash
# Update submodule to latest
cd trainer
git pull origin main
cd ..
git add trainer
git commit -m "Update trainer submodule"
```

### Running Trainer Tests

```bash
cd trainer
pytest  # Uses testpaths = tests from pytest.ini
```

The policy constructor also has its own tests:

```bash
cd trainer/policy_constructor
pytest tests/
```

## Adding New Components Checklist

### New Policy

- [ ] Create `env_actor/policy/policies/{name}/{name}.py` with `@POLICY_REGISTRY.register("{name}")`
- [ ] Implement all `Policy` protocol methods: `__init__`, `predict`, `guided_inference`, `warmup`, `freeze_all_model_params`
- [ ] Create `env_actor/policy/policies/{name}/{name}.yaml` with `policy.type: {name}`
- [ ] Create component YAML(s) under `env_actor/policy/policies/{name}/components/`
- [ ] Test with: `python run_inference.py --robot igris_b -P env_actor/policy/policies/{name}/{name}.yaml`

### New Robot

- [ ] Create `env_actor/runtime_settings_configs/robots/{name}/init_params.py`
- [ ] Create `env_actor/runtime_settings_configs/robots/{name}/inference_runtime_params.json`
- [ ] Create `env_actor/runtime_settings_configs/robots/{name}/inference_runtime_topics.json`
- [ ] Create `env_actor/runtime_settings_configs/robots/{name}/inference_runtime_params.py`
- [ ] Create `env_actor/robot_io_interface/robots/{name}/controller_bridge.py`
- [ ] Create `env_actor/auto/inference_algorithms/sequential/data_manager/robots/{name}/data_manager_bridge.py`
- [ ] Create `env_actor/auto/inference_algorithms/rtc/data_manager/robots/{name}/shm_manager_bridge.py`
- [ ] Create `env_actor/nom_stats_manager/robots/{name}/data_normalization_manager.py`
- [ ] Update factory `if/elif` blocks in:
  - `env_actor/robot_io_interface/controller_interface.py`
  - `env_actor/auto/inference_algorithms/sequential/data_manager/data_manager_interface.py`
  - `env_actor/auto/inference_algorithms/rtc/data_manager/shm_manager_interface.py`
  - `env_actor/nom_stats_manager/data_normalization_interface.py`
- [ ] Add `"{name}"` to `choices` in `run_inference.py` argparse

### New Inference Algorithm

- [ ] Create `env_actor/auto/inference_algorithms/{name}/`
- [ ] Implement a Ray remote actor class with a `start()` method
- [ ] Create robot-specific data manager bridges under `data_manager/robots/`
- [ ] Add the algorithm name to `--inference_algorithm` choices in `run_inference.py`
- [ ] Add the import and actor creation logic in `run_inference.py`

## Ray Cluster Management

```bash
# Start cluster (edit hostnames/IPs in start_ray.sh first)
bash start_ray.sh

# Check cluster status
ray status

# View Ray dashboard (default port 8265 on head node)
# http://<HEAD_IP>:8265

# Stop Ray on current machine
ray stop
```

## Debugging Tips

### Running Without Ray

The `SequentialActor` can be instantiated directly for debugging without a full Ray cluster. You can also use `sequential_actor.py` as a reference for setting up a standalone test script.

### Common Environment Issues

**CUDA not visible:**
```bash
nvidia-smi                          # Check GPU availability
echo $CUDA_VISIBLE_DEVICES          # Check if GPUs are filtered
python -c "import torch; print(torch.cuda.is_available())"
```

**ROS2 not sourced:**
```bash
source /opt/ros/<distro>/setup.bash  # Source ROS2 before running
```

**Multiprocessing errors:**
The entrypoints set `torch.multiprocessing.set_start_method("spawn")`. If you call inference from your own script, ensure this is set before any CUDA operations.
