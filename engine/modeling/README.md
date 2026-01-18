# Modeling

Model construction is delegated to policy_constructor via a thin adapter. This layer resolves component configs, loads checkpoints, and prepares modules for inference.

Key files
- `factories.py`: `PolicyConstructorModelFactory` calls `policy_constructor.model_constructor.build_model`.
- `build_models.py`: resolves component configs, builds modules, loads checkpoints, moves to device, and sets eval mode.

ModelBuildRequest
- `component_config_paths`: mapping from component name to policy_constructor config path.
- `checkpoint_path`: file or directory; if directory, checkpoints are expected as `{name}.pt`.
- `component_checkpoint_paths`: optional per-component checkpoint overrides.
- `device`, `strict`, `eval_mode`: torch loading and runtime flags.

Behavior
- Single-component builds are normalized to `{"policy": module}` for downstream builders.
- Checkpoint paths can be absolute or relative to the checkpoint root.
- All components are moved to the requested device and set to eval mode by default.
