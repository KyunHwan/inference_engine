# Config

Configuration for inference YAML files, with composition support and strict validation.

Files
- `loader.py`: loads YAML with `defaults` composition, resolves relative paths, and deep-merges dictionaries.
- `inference_schemas.py`: pydantic models for the inference YAML (algorithm, runtime config path, model/policy info).
- `errors.py`: `ConfigError` and `ConfigValidationIssue` for human-friendly error messages.

Flow
1) `load_config` reads YAML and expands `defaults` entries.
2) `validate_inference_config` returns a typed `InferenceConfig` or raises `ConfigError`.

Notes
- `defaults` is a list of single-key mappings; cycles are detected and rejected.
- `model.component_config_paths` points to policy_constructor configs; `checkpoint_path` can be a file or directory.
- `plugins` is a list of import paths that call `POLICY_REGISTRY.add(...)` at import time.
