# Engine

The `engine` package is the runtime core of the inference system. It owns configuration validation, model/policy construction, and the ROS2 control pipelines that publish actions.

Core flow
- Load inference YAML with `engine.config.loader` and validate it with `engine.config.inference_schemas`.
- Import plugin modules so registries are populated.
- Load runtime JSON with `engine.algorithms.config.ConfigLoader`.
- Build model components via `engine.modeling.build_models` (delegates to policy_constructor).
- Wrap components into an `InferencePolicy` with normalization stats via `engine.policies`.
- Execute the selected algorithm and publish actions to ROS2 topics.

Subpackages
- `algorithms`: sequential and real-time action chunking pipelines plus shared utilities.
- `config`: YAML loader, schemas, and error formatting.
- `modeling`: model construction adapter and checkpoint loader.
- `policies`: policy builders, registry, normalization, and interfaces.
- `registry`: registry + plugin loading mechanism.
- `experiment_models`: placeholder configs for experimental model definitions.
- `utils`: reserved for cross-cutting utilities (currently empty).

Entry point
- `inference.py` at repo root selects the algorithm and invokes the appropriate pipeline.
