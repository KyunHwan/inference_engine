# Runtime Config

Runtime JSON validation and loading for inference algorithms.

Files
- `config_loader.py`: loads JSON, optionally merges `inference_settings.json`, and validates with `RuntimeConfig`.
- `runtime_schema.py`: pydantic schema for `robot_id`, `HZ`, `image_resize`, `camera_names`, `topics`, and `inference` settings.
- `__init__.py`: package marker.

Behavior
- Validates message types and field rules; all fields must be `/observation/...` keys.
- Disallows `/observation/images/*` in topics because images are captured from cameras directly.
- Optional `inference_settings.json` in the same directory overrides the `inference` block.

Accessors
- `get_camera_names`, `get_image_resize`, `get_observation_keys`, and `get_inference_settings` provide a stable API for controllers.
