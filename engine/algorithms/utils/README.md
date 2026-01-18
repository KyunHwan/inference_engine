# Shared Utilities

Common helpers used by both inference pipelines to keep control-loop plumbing consistent.

Files
- `data_dict.py`: `GenericRecorder` subscribes to configured ROS2 topics, extracts fields, and stores the latest observation values. It exposes `get_dict` and `get_observation_dict` and prints timing warnings when data is stale.
- `camera_utils.py`: `RBRSCamera` wraps two V4L2 devices, runs a background capture thread, and returns a merged RGB frame.
- `inference_utils.py`:
  - Shared-memory helpers (`ShmArraySpec`, `attach_shared_ndarray`, `init_shared_action_chunk`).
  - Runtime config accessors (`get_runtime_config_params`, `get_model_io_params`).
  - Signal handling (`make_signal_handler`) to wake processes blocked on condition variables.
  - Motion smoothing setup (`motion_smoothing_setup`) and delay estimation (`MaxDeque`).
- `picknplace.json`: example runtime JSON for a pick-and-place task.

Operational notes
- Image observations are captured directly from cameras; runtime JSON topics must not include `/observation/images/*`.
- Shared-memory blocks are created in the parent process and only attached in children.
- The initial action chunk is seeded with a safe posture to avoid sudden jumps at startup.
