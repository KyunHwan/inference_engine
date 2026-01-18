# Sequential Inference Pipeline

Single-process controller that runs policy inference and ROS2 publishing in one loop. It is selected with `inference.py --algorithm sequential`.

What it does
- Loads runtime JSON via `ConfigLoader` and inference YAML via `build_policy`.
- Subscribes to ROS2 topics using `GenericRecorder`.
- Captures camera frames with `RBRSCamera` and maintains per-camera history buffers.
- Normalizes robot history using dataset stats, runs the policy, and de-normalizes actions.
- Optionally performs temporal ensembling and policy update throttling.
- Applies per-joint slew-rate limits and publishes joint/finger targets.

Timing
- Uses ROS2 rate objects and explicit drift correction to hit the target HZ.
- `max_timesteps` controls run duration; `policy_update_period` controls inference cadence.

Notes
- Camera names in runtime JSON must match `policy.params.camera_names`.
- Image observation cadence can be adjusted with `image_obs_every` or policy `image_observation_skip`.
