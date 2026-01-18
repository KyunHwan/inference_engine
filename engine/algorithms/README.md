# Algorithms

Inference pipelines that turn observations into action commands. The top-level selection happens in `inference.py` based on the inference YAML.

Pipelines
- `sequential`: single-process controller in `sequential/sequential_inference.py`.
- `real_time_action_chunking`: dual-process controller and inference worker in `real_time_action_chunking/`.

Shared infrastructure
- `config`: runtime JSON loader + pydantic schema for ROS2 topics and inference settings.
- `utils`: camera capture, ROS2 recorder, shared-memory helpers, and signal handling.

Choosing a pipeline
- Set `algorithm` in the inference YAML to `sequential` or `real_time_action_chunking`.
- Both pipelines expect the same runtime JSON format and policy interface.
