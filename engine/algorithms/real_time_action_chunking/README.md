# Real-Time Action Chunking Pipeline

This pipeline decouples ROS2 control timing from model inference by running two processes that share state via `multiprocessing.shared_memory`. It is selected with `inference.py --algorithm real_time_action_chunking` (or by inference YAML).

Key files
- `real_time_action_chunking_inference.py`: parent orchestrator; allocates shared memory, spawns controller + inference worker, installs signal handlers, and cleans up.
- `modules/controller_interface.py`: ROS2 control loop that streams observations and publishes actions.
- `modules/inference_engine.py`: policy runner that generates action chunks with guided inference.
- `modules/guided_inference.py`: stateless guided action-chunk routine using VJP.

Data flow
1) Parent allocates shared buffers for robot history, camera stacks, and action chunk.
2) Controller attaches to buffers, waits for inference readiness, seeds the action chunk, and begins publishing.
3) Controller writes latest robot state and camera frames into shared memory and decrements the action counter as it consumes actions.
4) Inference worker waits until enough actions have been consumed, then reads shared buffers and produces a new action chunk.
5) Both processes coordinate through a shared counter and a condition variable.

Shared buffer shapes
- Robot history: `(num_robot_observations, state_dim)`
- Camera stack: `(num_cams, num_image_observations, 3, H, W)`
- Action chunk: `(num_queries, action_dim)`

Why two processes
- The controller can keep ROS2 timing tight while inference spends extra time on guided updates.
- Shared memory avoids copying large camera tensors across processes.

Shutdown behavior
- `make_signal_handler` sets a shared stop event and notifies the condition variable to prevent deadlocks.
- The parent process is responsible for unlinking shared memory blocks.
