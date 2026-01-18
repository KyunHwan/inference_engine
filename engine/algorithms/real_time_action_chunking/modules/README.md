# Real-Time Modules

These modules implement the controller/inference pair for action chunking.

Modules
- `controller_interface.py`: ROS2 controller.
  - Attaches to shared memory and waits for the inference-ready flag.
  - Uses `GenericRecorder` for ROS2 topics and `RBRSCamera` for images.
  - Writes robot state and camera stacks to shared memory each control tick.
  - Reads the next action from the shared chunk, applies smoothing, and publishes joint/finger targets.
- `inference_engine.py`: policy runner.
  - Loads runtime config and builds the policy via `build_policy`.
  - Reads shared buffers, normalizes inputs, and runs guided action-chunk inference.
  - De-normalizes actions and writes the refreshed chunk back to shared memory.
  - Tracks execution delay with `MaxDeque` to adapt inference timing.
- `guided_inference.py`: flow-matching friendly action-chunk refinement using vector-Jacobian products.

Synchronization
- `shared_num_control_iters` tracks how many actions have been executed since the last inference pass.
- `step_cond` and `lock` coordinate buffer reads/writes between processes.
- `shared_inference_ready_flag` ensures the controller waits for inference warm-up.
