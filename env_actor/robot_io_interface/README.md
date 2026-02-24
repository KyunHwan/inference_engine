# robot_io_interface

Hardware abstraction layer for robot communication. Provides a factory-based interface so that inference algorithms can read robot state and publish actions without knowing the underlying hardware.

## Structure

```
robot_io_interface/
├── controller_interface.py      # Factory: robot name → controller bridge
└── robots/
    ├── igris_b/
    │   ├── controller_bridge.py # Full ROS2 implementation
    │   └── utils/               # Robot-specific helpers
    └── igris_c/
        ├── controller_bridge.py # Stub (NotImplementedError)
        ├── utils/
        └── README.md            # Implementation checklist
```

## ControllerInterface

The factory in `controller_interface.py` selects the robot-specific bridge at runtime:

```python
class ControllerInterface:
    def __init__(self, runtime_params, inference_runtime_topics_config, robot):
        if robot == "igris_b":
            from .robots.igris_b.controller_bridge import ControllerBridge
        elif robot == "igris_c":
            from .robots.igris_c.controller_bridge import ControllerBridge
        self.controller_bridge = ControllerBridge(...)
```

All operations delegate to the robot-specific bridge.

## Required Bridge Interface

Every controller bridge must provide:

| Member | Type | Purpose |
|---|---|---|
| `DT` | `float` | Control period in seconds (`1.0 / HZ`) |
| `policy_update_period` | `int` | Steps between policy calls |
| `read_state()` | method → `dict` | Read proprioception + camera images |
| `publish_action(action, prev_joint)` | method → `tuple` | Send action to robot, return joint/finger state |
| `start_state_readers()` | method | Launch background threads for sensor acquisition |
| `init_robot_position()` | method → `np.ndarray` | Move robot to initial pose, return initial joint state |
| `shutdown()` | method | Clean up ROS2 resources and threads |
| `recorder_rate_controller()` | method | Return a rate limiter for the control loop |

### `read_state()` Return Format

```python
{
    "proprio": np.ndarray,  # shape (state_dim,) float32
    "head":    np.ndarray,  # shape (3, H, W) uint8
    "left":    np.ndarray,  # shape (3, H, W) uint8
    "right":   np.ndarray,  # shape (3, H, W) uint8
}
```

## IGRIS_B Implementation

Full ROS2-based implementation in `robots/igris_b/controller_bridge.py`.

**Dependencies:** `rclpy`, `sensor_msgs`, `std_msgs`, `geometry_msgs`, OpenCV

**Cameras:**
- Head: `/dev/head_camera1`
- Left: `/dev/left_camera2`
- Right: `/dev/right_camera1`
- Captured at 1600x1200, resized to target resolution (default 320x240)

**ROS2 topics** (for robot `packy`):
- Subscribe: `/igris_b/packy/joint_states` (JointState), `/igris_b/packy/finger_state` (Float32MultiArray)
- Publish: `/igris_b/packy/target_joints` (JointState), `/igris_b/packy/finger_target` (Float32MultiArray)

**State structure:** 24-dimensional proprioception — 6 DOF per arm (left + right) + 6 DOF per hand (left + right)

**Action publishing:** 24-dimensional actions with slew-rate limiting (`max_delta_deg` per step)

## IGRIS_C

Interface stubs only — all methods raise `NotImplementedError`. See `robots/igris_c/README.md` for the hardware specification checklist needed before implementation.

## Adding a New Robot

1. Create `robots/your_robot/controller_bridge.py` implementing all required methods
2. Add an `elif robot == "your_robot"` branch in `controller_interface.py`
3. Create corresponding:
   - Runtime configs in `runtime_settings_configs/robots/your_robot/`
   - Data manager bridges in both `sequential/data_manager/robots/your_robot/` and `rtc/data_manager/robots/your_robot/`
   - Normalization bridge in `nom_stats_manager/robots/your_robot/`
4. Use the IGRIS_B implementation as a reference template
