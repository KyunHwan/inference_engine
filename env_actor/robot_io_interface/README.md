# robot_io_interface

**Parent:** [env_actor](../README.md)

Hardware abstraction layer for robot communication. A factory-based interface lets inference algorithms read robot state and publish actions without knowing the underlying hardware.

## Table of contents

- [Structure](#structure)
- [ControllerInterface](#controllerinterface)
- [Required bridge methods](#required-bridge-methods)
- [IGRIS_B implementation](#igris_b-implementation)
- [IGRIS_C](#igris_c)
- [Extension points](#extension-points)
- [Related docs](#related-docs)

## Structure

```
robot_io_interface/
├── controller_interface.py      # Factory: robot name → controller bridge
└── robots/
    ├── igris_b/
    │   ├── controller_bridge.py # Full ROS2 + V4L2 implementation
    │   ├── utils/
    │   │   ├── data_dict.py     # GenericRecorder (ROS2 subscription manager)
    │   │   └── camera_utils.py  # RBRSCamera (V4L2 wrapper)
    │   └── README.md
    └── igris_c/
        ├── controller_bridge.py # Stub (NotImplementedError)
        ├── utils/
        └── README.md            # Implementation checklist
```

## ControllerInterface

Factory in [`controller_interface.py`](controller_interface.py); picks the bridge at construction time:

```python
class ControllerInterface:
    def __init__(self, runtime_params, inference_runtime_topics_config, robot):
        if robot == "igris_b":
            from .robots.igris_b.controller_bridge import ControllerBridge
        elif robot == "igris_c":
            from .robots.igris_c.controller_bridge import ControllerBridge
        self.controller_bridge = ControllerBridge(...)
```

Every method on the interface forwards to the bridge.

## Required bridge methods

| Member | Type | Purpose |
|---|---|---|
| `DT` | `float` property | `1.0 / HZ` |
| `policy_update_period` | `int` property | Steps between policy calls (Sequential only) |
| `read_state()` | method → `dict` | Read proprio + camera images |
| `publish_action(action, prev_joint)` | method → tuple | Send slew-rate-limited action; return `(smoothed_joints, fingers)` |
| `start_state_readers()` | method | Launch camera + rclpy executor threads |
| `init_robot_position()` | method → `np.ndarray` | Move to home; return initial joint state |
| `shutdown()` | method | Clean up rclpy + executor |
| `recorder_rate_controller()` | method | Return an rclpy rate object (not currently used) |

`read_state()` return shape:

```python
{
    "proprio": np.ndarray,  # (state_dim,) float32
    "head":    np.ndarray,  # (3, H, W) uint8
    "left":    np.ndarray,  # (3, H, W) uint8
    "right":   np.ndarray,  # (3, H, W) uint8
}
```

Full method-by-method reference: [docs/api.md § ControllerInterface](../../docs/api.md#controllerinterface).

## IGRIS_B implementation

Full ROS2 + V4L2 implementation in [`robots/igris_b/`](robots/igris_b/README.md).

**Dependencies**: `rclpy`, `sensor_msgs`, `std_msgs`, `geometry_msgs`, OpenCV.

**Cameras**:

| Name | Device |
|---|---|
| `head` | `/dev/head_camera1` |
| `right` | `/dev/right_camera1` |
| `left` | `/dev/left_camera2` |

Captured at 1600×1200, internally resized to 800×600 by `RBRSCamera`, then to `mono_img_resize` (default 320×240) by the bridge.

**ROS2 topics** (for `robot_id = packy`):

- Subscribe: `/igris_b/packy/joint_states` (JointState), `/igris_b/packy/finger_state` (Float32MultiArray), `/igris_b/packy/finger_current` (Float32MultiArray).
- Publish: `/igris_b/packy/target_joints` (JointState), `/igris_b/packy/finger_target` (Float32MultiArray).

**State structure**: 24-d proprio assembled from 4 6-d topic slices (`joint_pos/left`, `joint_pos/right`, `hand_joint_pos/left`, `hand_joint_pos/right`) — see [`init_params.py`](../runtime_settings_configs/robots/igris_b/init_params.py) for the canonical key list.

**Action publishing**: 24-d actions, sliced into 4 6-d vectors, joints concatenated as `[right, left]` for the ROS message, fingers in `[right, left]` order. Slew-rate clipped to `±max_delta` rad.

## IGRIS_C

Interface stubs only — every method raises `NotImplementedError`. See [`robots/igris_c/README.md`](robots/igris_c/README.md) for the hardware spec checklist required before implementation.

## Extension points

To add a new robot:

1. Create [`robots/your_robot/controller_bridge.py`](robots/igris_b/controller_bridge.py) implementing the required interface.
2. Add an `elif robot == "your_robot":` branch in [`controller_interface.py`](controller_interface.py).
3. Create the matching configs and bridges per [docs/walkthroughs/04_add_a_new_robot.md](../../docs/walkthroughs/04_add_a_new_robot.md).

## Related docs

- [docs/api.md § ControllerInterface](../../docs/api.md#controllerinterface)
- [docs/walkthroughs/04_add_a_new_robot.md](../../docs/walkthroughs/04_add_a_new_robot.md) — full robot bring-up.
- [docs/concepts.md § ROS2 primer](../../docs/concepts.md#ros2-primer-scoped-to-this-codebase)
- [docs/troubleshooting.md § ROS2](../../docs/troubleshooting.md#ros2) and [§ Cameras](../../docs/troubleshooting.md#cameras)
