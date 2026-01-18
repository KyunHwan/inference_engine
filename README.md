# inference_engine

Inference runtime for robotic manipulation policies. This repository ties together configuration-driven model construction (via `policy_constructor`), ROS2 observation I/O, and two inference pipelines:

- `sequential`: a single-process control loop that runs policy inference and publishes commands in one thread of execution.
- `real_time_action_chunking`: a dual-process pipeline that shares memory between a controller and an inference worker for tighter real-time behavior.

Key capabilities
- Strict validation of inference YAML and runtime JSON configs with actionable error messages.
- Pluggable policy builders registered via a lightweight registry and plugin modules.
- Multi-camera V4L2 capture with configurable image resizing and history stacking.
- Shared-memory action chunking with guided refinement for flow-matching style policies.
- ROS2 integration for observation subscription and action publication.

Architecture overview
- `inference.py` loads the inference YAML, validates it, imports plugins, and dispatches to the selected algorithm.
- `engine/config` owns the YAML loader and schema validation.
- `engine/algorithms/config` validates runtime JSON and extracts ROS2 topic layouts and inference settings.
- `engine/modeling` builds model components via `policy_constructor` and loads checkpoints.
- `engine/policies` wraps components into an `InferencePolicy` with normalization tensors and metadata.
- `engine/algorithms/*` executes either the sequential or real-time action-chunking control loop.

Directory guide
- `engine/algorithms`: inference pipelines and shared ROS2/camera utilities.
- `engine/config`: inference YAML loader + schema definitions.
- `engine/modeling`: model construction + checkpoint loading.
- `engine/policies`: policy interfaces, builders, normalization, registry.
- `engine/registry`: generic registry + plugin loader.
- `engine/experiment_models`: placeholders for experimental policy_constructor configs.
- `policy_constructor`: model building subsystem (read-only in this repo).

Configuration inputs
- Inference YAML (validated by `engine/config/inference_schemas.py`)
  - `algorithm`: `sequential` or `real_time_action_chunking`
  - `runtime_config_path`: JSON file for ROS2 topics and camera layout
  - `checkpoint_path`: checkpoint file or directory
  - `model`: component config paths and optional per-component checkpoints
  - `policy`: builder `type` and policy parameters
  - `plugins`: import paths that register builders
  - `hz`: optional override of runtime `HZ`
- Runtime JSON (validated by `engine/algorithms/config/runtime_schema.py`)
  - `robot_id`, `HZ`, `camera_names`, `image_resize`
  - `topics`: ROS2 topic map -> field extraction rules
  - `inference`: shared inference settings (plus sequential-only fields)

Constraints and restrictions
- Supported `algorithm` values are `sequential` and `real_time_action_chunking`.
- Inference YAML must include non-empty `runtime_config_path`, `checkpoint_path`, and `model.component_config_paths`; `component_checkpoint_paths` keys must be a subset of `component_config_paths`; `hz` (if set) must be > 0.
- Policy params must be positive integers for dims/counts; `camera_names` must be non-empty and unique; `stats_path` is required and `stats_eps` must be > 0.
- The default `component_policy` builder rejects extra `policy.params` beyond the schema and requires a single policy component (or only one component overall).
- `checkpoint_path` must exist; for multi-component builds it must be a directory containing `{component}.pt` unless `component_checkpoint_paths` is provided.
- Normalization stats must exist at `stats_path` and include `state_mean`, `state_std`, `action_mean`, `action_std` with lengths matching the configured dims.
- Runtime JSON must be an object with non-empty `robot_id`, `camera_names`, and `topics`; `HZ`, `image_resize`, and `image_obs_every` must be > 0; sequential inference requires `inference.sequential`.
- Runtime topics must use `/observation/...` keys only and cannot include `/observation/images/*`; allowed `msg_type` values are `PoseStamped`, `JointState`, `Float32MultiArray`, `Bool`, `Int32`, `Float32`, `Int64`, `Float64`, `String`; field rules must be `pose.position`, `pose.orientation`, `data`, or a `slice` with `0 <= start < end`.
- Runtime `camera_names` must exactly match `policy.params.camera_names`, or inference raises an error.
- Camera capture assumes V4L2 devices at `/dev/head_camera1`, `/dev/head_camera2`, `/dev/left_camera1`, `/dev/left_camera2`, `/dev/right_camera1`, `/dev/right_camera2`.
- The real-time action chunking pipeline expects policies that expose `encode_vision` and `body` for guided inference.

Concrete example: guided real-time action chunking pick-and-place
This example exercises the most capable pipeline: dual-process action chunking with guided refinement and multi-camera inputs.

1) Inference YAML (`configs/picknplace_rtac.yaml`)
```yaml
algorithm: real_time_action_chunking
runtime_config_path: engine/algorithms/utils/picknplace.json
checkpoint_path: /path/to/checkpoints/picknplace
model:
  component_config_paths:
    policy: policy_constructor/configs/experiments/cfg_vqvae_flow_matching.yaml
policy:
  type: component_policy
  params:
    state_dim: 62
    action_dim: 24
    num_queries: 40
    num_robot_observations: 40
    num_image_observations: 2
    image_observation_skip: 1
    camera_names: [head, left, right]
    stats_path: /path/to/stats/picknplace_stats.pkl
    stats_eps: 0.01
plugins:
  - engine.policies.plugins.component_policy
# hz: 20
```

2) Runtime JSON (`engine/algorithms/utils/picknplace.json`)
```json
{
  "robot_id": "packy",
  "HZ": 20,
  "image_resize": {
    "width": 640,
    "height": 240
  },
  "camera_names": [
    "head",
    "left",
    "right"
  ],
  "topics": {
    "left_pose": {
      "topic": "/igris_b/packy/left_pose_state",
      "msg_type": "PoseStamped",
      "fields": {
        "/observation/xpos/left": "pose.position",
        "/observation/quaternion/left": "pose.orientation"
      }
    },
    "right_pose": {
      "topic": "/igris_b/packy/right_pose_state",
      "msg_type": "PoseStamped",
      "fields": {
        "/observation/xpos/right": "pose.position",
        "/observation/quaternion/right": "pose.orientation"
      }
    },
    "finger": {
      "topic": "/igris_b/packy/finger_state",
      "msg_type": "Float32MultiArray",
      "fields": {
        "/observation/hand_joint_pos/left": {
          "slice": [
            6,
            12
          ]
        },
        "/observation/hand_joint_pos/right": {
          "slice": [
            0,
            6
          ]
        }
      }
    },
    "finger_current": {
      "topic": "/igris_b/packy/finger_current",
      "msg_type": "Float32MultiArray",
      "fields": {
        "/observation/hand_joint_cur/left": {
          "slice": [
            6,
            12
          ]
        },
        "/observation/hand_joint_cur/right": {
          "slice": [
            0,
            6
          ]
        }
      }
    },
    "joints": {
      "topic": "/igris_b/packy/joint_states",
      "msg_type": "JointState",
      "fields": {
        "/observation/joint_pos/left": {
          "slice": [
            6,
            12
          ],
          "attr": "position"
        },
        "/observation/joint_pos/right": {
          "slice": [
            0,
            6
          ],
          "attr": "position"
        },
        "/observation/joint_cur/left": {
          "slice": [
            6,
            12
          ],
          "attr": "effort"
        },
        "/observation/joint_cur/right": {
          "slice": [
            0,
            6
          ],
          "attr": "effort"
        }
      }
    }
  },
  "inference": {
    "max_delta": 10.0,
    "image_obs_every": 1,
    "sequential": {
      "max_timesteps": 9000,
      "temporal_ensemble": false,
      "esb_k": 0.01,
      "policy_update_period": 1
    }
  }
}
```

3) Run
```
python inference.py --config configs/picknplace_rtac.yaml
```

What this demonstrates
- Multi-camera image histories are captured and resized in the controller process.
- The inference process performs guided action-chunk refinement and updates shared memory.
- Motion smoothing clamps per-joint deltas before publishing targets on ROS2 topics.
- The same runtime JSON can also be used for sequential inference by changing `algorithm`.

Note: The modeling layer supports multi-component policies with per-component checkpoints; adding a custom policy builder plugin lets you consume those components beyond the default `component_policy`.
