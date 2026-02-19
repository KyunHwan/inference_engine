"""Entry point for openpi inference via build_policy().

Usage:
  python run_inference_openpi.py \
    --policy_yaml_path ./env_actor/policy/policies/openpi_policy/openpi_policy.yaml \
    --robot igris_b \
    --inference_runtime_params_config /path/to/config.json \
    --inference_runtime_topics_config /path/to/topics.json
"""

import os
import sys
import argparse

import ray
import torch
import json

from ctypes import c_bool
from multiprocessing import Condition, Event, RLock, Value
import numpy as np

ALGORITHM = 'rtc'

robot_obs_history_dtype = np.float32
cam_images_dtype = np.uint8
action_chunk_dtype = np.float32


def start_openpi_inference(
    policy_yaml_path,
    robot,
    inference_runtime_params_config,
    inference_runtime_topics_config,
):
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(address="auto", namespace="inference_engine")

    # Load robot-specific RuntimeParams
    if robot == "igris_b":
        from env_actor.runtime_settings_configs.igris_b.inference_runtime_params import RuntimeParams
    # elif robot == "igris_c":
    #     from env_actor.runtime_settings_configs.igris_c.inference_runtime_params import RuntimeParams
    else:
        raise ValueError(f"Unknown robot: {robot}")

    if isinstance(inference_runtime_params_config, str):
        with open(inference_runtime_params_config, 'r') as f:
            inference_runtime_params_config = json.load(f)
    runtime_params = RuntimeParams(inference_runtime_params_config)

    

    if isinstance(inference_runtime_topics_config, str):
        with open(inference_runtime_topics_config, 'r') as f:
            inference_runtime_topics_config = json.load(f)

    if ALGORITHM == 'rtc':
        from env_actor.auto.inference_algorithms.rtc.rtc_actor_openpi import RTCActorOpenpi

        env_actor = RTCActorOpenpi.\
            options(resources={"inference_pc": 1}, num_cpus=4, num_gpus=1).\
            remote(
                robot=robot,
                inference_runtime_params_config=inference_runtime_params_config,
                inference_runtime_topics_config=inference_runtime_topics_config,
                min_num_actions_executed=35,

                policy_yaml_path=policy_yaml_path,
            )
        # from env_actor.auto.inference_algorithms.rtc.control_actor import ControllerActor as RTCControllerActor
        # from env_actor.auto.inference_algorithms.rtc.inference_actor_openpi import InferenceActorOpenpi as RTCInferenceActorOpenpi
        # from env_actor.auto.inference_algorithms.rtc.data_manager.utils.utils import create_shared_ndarray


        # # Create SharedMemory blocks in parent process
        # rob_shm, _, rob_spec = create_shared_ndarray(
        #     (runtime_params.proprio_history_size, runtime_params.proprio_state_dim), robot_obs_history_dtype
        # )
        # head_cam_shm, _, head_cam_spec = create_shared_ndarray(
        #     (runtime_params.num_img_obs, 3, runtime_params.mono_img_resize_height, runtime_params.mono_img_resize_width), cam_images_dtype
        # )
        # left_cam_shm, _, left_cam_spec = create_shared_ndarray(
        #     (runtime_params.num_img_obs, 3, runtime_params.mono_img_resize_height, runtime_params.mono_img_resize_width), cam_images_dtype
        # )
        # right_cam_shm, _, right_cam_spec = create_shared_ndarray(
        #     (runtime_params.num_img_obs, 3, runtime_params.mono_img_resize_height, runtime_params.mono_img_resize_width), cam_images_dtype
        # )
        # act_shm, _, act_spec = create_shared_ndarray(
        #     (runtime_params.action_chunk_size, runtime_params.action_dim), action_chunk_dtype
        # )

        # shm_specs = {
        #     "proprio": rob_spec,
        #     "head": head_cam_spec,
        #     "left": left_cam_spec,
        #     "right": right_cam_spec,
        #     "action": act_spec
        # }

        # # Create synchronization primitives
        # lock = RLock()
        # control_iter_cond = Condition(lock)      # For num_control_iters waits
        # inference_ready_cond = Condition(lock)   # For inference_ready waits
        # stop_event = Event()
        # episode_complete_event = Event()         # For episode completion signaling
        # num_control_iters = Value('i', 0, lock=False)
        # inference_ready_flag = Value(c_bool, False, lock=False)

        # # Create inference actor (GPU-resident) with SharedMemory specs
        # inference_engine = RTCInferenceActorOpenpi.\
        #                 options(resources={"inference_pc": 1}).\
        #                 remote(
        #                     ckpt_dir=ckpt_dir,
        #                     runtime_params=runtime_params,
        #                     shm_specs=shm_specs,
        #                     lock=lock,
        #                     control_iter_cond=control_iter_cond,
        #                     inference_ready_cond=inference_ready_cond,
        #                     stop_event=stop_event,
        #                     episode_complete_event=episode_complete_event,
        #                     num_control_iters=num_control_iters,
        #                     inference_ready_flag=inference_ready_flag,
        #                 )

        # # Create controller actor (Robot I/O) with SharedMemory specs
        # controller = RTCControllerActor.\
        #                 options(resources={"inference_pc": 1}).\
        #                 remote(
        #                     runtime_params=runtime_params,
        #                     inference_runtime_topics_config=inference_runtime_topics_config,
        #                     robot=robot,
        #                     shm_specs=shm_specs,
        #                     lock=lock,
        #                     control_iter_cond=control_iter_cond,
        #                     inference_ready_cond=inference_ready_cond,
        #                     stop_event=stop_event,
        #                     episode_complete_event=episode_complete_event,
        #                     num_control_iters=num_control_iters,
        #                     inference_ready_flag=inference_ready_flag,
        #                 )

        # # Start the RTC actors
        # inference_engine.start.remote()
        # controller.start.remote()
    else:
        # Sequential inference
        # Import and create openpi sequential actor
        from env_actor.auto.inference_algorithms.sequential.sequential_actor_openpi import (
            SequentialActorOpenpi,
        )
        env_actor = SequentialActorOpenpi.\
                    options(resources={"inference_pc": 1}).\
                    remote(
                        runtime_params=runtime_params,
                        inference_runtime_topics_config=inference_runtime_topics_config,
                        robot=robot,
                        policy_yaml_path=policy_yaml_path,
                    )
    #print(ray.get(env_actor.__ray_ready__.remote()))
    env_actor.start.remote()
    

    # Block until Ray shuts down
    try:
        import signal
        signal.pause()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        ray.shutdown()
        sys.exit()


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Run openpi inference via build_policy()")
    parser.add_argument(
        "--policy_yaml_path", "-P", type=str,
        default="./env_actor/policy/policies/openpi_policy/openpi_policy.yaml",
        help="Path to policy YAML config (ckpt_dir and default_prompt are set in the component YAML)",
    )
    parser.add_argument("--robot", required=True, choices=["igris_b", "igris_c"])
    parser.add_argument(
        "--inference_runtime_params_config",
        default="./env_actor/runtime_settings_configs/igris_b/inference_runtime_params.json",
        help="Path to inference runtime params JSON config",
    )
    parser.add_argument(
        "--inference_runtime_topics_config",
        default="./env_actor/runtime_settings_configs/igris_b/inference_runtime_topics.json",
        help="Path to inference runtime topics config",
    )
    args = parser.parse_args()

    start_openpi_inference(
        policy_yaml_path=args.policy_yaml_path,
        robot=args.robot,
        inference_runtime_params_config=args.inference_runtime_params_config,
        inference_runtime_topics_config=args.inference_runtime_topics_config,
    )
