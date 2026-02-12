"""Entry point for pi05_igris (openpi) sequential inference.

Usage:
  python run_inference_openpi.py \
    --ckpt_dir /path/to/vla_checkpoint/15000 \
    --robot igris_b \
    --inference_runtime_params_config /path/to/config.json \
    --inference_runtime_topics_config /path/to/topics.json \
    --default_prompt "pick and place"
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
    ckpt_dir,
    robot,
    inference_runtime_params_config,
    inference_runtime_topics_config,
    default_prompt=None,
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
        from env_actor.auto.inference_algorithms.rtc.control_actor import ControllerActor as RTCControllerActor
        from env_actor.auto.inference_algorithms.rtc.inference_actor_openpi import InferenceActorOpenpi as RTCInferenceActorOpenpi
        from env_actor.auto.inference_algorithms.rtc.data_manager.utils.utils import create_shared_ndarray


        # Create SharedMemory blocks in parent process
        rob_shm, _, rob_spec = create_shared_ndarray(
            (runtime_params.proprio_history_size, runtime_params.proprio_state_dim), robot_obs_history_dtype
        )
        head_cam_shm, _, head_cam_spec = create_shared_ndarray(
            (runtime_params.num_img_obs, 3, runtime_params.mono_img_resize_height, runtime_params.mono_img_resize_width), cam_images_dtype
        )
        left_cam_shm, _, left_cam_spec = create_shared_ndarray(
            (runtime_params.num_img_obs, 3, runtime_params.mono_img_resize_height, runtime_params.mono_img_resize_width), cam_images_dtype
        )
        right_cam_shm, _, right_cam_spec = create_shared_ndarray(
            (runtime_params.num_img_obs, 3, runtime_params.mono_img_resize_height, runtime_params.mono_img_resize_width), cam_images_dtype
        )
        act_shm, _, act_spec = create_shared_ndarray(
            (runtime_params.action_chunk_size, runtime_params.action_dim), action_chunk_dtype
        )

        shm_specs = {
            "proprio": rob_spec,
            "head": head_cam_spec,
            "left": left_cam_spec,
            "right": right_cam_spec,
            "action": act_spec
        }

        # Create synchronization primitives
        lock = RLock()
        control_iter_cond = Condition(lock)      # For num_control_iters waits
        inference_ready_cond = Condition(lock)   # For inference_ready waits
        stop_event = Event()
        episode_complete_event = Event()         # For episode completion signaling
        num_control_iters = Value('i', 0, lock=False)
        inference_ready_flag = Value(c_bool, False, lock=False)

        # Create inference actor (GPU-resident) with SharedMemory specs
        inference_engine = RTCInferenceActorOpenpi.\
                        options(resources={"inference_pc": 1}).\
                        remote(
                            runtime_params=runtime_params,
                            shm_specs=shm_specs,
                            lock=lock,
                            control_iter_cond=control_iter_cond,
                            inference_ready_cond=inference_ready_cond,
                            stop_event=stop_event,
                            episode_complete_event=episode_complete_event,
                            num_control_iters=num_control_iters,
                            inference_ready_flag=inference_ready_flag,
                        )

        # Create controller actor (Robot I/O) with SharedMemory specs
        controller = RTCControllerActor.\
                        options(resources={"inference_pc": 1}).\
                        remote(
                            runtime_params=runtime_params,
                            inference_runtime_topics_config=inference_runtime_topics_config,
                            robot=robot,
                            shm_specs=shm_specs,
                            lock=lock,
                            control_iter_cond=control_iter_cond,
                            inference_ready_cond=inference_ready_cond,
                            stop_event=stop_event,
                            episode_complete_event=episode_complete_event,
                            num_control_iters=num_control_iters,
                            inference_ready_flag=inference_ready_flag,
                        )

        # Start the RTC actors
        inference_engine.start.remote()
        controller.start.remote()
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
                        ckpt_dir=ckpt_dir,
                        default_prompt=default_prompt,
                    )
        print(ray.get(env_actor.__ray_ready__.remote()))
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

    parser = argparse.ArgumentParser(description="Run pi05_igris (openpi) inference")
    parser.add_argument(
        "--ckpt_dir", "-C", type=str,
        default="/home/robros/Projects/robros_vla_inference_engine/openpi_film/checkpoints/pi05_igris/pi05_igris_b_pnp_v3.3.3/cut_15000",
        help="Path to OpenPI checkpoint step directory (contains model.safetensors + assets/)",
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
    parser.add_argument(
        "--default_prompt", type=str,
        default="Pick up objects on the table and place them into the box.",
        help="Default language prompt for the policy (e.g., 'pick and place')",
    )
    args = parser.parse_args()

    start_openpi_inference(
        ckpt_dir=args.ckpt_dir,
        robot=args.robot,
        inference_runtime_params_config=args.inference_runtime_params_config,
        inference_runtime_topics_config=args.inference_runtime_topics_config,
        default_prompt=args.default_prompt,
    )
