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
    ray.init(address="auto", namespace="inference_engine", log_to_driver=True)

    if isinstance(inference_runtime_topics_config, str):
        with open(inference_runtime_topics_config, 'r') as f:
            inference_runtime_topics_config = json.load(f)

    if ALGORITHM == 'rtc':
        from env_actor.auto.inference_algorithms.rtc.rtc_actor import RTCActor

        env_actor = RTCActor.\
            options(resources={"inference_pc": 1}, num_cpus=2, num_gpus=1).\
            remote(
                robot=robot,
                policy_yaml_path=policy_yaml_path,
                inference_runtime_params_config=inference_runtime_params_config,
                inference_runtime_topics_config=inference_runtime_topics_config,
            )
    else:
        from env_actor.auto.inference_algorithms.sequential.sequential_actor import SequentialActor
        
        env_actor = SequentialActor.\
                    options(resources={"inference_pc": 1}).\
                    remote(
                        inference_runtime_params_config=inference_runtime_params_config,
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
