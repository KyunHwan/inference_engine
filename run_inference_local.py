"""Local (no-Ray) entry point for openpi inference via build_policy().

Mirror of `run_inference.py` that runs the same RTC or sequential inference
pipeline on a single machine without a Ray cluster. CLI flags, defaults,
and dispatch logic are identical.

Usage:
  python run_inference_local.py \
    --policy_yaml_path ./env_actor/policy/policies/openpi_policy/openpi_policy.yaml \
    --robot igris_b \
    --inference_runtime_params_config /path/to/config.json \
    --inference_runtime_topics_config /path/to/topics.json
"""

import os
import sys
import argparse

import torch
import json

from ctypes import c_bool
from multiprocessing import Condition, Event, RLock, Value
import numpy as np

robot_obs_history_dtype = np.float32
cam_images_dtype = np.uint8
action_chunk_dtype = np.float32


def start_inference(
    policy_yaml_path,
    robot,
    inference_runtime_params_config,
    inference_runtime_topics_config,
    inference_algorithm
):
    if isinstance(inference_runtime_topics_config, str):
        with open(inference_runtime_topics_config, 'r') as f:
            inference_runtime_topics_config = json.load(f)

    if inference_algorithm == 'rtc':
        from env_actor.auto.inference_algorithms.rtc_local import RTCLocalActor

        env_actor = RTCLocalActor(
            robot=robot,
            policy_yaml_path=policy_yaml_path,
            inference_runtime_params_config=inference_runtime_params_config,
            inference_runtime_topics_config=inference_runtime_topics_config,
        )
    else:
        from env_actor.auto.inference_algorithms.sequential_local import SequentialLocalActor

        env_actor = SequentialLocalActor(
            inference_runtime_params_config=inference_runtime_params_config,
            inference_runtime_topics_config=inference_runtime_topics_config,
            robot=robot,
            policy_yaml_path=policy_yaml_path,
        )

    try:
        env_actor.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        sys.exit()


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Run openpi inference locally (no Ray) via build_policy()")
    parser.add_argument(
        "--policy_yaml_path", "-P", type=str,
        default="./env_actor/policy/policies/dsrl_openpi_policy/dsrl_openpi_policy.yaml",
        help="Path to policy YAML config (ckpt_dir and default_prompt are set in the component YAML)",
    )
    parser.add_argument("--robot", default="igris_b", choices=["igris_b", "igris_c"])
    parser.add_argument(
        "--inference_runtime_params_config",
        default="./env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_params.json",
        help="Path to inference runtime params JSON config",
    )
    parser.add_argument(
        "--inference_runtime_topics_config",
        default="./env_actor/runtime_settings_configs/robots/igris_b/inference_runtime_topics.json",
        help="Path to inference runtime topics config",
    )
    parser.add_argument("--inference_algorithm",
                        default="rtc",
                        choices=["sequential", "rtc"],
                        help="inference algorithm: 'sequential' or 'rtc' (real-time action chunking)")

    args = parser.parse_args()

    start_inference(
        policy_yaml_path=args.policy_yaml_path,
        robot=args.robot,
        inference_runtime_params_config=args.inference_runtime_params_config,
        inference_runtime_topics_config=args.inference_runtime_topics_config,
        inference_algorithm=args.inference_algorithm
    )
