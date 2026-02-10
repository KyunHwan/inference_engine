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
    ray.init(runtime_env={"working_dir": os.getcwd()}, namespace="online_rl")

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

    # Import and create openpi sequential actor
    from env_actor.auto.inference_algorithms.sequential.sequential_actor_openpi import (
        SequentialActorOpenpi,
    )

    if isinstance(inference_runtime_topics_config, str):
        with open(inference_runtime_topics_config, 'r') as f:
            inference_runtime_topics_config = json.load(f)
    env_actor = SequentialActorOpenpi.remote(
        runtime_params=runtime_params,
        inference_runtime_topics_config=inference_runtime_topics_config,
        robot=robot,
        ckpt_dir=ckpt_dir,
        default_prompt=default_prompt,
    )
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
        default="/home/robros/Projects/robros_vla_inference_engine/openpi_orig/checkpoints/pi05_igris/pi05_igrig_b_pnp_v3.3.3/15000",
        help="Path to OpenPI checkpoint step directory (contains model.safetensors + assets/)",
    )
    parser.add_argument("--robot", required=True, choices=["igris_b", "igris_c"])
    parser.add_argument(
        "--inference_runtime_params_config", required=True,
        help="Path to inference runtime params JSON config",
    )
    parser.add_argument(
        "--inference_runtime_topics_config", required=True,
        help="Path to inference runtime topics config",
    )
    parser.add_argument(
        "--default_prompt", type=str, required=True,
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
