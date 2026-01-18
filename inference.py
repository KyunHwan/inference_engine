from multiprocessing import set_start_method
import argparse
import rclpy
import sys

from engine.config.loader import load_config
from engine.config.inference_schemas import InferenceConfig, validate_inference_config
from engine.registry.plugins import load_plugins

from engine.algorithms.sequential import sequential_inference
from engine.algorithms.real_time_action_chunking import real_time_action_chunking_inference

def run_inference(config_path: str) -> None:
    raw = load_config(config_path)
    config: InferenceConfig = validate_inference_config(raw)
    load_plugins(config.plugins)

    args = {
        "runtime_config_path": config.runtime_config_path,
        "checkpoint_path": config.checkpoint_path,
        "model": config.model.model_dump(),
        "policy": config.policy.model_dump(),
        "plugins": list(config.plugins),
        "hz": config.hz if config.hz is not None else None,
    }

    if config.algorithm == "sequential":
        sequential_inference.main(args)
    elif config.algorithm == "real_time_action_chunking":
        set_start_method("spawn", force=True)
        real_time_action_chunking_inference.main(args)
    else:
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")

if __name__ == "__main__":
    rclpy.init()
    try:
        parser = argparse.ArgumentParser(description="Parse for inference config .yaml file")
        parser.add_argument("--config", help="absolute path to the inference config .yaml file.", required=True)
        args = parser.parse_args()
        run_inference(args.config)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
