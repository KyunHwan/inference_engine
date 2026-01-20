from multiprocessing import Process, RLock, Value, set_start_method, Condition, Event
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from ctypes import c_bool
import argparse
import signal

import numpy as np

from engine.algorithms.config.config_loader import ConfigLoader
from engine.algorithms.real_time_action_chunking.modules.controller_interface import controller_interface
from engine.algorithms.real_time_action_chunking.modules.inference_engine import run_inference
from engine.algorithms.utils.inference_utils import (
    make_signal_handler,
    ShmArraySpec,
)
from engine.config.inference_schemas import PolicyConfig
from engine.registry.plugins import load_plugins

robot_obs_history_dtype = np.float32
cam_images_dtype  = np.uint8
action_chunk_dtype = np.float32

def create_shared_ndarray(shape, dtype: np.dtype, zero: bool = True):
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    shm = SharedMemory(create=True, size=nbytes)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    if zero:
        arr[...] = 0
    spec = ShmArraySpec(name=shm.name, shape=shape, dtype_str=dtype.str)
    return shm, arr, spec

def main(args):
    load_plugins(args.get("plugins", []))
    stop_event = Event()
    lock = RLock()
    step_cond = Condition(lock)
    
    signal.signal(signal.SIGINT,  make_signal_handler(stop_event, step_cond))
    signal.signal(signal.SIGTERM, make_signal_handler(stop_event, step_cond))

    policy_cfg = PolicyConfig.model_validate(args["policy"])
    runtime_config = ConfigLoader(args["runtime_config_path"])
    camera_names = runtime_config.get_camera_names()
    if set(camera_names) != set(policy_cfg.params.camera_names):
        raise ValueError("Runtime camera_names does not match policy camera_names")
    image_width, image_height = runtime_config.get_image_resize()
    rob_shape = (policy_cfg.params.num_robot_observations, policy_cfg.params.state_dim)
    cam_shape = (
        len(camera_names),
        policy_cfg.params.num_image_observations,
        3,
        image_height,
        image_width,
    )
    act_shape = (policy_cfg.params.num_queries, policy_cfg.params.action_dim)
    
    # Create shared blocks once in the parent
    rob_shm, _rob_arr, rob_spec = create_shared_ndarray(rob_shape, robot_obs_history_dtype, True)
    cam_shm, _cam_arr, cam_spec = create_shared_ndarray(cam_shape,  cam_images_dtype, True)
    act_shm, _act_arr, act_spec = create_shared_ndarray(act_shape, action_chunk_dtype, True)

    
    shared_num_control_iters = Value('i', 0, lock=False)  # RawValue would also be fine
    shared_inference_ready_flag = Value(c_bool, False, lock=False) 

    # Optional: zero quickly
    # with lock:
    #     shared_robot_state.fill(0)
    #     shared_cam_images.fill(0)
    #     shared_action_chunk.fill(0)

    # Spin up processes

    # Pass ONLY specs to children; they will attach by name
    inference_runner = Process(
        target=run_inference,
        args=(args, rob_spec, cam_spec, act_spec, shared_num_control_iters, lock, step_cond, stop_event, shared_inference_ready_flag),
        daemon=False
    )
    controller = Process(
        target=controller_interface,
        args=(args, rob_spec, cam_spec, act_spec, shared_num_control_iters, lock, step_cond, stop_event, shared_inference_ready_flag),
        daemon=False
    )
    inference_runner.start()
    controller.start()

    try:
        # Robust join loop: timeouts + exitcode monitoring
        procs = [inference_runner, controller]
        while any(p.is_alive() for p in procs):
            for p in procs:
                p.join(timeout=0.5)
                # If a child died unexpectedly, request shutdown so waiters wake up
                if p.exitcode is not None and p.exitcode != 0:
                    stop_event.set()
                    with step_cond:
                        step_cond.notify_all()
    finally:
        stop_event.set()
        try:
            with step_cond:
                step_cond.notify_all()
        except Exception:
            pass
        for p in (inference_runner, controller):
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=3)
        # Only the parent unlinks; guard against child auto-unlinks/resource_tracker
        for shm in (rob_shm, cam_shm, act_shm):
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
            try:
                resource_tracker.unregister(shm._name, "shared_memory")
            except Exception:
                pass


if __name__ == "__main__":
    # Parse args early and start ROS
    set_start_method("spawn", force=True)  # <- CUDA-safe
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config", "-I", type=str, required=True,
                        help="Path to inference YAML (validated schema)")
    from engine.config.loader import load_config
    from engine.config.inference_schemas import validate_inference_config
    args = parser.parse_args()
    raw = load_config(args.inference_config)
    cfg = validate_inference_config(raw)
    main(
        {
            "runtime_config_path": cfg.runtime_config_path,
            "checkpoint_path": cfg.checkpoint_path,
            "model": cfg.model.model_dump(),
            "policy": cfg.policy.model_dump(),
            "plugins": list(cfg.plugins),
            "hz": cfg.hz if cfg.hz is not None else None,
        }
    )
