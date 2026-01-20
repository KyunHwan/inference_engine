import numpy as np
from engine.algorithms.config.config_loader import ConfigLoader

# Initial positions (deg → rad) and initial finger targets
INIT_JOINT_LIST = [+20.0,+30.0,0.0,-120.0,0.0,0.0, -20.0,-30.0,0.0,+120.0,0.0,0.0] # in right and left order
INIT_HAND_LIST = [1.0,1.0,1.0,1.0,1.0,0.5, 1.0,1.0,1.0,1.0,1.0,0.5]
INIT_JOINT = np.array(
    INIT_JOINT_LIST,
    dtype=np.float32
) * np.pi / 180.0

from dataclasses import dataclass
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory

@dataclass(frozen=True)
class ShmArraySpec:
    name: str
    shape: tuple
    dtype_str: str  # e.g., np.float32().dtype.str

def init_shared_action_chunk(num_rows: int = 40, dtype=np.float32):
    # 24-D vector: [L arm 6] + [L hand 6] + [R arm 6] + [R hand 6]
    init_vec = np.asarray(
        INIT_JOINT_LIST[6:] + INIT_JOINT_LIST[:6] + INIT_HAND_LIST[:6] + INIT_HAND_LIST[6:],
        dtype=dtype,
    )  # shape (24,)
    init_vec[:12] *= (np.pi / 180.0)
    init_vec[12:] *= 0.03

    # Repeat across rows -> (40, 24)
    action_chunk = np.tile(init_vec, (num_rows, 1))  # writable, contiguous

    return action_chunk

# used in child process
def attach_shared_ndarray(spec: ShmArraySpec, unregister: bool = True):
    shm = SharedMemory(name=spec.name)
    if unregister:
        # Let the parent manage unlinking; avoid auto-unlink from the child's resource tracker.
        try:
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
    arr = np.ndarray(spec.shape, dtype=np.dtype(spec.dtype_str), buffer=shm.buf)
    return shm, arr

def make_signal_handler(stop_event, step_cond=None):
    def _handler(signum, frame):
        # 1) tell everyone to stop
        stop_event.set()
        # 2) wake up any waiters blocked on the condition
        if step_cond is not None:
            try:
                if step_cond.acquire(blocking=False):
                    try:
                        step_cond.notify_all()
                    finally:
                        step_cond.release()
            except Exception:
                pass  # best-effort
        # Do NOT sys.exit() here; let the loop notice stop_event and exit gracefully.
    return _handler

def get_model_io_params(policy, inference_settings, camera_names, image_size):
    # --- Unpack model I/O dimensions and image cadence ---
    state_dim      = policy.state_dim
    action_dim     = policy.action_dim
    num_queries    = policy.num_queries
    num_robot_obs  = policy.num_robot_observations
    num_image_obs  = policy.num_image_observations
    image_obs_every_setting = inference_settings.get("image_obs_every", 1)
    image_obs_every = image_obs_every_setting if image_obs_every_setting != -1 else policy.image_observation_skip

    # Per-camera image history buffer: N x C x H x W (uint8)
    image_width, image_height = image_size
    image_obs_history = {
        cam: np.zeros((num_image_obs, 3, image_height, image_width), dtype=np.uint8)
        for cam in camera_names
    }
    image_frame_counter = 0
    return (state_dim,
            action_dim,
            num_queries,
            num_robot_obs,
            num_image_obs,
            image_obs_every,
            image_obs_history,
            image_frame_counter)

def get_runtime_config_params(args, algorithm: str = "real_time_action_chunking"):
    # --- Load runtime config & resolve runtime parameters ---
    config       = ConfigLoader(args["runtime_config_path"])
    task_config  = config.get_config()
    inference_settings = config.get_inference_settings(algorithm)

    robot_id = task_config["robot_id"]
    camera_names = config.get_camera_names()

    HZ_override = args.get("hz")
    HZ = task_config["HZ"] if HZ_override is None else HZ_override
    max_delta_deg = inference_settings.get("max_delta", 10.0)   # input is degrees
    checkpoint_path = args["checkpoint_path"]
    return (
        config,
        task_config,
        robot_id,
        camera_names,
        HZ,
        max_delta_deg,
        checkpoint_path,
        inference_settings,
    )

def motion_smoothing_setup(max_delta_deg):
    # --- Motion smoothing setup ---
    prev_joint = INIT_JOINT.copy()
    max_delta = np.deg2rad(max_delta_deg)  # convert degrees → radians once
    print(f"max_delta: {max_delta}")
    
    return (prev_joint, max_delta)

# This is a class to keep track of maximum number of control steps experienced during inference loop
# heapq is minheap by default
from collections import deque
class MaxDeque:
    def __init__(self, buffer_len=5):
        self.dq = deque([])
        self.buffer_len = buffer_len
    
    def add(self, delay):
        if len(self.dq) == self.buffer_len:
            self.dq.popleft()
        self.dq.append(delay)
    
    def max(self):
        return max(self.dq)
