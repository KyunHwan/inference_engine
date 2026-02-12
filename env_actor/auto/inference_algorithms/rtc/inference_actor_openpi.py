"""InferenceActor - GPU-resident Ray actor for RTC inference.

This actor runs the guided action chunk inference algorithm for real-time
action chunking (RTC) policy execution. It coordinates with SharedMemoryManager
for shared state and the PolicyStateManager for online weight updates.

Key responsibilities:
- Load and manage the policy model on GPU
- Run guided action chunk inference with flow-matching denoising
- Coordinate state reads/writes via SharedMemoryManager (direct SharedMemory access)
- Poll PolicyStateManager for weight updates (online RL)
"""
from __future__ import annotations

import ray
from multiprocessing.synchronize import Condition as ConditionType
from multiprocessing.synchronize import Event as EventType
from multiprocessing.synchronize import RLock as RLockType
from typing import Any

import numpy as np
import math

from .data_manager.utils.utils import ShmArraySpec

def _compute_guided_prefix_weights(
    delay_steps: int,
    executed_steps: int,
    total: int,
    *,
    schedule: str = "exp",
) -> np.ndarray:
    """Guided-inference prefix weighting for blending neighboring action chunks."""
    start = max(min(int(delay_steps), total), 0)
    if start >= total:
        return np.ones(total, dtype=np.float32)
    span = max(int(executed_steps), 1)
    #span = min(span, max(total - start, 1))
    
    indices = np.arange(total, dtype=np.float32)
    if schedule == "ones":
        return np.ones(total, dtype=np.float32)
    if schedule == "zeros":
        return (indices < start).astype(np.float32)
    weights = np.zeros(total, dtype=np.float32)
    weights[:start] = 1.0
    denom = total - span - start + 1
    if denom > 0 and (total - span) > start:
        c_i = (total - span - indices) / float(denom)
        inter_vals = c_i * np.expm1(c_i) / (math.e - 1.0)
        weights[start : total - span] = inter_vals[start : total - span]
    weights[total - span :] = 0.0
    return weights

@ray.remote(num_gpus=1)
class InferenceActorOpenpi:
    """GPU-resident inference actor for RTC algorithm.

    Runs the guided action chunk inference loop, coordinating with:
    - SharedMemoryManager: Direct SharedMemory access for observation/action state
    - PolicyStateManager: Polls for policy weight updates

    The inference loop waits until a minimum number of control steps have
    been executed (default: 15), then runs guided inference to produce
    a new action chunk that guides the policy toward temporal consistency.

    Args:
        runtime_params: Runtime parameters for inference
        policy_yaml_path: Path to policy YAML configuration
        policy_state_manager_handle: Ray handle to PolicyStateManagerActor
        shm_specs: Dict of ShmArraySpec for SharedMemory blocks
        lock: Shared RLock for atomic operations
        control_iter_cond: Shared Condition for control iteration waits
        inference_ready_cond: Shared Condition for inference ready waits
        stop_event: Shared Event for shutdown signaling
        num_control_iters: Shared Value for control iteration counter
        inference_ready_flag: Shared Value for inference ready signal
    """

    
    def __init__(
        self,
        runtime_params,
        shm_specs: dict[str: ShmArraySpec],
        lock: RLockType,
        control_iter_cond: ConditionType,
        inference_ready_cond: ConditionType,
        stop_event: EventType,
        episode_complete_event: EventType,
        num_control_iters: Any,  # multiprocessing.Value
        inference_ready_flag: Any,  # multiprocessing.Value
        ckpt_dir,
        default_prompt=None,
    ):
        import torch
        from .data_manager.data_normalization_interface import DataNormalizationInterface
        from .data_manager.shm_manager_interface import SharedMemoryInterface
        from env_actor.policy.policies.pi05_igris.pi05_igris import Pi05IgrisVlaAdapter

        self.runtime_params = runtime_params

        """Initialize the inference actor."""
        # Set up device and CUDA optimizations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        # Build policy using env_actor loader
        # Create pi05_igris via vla_'s factory (NOT build_policy)
        self.policy = Pi05IgrisVlaAdapter(
            ckpt_dir=ckpt_dir,
            device=str(self.device),
            default_prompt=default_prompt,
        )

        self.policy.eval()

        # Create SharedMemoryManager from specs (attaches to existing SharedMemory)
        self.shm_manager = SharedMemoryInterface.attach_from_specs(
            shm_specs=shm_specs,
            lock=lock,
            control_iter_cond=control_iter_cond,
            inference_ready_cond=inference_ready_cond,
            stop_event=stop_event,
            episode_complete_event=episode_complete_event,
            num_control_iters=num_control_iters,
            inference_ready_flag=inference_ready_flag,
        )

        self.data_normalization_bridge = DataNormalizationInterface(self.runtime_params.read_stats_file())

        # Control parameters
        self.min_num_actions_executed = 30
        
    def start(self) -> None:
        """Main inference loop - runs until explicitly stopped.

        This method runs the RTC inference loop:
        1. Wait for minimum control iterations (15)
        2. Read state atomically from SharedMemory
        3. Shift unexecuted actions to front of prev_action_chunk
        4. Normalize observations and prev_action_chunk
        5. Run guided action chunk inference
        6. Denormalize and write new action chunk
        7. Check for policy weight updates
        """
        try:
            self._async_start()
        finally:
            # Cleanup SharedMemory on exit
            self.shm_manager.cleanup()

    def _async_start(self) -> None:
        import torch
        """Main inference loop implementation.

        Note: Changed from async to sync since SharedMemoryManager uses
        standard multiprocessing primitives, not asyncio.

        Structure:
        - Warmup is called once (outside all loops)
        - Outer loop handles per-episode transitions
        - Inner loop handles inference iterations within an episode
        """
        # Warm up CUDA (once, outside all loops)
        print("Warming up CUDA kernels...")
        with torch.no_grad():
            # Warmup encode_memory
            try:
                self.policy.warmup()
            except Exception as e:
                print(f"Warmup encountered error (may be expected for minimal inputs): {e}")

        print("Starting inference loop...")

        while True:  # Outer loop - per episode
            print("Signaling inference ready...")
            self.shm_manager.set_inference_ready()

            while True:  # Inner loop - inference iterations within episode
                # Wait for minimum actions executed (blocks until threshold, episode_complete, or stop)
                result = self.shm_manager.wait_for_min_actions(self.min_num_actions_executed)

                if result == 'stop':
                    print("Stop event received, exiting inference loop")
                    return  # Exit completely

                if result == 'episode_complete':
                    print("Episode complete, waiting for next episode")
                    break  # Exit inner loop, continue to outer loop

                # result == 'min_actions' - proceed with inference
                self.shm_manager.set_inference_not_ready()

                # Atomically read state from SharedMemory
                # should be torch tensor
                input_data = self.shm_manager.atomic_read_for_inference()
                print(f"Inference triggered: {input_data['num_control_iters']} actions executed")

                # Normalize observations and prev_action_chunk
                # normalized_input_data = self.data_normalization_bridge.normalize_state_action(input_data)

                pred_actions = self.policy.predict(obs=input_data, noise=None)

                # blend_steps = max(1, min(input_data['est_delay'], 
                #                          self.min_num_actions_executed - input_data['est_delay']))
                weights = _compute_guided_prefix_weights(
                    input_data['est_delay'],
                    input_data['num_control_iters'],
                    pred_actions.shape[0],
                    schedule="exp",
                ).reshape(-1, 1)
                next_actions = input_data['prev_action'] * weights + pred_actions * (1.0 - weights)
                
                # Denormalize and write new action chunk
                # denormalized_actions = self.data_normalization_bridge.denormalize_action(pred_actions)
                self.shm_manager.write_action_chunk_n_update_iter_val(
                    next_actions, input_data['num_control_iters']
                )
