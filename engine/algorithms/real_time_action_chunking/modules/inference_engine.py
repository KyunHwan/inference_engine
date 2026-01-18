import numpy as np

import torch
import signal
from .guided_inference import guided_action_chunk_inference
from engine.algorithms.utils.inference_utils import (
    MaxDeque,
    attach_shared_ndarray,
    get_model_io_params,
    get_runtime_config_params,
    make_signal_handler,
)
from engine.config.inference_schemas import ModelConfig, PolicyConfig
from engine.policies.loader import build_policy
from engine.registry.plugins import load_plugins


def inference_loop(policy,
                   model_io_params,
                   normalization_stats,
                   shared_robot_obs_history,
                   shared_cam_images,
                   shared_action_chunk,
                   shared_num_control_iters,
                   lock,
                   step_cond,
                   stop_event,
                   shared_inference_ready_flag):
    sm, ss, am, asd, eps = normalization_stats
    device = sm.device

    (state_dim,
     action_dim,
     num_queries,
     num_robot_obs,
     num_image_obs,
     image_obs_every,
     image_obs_history,
     image_frame_counter) = model_io_params
    
    with lock:
        robot_obs_history = np.zeros((num_robot_obs, state_dim), dtype=np.float32)
        #print(f"num_robot_obs: {num_robot_obs}")
        #robot_obs_history[:] = np.repeat(shared_robot_state.reshape(1, -1), num_robot_obs, axis=0)
        all_cam_images = np.zeros_like(shared_cam_images, dtype=np.uint8)
        prev_action_chunk = np.zeros(shared_action_chunk.shape, dtype=np.float32)

        am_rep  = am  if am.numel()  == action_dim else am.repeat((action_dim + am.numel() - 1) // am.numel())[:action_dim]
        asd_rep = asd if asd.numel() == action_dim else asd.repeat((action_dim + asd.numel() - 1) // asd.numel())[:action_dim]
        
        delayQueue = MaxDeque()
        delayQueue.add(5)
        min_num_actions_executed_from_last_infer = 15
        num_actions_executed_from_last_infer = 0
        shared_inference_ready_flag.value = True
        step_cond.notify_all()
        # Light warm-up to ensure CUDA kernels are initialized
        img_dtype = torch.float16 if device.type == "cuda" else torch.float32
        with torch.no_grad():
            _ = policy.encode_memory(
                torch.from_numpy(all_cam_images / 255.0).to(
                    device=device, dtype=img_dtype, non_blocking=True
                ).unsqueeze(0)
            )
    # =========================
    #   MAIN CONTROL LOOP
    # =========================
    while not stop_event.is_set():
        #print("inferencing...")
        with step_cond:
            step_cond.wait_for(
                lambda: stop_event.is_set() or shared_num_control_iters.value >= min_num_actions_executed_from_last_infer
            )
        
        if stop_event.is_set():
            break

        with lock:
            num_actions_executed_from_last_infer = shared_num_control_iters.value
            print(f"Number of actions exectued: {shared_num_control_iters.value}")
            np.copyto(robot_obs_history, shared_robot_obs_history, casting='no')
            np.copyto(all_cam_images, shared_cam_images, casting='no')

            prev_action_chunk.fill(0.0)
            k = max(num_queries - num_actions_executed_from_last_infer, 0)
            #if k > 0:
            np.copyto(prev_action_chunk[:k, :], shared_action_chunk[num_actions_executed_from_last_infer:, :], casting='no')
            # else:
            #     print("num_actions_executed_from_last_inference exceeded num_queries!")
            #     np.copyto(prev_action_chunk, shared_action_chunk, casting='no')
            # np.copyto(prev_action_chunk[:num_queries-num_actions_executed_from_last_infer, :], 
            #           shared_action_chunk[num_actions_executed_from_last_infer:, :], casting='no')
            est_delay = delayQueue.max()
            

        input_prev_action_chunk = torch.from_numpy(prev_action_chunk).to(
                device=device, dtype=torch.float32, non_blocking=True
            ).unsqueeze(0)
        input_prev_action_chunk = (input_prev_action_chunk - am_rep.view(1, 1, -1)) / (asd_rep.view(1, 1, -1) + eps)
        input_prev_action_chunk[0, k:, :] = 0.0
        cam_images = torch.from_numpy(all_cam_images / 255.0).to(
                device=device, dtype=img_dtype, non_blocking=True
            ).unsqueeze(0)

        # Normalize state history for model input
        rh = torch.from_numpy(robot_obs_history).to(device=device, dtype=torch.float32, non_blocking=True)
        if sm.numel() == state_dim and ss.numel() == state_dim:
            rh = (rh - sm.view(1, -1)) / (ss.view(1, -1) + eps)
        else:
            # In case stats are flattened differently, repeat to match shape
            sm_rep = sm if sm.numel() == num_robot_obs * state_dim else sm.repeat((num_robot_obs * state_dim + sm.numel() - 1) // sm.numel())[:num_robot_obs * state_dim]
            ss_rep = ss if ss.numel() == num_robot_obs * state_dim else ss.repeat((num_robot_obs * state_dim + ss.numel() - 1) // ss.numel())[:num_robot_obs * state_dim]
            rh = (rh - sm_rep.view(num_robot_obs, state_dim)) / (ss_rep.view(num_robot_obs, state_dim) + eps)
        rh = rh.reshape(1, -1)

        # Guided policy update for real-time action chunking
        with torch.no_grad():
            cond_memory, semantic_input = policy.encode_memory(rh, cam_images)

        with torch.enable_grad():
            new_actions = guided_action_chunk_inference(
                action_decoder=policy.body,
                cond_memory=cond_memory,
                discrete_semantic_input=semantic_input,
                prev_action_chunk=input_prev_action_chunk,
                delay=int(est_delay),
                executed_steps=int(num_actions_executed_from_last_infer),
                num_ode_sim_steps=getattr(policy, "num_ode_sim_steps", 1),
                num_queries=num_queries,
                action_dim=action_dim,
            )

        action_chunk = (new_actions * asd_rep.view(1, 1, -1) + am_rep.view(1, 1, -1)) # De-normalize 
        with lock:
            np.copyto(shared_action_chunk, action_chunk.detach().float().cpu().numpy().squeeze(0), casting='no') # (num_queries, action_dim)
            shared_num_control_iters.value = shared_num_control_iters.value - num_actions_executed_from_last_infer
            print(f"Number of actions exectued from last inference: {shared_num_control_iters.value}")
            delayQueue.add(shared_num_control_iters.value)
            









def run_inference(args, 
                  rob_spec, 
                  cam_spec, 
                  act_spec, 
                  shared_num_control_iters, 
                  lock, 
                  step_cond, 
                  stop_event,
                  shared_inference_ready_flag):
    
    # Install signal handlers early
    signal.signal(signal.SIGINT,  make_signal_handler(stop_event, step_cond))
    signal.signal(signal.SIGTERM, make_signal_handler(stop_event, step_cond))
    load_plugins(args.get("plugins", []))

    rob_shm, shared_robot_obs_history  = attach_shared_ndarray(rob_spec)
    cam_shm, shared_cam_images   = attach_shared_ndarray(cam_spec)
    act_shm, shared_action_chunk = attach_shared_ndarray(act_spec)
    # torch settings
    # Let cuDNN search fast kernels dynamically
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")

    try:
        # --- Load runtime config & resolve runtime parameters ---
        (config,
        _task_config,
        _robot_id,
        camera_names,
        _HZ,
        _max_delta_deg,
        checkpoint_path,
        inference_settings) = get_runtime_config_params(args)

        image_width, image_height = config.get_image_resize()

        model_cfg = ModelConfig.model_validate(args["model"])
        policy_cfg = PolicyConfig.model_validate(args["policy"])
        if set(camera_names) != set(policy_cfg.params.camera_names):
            raise ValueError("Runtime camera_names does not match policy camera_names")

        policy = build_policy(
            model_cfg,
            policy_cfg,
            checkpoint_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        policy.freeze_all_model_params() # need to do this since for vjp, need requires_grad to be True for the input noisy action
        normalization_stats = policy.normalization_tensors
        camera_names = policy.camera_names
        (state_dim,
         action_dim,
         num_queries,
         num_robot_obs,
         num_image_obs,
         image_obs_every,
         image_obs_history,
         image_frame_counter) = get_model_io_params(
            policy,
            inference_settings,
            camera_names,
            (image_width, image_height),
        )
        model_io_params = (state_dim,
                           action_dim,
                           num_queries,
                           num_robot_obs,
                           num_image_obs,
                           image_obs_every,
                           image_obs_history,
                           image_frame_counter)
        
        inference_loop(policy,
                    model_io_params,
                    normalization_stats,
                    shared_robot_obs_history,
                    shared_cam_images,
                    shared_action_chunk,
                    shared_num_control_iters,
                    lock,
                    step_cond,
                    stop_event,
                    shared_inference_ready_flag)
    finally:
        # Children: close only (no unlink)
        rob_shm.close(); cam_shm.close(); act_shm.close()
