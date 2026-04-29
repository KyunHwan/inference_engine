from typing import Any
import numpy as np
import torch

from env_actor.runtime_settings_configs.robots.igris_c.inference_runtime_params import RuntimeParams
from env_actor.runtime_settings_configs.robots.igris_c.init_params import INIT_ACTION_17


class DataManagerBridge:
    """
    Stateful data manager for IGRIS_C — handles observation history and action buffering.

    Algorithm is identical to igris_b's bridge; only the action-init shape differs
    (action_dim=17 with broadcast hand values vs. igris_b's 24-D layout).
    """

    def __init__(self, runtime_params: RuntimeParams):
        self.runtime_params = runtime_params

        self.num_robot_obs = self.runtime_params.proprio_history_size
        self.num_image_obs = self.runtime_params.num_img_obs
        self.num_queries = self.runtime_params.action_chunk_size
        self.state_dim = self.runtime_params.proprio_state_dim
        self.action_dim = self.runtime_params.action_dim

        self.camera_names = self.runtime_params.camera_names
        self.eps = 1e-8

        self.img_obs_history = None
        self.robot_proprio_history = None
        self.image_frame_counter = 0

        self.last_action_chunk = None
        self.last_policy_step = -1

    def update_state_history(self, obs_data):
        if self.runtime_params.proprio_history_size > 1:
            self.robot_proprio_history[1:] = self.robot_proprio_history[:-1]
        self.robot_proprio_history[0] = obs_data['proprio']

        for cam_name in self.camera_names:
            if self.runtime_params.img_obs_every <= 1 or \
                    (self.image_frame_counter % self.runtime_params.img_obs_every == 0):
                if self.runtime_params.num_img_obs > 1:
                    self.img_obs_history[cam_name][1:] = self.img_obs_history[cam_name][:-1]
            self.img_obs_history[cam_name][0] = obs_data[cam_name]

        self.image_frame_counter += 1

    def buffer_action_chunk(self, policy_output: torch.Tensor, current_step: int):
        self.last_action_chunk = (
            policy_output.squeeze(0).cpu().numpy()
            if policy_output.ndim == 3
            else policy_output.cpu().numpy()
        )
        self.last_policy_step = current_step

    def get_current_action(self, current_step: int) -> np.ndarray:
        if self.last_action_chunk is None:
            raise ValueError("No action chunk available. Call buffer_action_chunk first.")

        offset = current_step - self.last_policy_step
        idx = int(np.clip(offset, 0, self.last_action_chunk.shape[0] - 1))
        return self.last_action_chunk[idx]

    def init_inference_obs_state_buffer(self, init_data):
        self.image_frame_counter = 0
        self.last_policy_step = -1

        self.img_obs_history = {
            cam: np.repeat(
                init_data[cam][np.newaxis, ...],
                self.num_image_obs,
                axis=0,
            )
            for cam in self.camera_names
        }

        self.robot_proprio_history = np.repeat(
            init_data['proprio'][np.newaxis, ...],
            self.num_robot_obs,
            axis=0,
        )

    def serve_raw_obs_state(self) -> dict:
        raw_obs = {'proprio': self.robot_proprio_history.copy()}
        for cam in self.camera_names:
            raw_obs[cam] = self.img_obs_history[cam]
        return raw_obs

    def serve_init_action(self):
        return np.tile(
            INIT_ACTION_17[None, :],
            (self.runtime_params.action_chunk_size, 1),
        )
