from pathlib import Path
import pickle
import numpy as np


class RuntimeParams:
    def __init__(self, inference_runtime_config):
        self._HZ = inference_runtime_config['HZ']
        self._policy_update_period = inference_runtime_config['policy_update_period']

        self._max_delta = np.deg2rad(inference_runtime_config['max_delta_deg'])
        self._proprio_state_dim = inference_runtime_config['proprio_state_dim']
        self._proprio_history_size = inference_runtime_config['proprio_history_size']

        self._camera_names = inference_runtime_config['camera_names']
        self._num_img_obs = inference_runtime_config['num_img_obs']
        self._img_obs_every = inference_runtime_config['img_obs_every']
        self._mono_img_resize_width = inference_runtime_config['mono_image_resize']['width']
        self._mono_img_resize_height = inference_runtime_config['mono_image_resize']['height']

        self._action_dim = inference_runtime_config['action_dim']
        self._action_chunk_size = inference_runtime_config['action_chunk_size']

        self._norm_stats_file_path = inference_runtime_config['norm_stats_file_path']
        self._dds = inference_runtime_config['dds']

    @property
    def HZ(self):
        return self._HZ

    @property
    def policy_update_period(self):
        return self._policy_update_period

    @property
    def max_delta(self):
        return self._max_delta

    @property
    def proprio_state_dim(self):
        return self._proprio_state_dim

    @property
    def proprio_history_size(self):
        return self._proprio_history_size

    @property
    def camera_names(self):
        return self._camera_names

    @property
    def mono_img_resize_width(self):
        return self._mono_img_resize_width

    @property
    def mono_img_resize_height(self):
        return self._mono_img_resize_height

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def action_chunk_size(self):
        return self._action_chunk_size

    @property
    def num_img_obs(self):
        return self._num_img_obs

    @property
    def img_obs_every(self):
        return self._img_obs_every

    @property
    def dds_namespace(self):
        return self._dds['namespace']

    @property
    def dds_state_domain_id(self):
        return self._dds['state_domain_id']

    @property
    def dds_camera_domain_id(self):
        return self._dds['camera_domain_id']

    @property
    def dds_state_xml(self):
        return self._dds.get('state_dds_xml', '')

    @property
    def dds_camera_xml(self):
        return self._dds.get('camera_dds_xml', '')

    @property
    def dds_topics(self):
        return self._dds['topics']

    @property
    def init_robot_at_startup(self):
        return self._dds.get('init_robot_at_startup', False)

    @property
    def joint_kp(self):
        from env_actor.runtime_settings_configs.robots.igris_c.init_params import DEFAULT_JOINT_KP
        gains = self._dds.get('joint_gains', {})
        kp = gains.get('kp')
        return np.asarray(kp, dtype=np.float32) if kp is not None else DEFAULT_JOINT_KP.copy()

    @property
    def joint_kd(self):
        from env_actor.runtime_settings_configs.robots.igris_c.init_params import DEFAULT_JOINT_KD
        gains = self._dds.get('joint_gains', {})
        kd = gains.get('kd')
        return np.asarray(kd, dtype=np.float32) if kd is not None else DEFAULT_JOINT_KD.copy()

    @property
    def hand_kp(self):
        from env_actor.runtime_settings_configs.robots.igris_c.init_params import DEFAULT_HAND_KP
        return float(self._dds.get('joint_gains', {}).get('hand_kp', DEFAULT_HAND_KP))

    @property
    def hand_kd(self):
        from env_actor.runtime_settings_configs.robots.igris_c.init_params import DEFAULT_HAND_KD
        return float(self._dds.get('joint_gains', {}).get('hand_kd', DEFAULT_HAND_KD))

    def read_stats_file(self):
        norm_stats = None
        norm_stats_file_path = Path(self._norm_stats_file_path)
        if norm_stats_file_path.is_file():
            with norm_stats_file_path.open('rb') as file:
                norm_stats = pickle.load(file)
        else:
            print(f"File not found at: {norm_stats_file_path}")
        return norm_stats
