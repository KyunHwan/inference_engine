import numpy as np


class DataNormalizationBridge:
    """
    Normalization bridge for igris_c proprio (86-D) and action (17-D).

    Assumes the trainer wrote a single 86-length `observation.state` mean/std
    covering [body_q (31), hand_q (12), body_tau (31), hand_tau (12)] in that
    order, plus an `action` mean/std of length 17. If the pkl uses split keys,
    update normalize_state to concatenate them in the same proprio order before
    slicing.
    """

    def __init__(self, norm_stats):
        self.norm_stats = norm_stats

    def normalize_state(self, state: dict[str, np.ndarray]):
        state_mean = self.norm_stats['observation.state']['mean']
        state_std = self.norm_stats['observation.state']['std']

        eps = 1e-8

        proprio_len = state['proprio'].shape[-1]
        state['proprio'] = (state['proprio'] - state_mean[:proprio_len]) / (state_std[:proprio_len] + eps)

        for key in state.keys():
            if key != 'proprio':
                state[key] = state[key] / 255.0

        return state

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        action_mean = self.norm_stats['action']['mean']
        action_std = self.norm_stats['action']['std']

        eps = 1e-8

        return (action - action_mean) / (action_std + eps)

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        action_mean = self.norm_stats['action']['mean']
        action_std = self.norm_stats['action']['std']

        return action * action_std + action_mean
