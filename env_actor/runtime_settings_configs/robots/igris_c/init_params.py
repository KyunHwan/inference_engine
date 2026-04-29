"""
IGRIS_C robot initialization parameters and constants.

All motor indices are 0-based (matches the igris_c_sdk wire format:
LowState.motor_state[i] and LowCmd.motors[i].id() use 0..30).
The user-facing spec uses 1-based labels; the mapping is i_0based = i_1based - 1.

Joint name reference (0-based):
   0  Waist_Yaw
   1  Waist_Roll
   2  Waist_Pitch
   3..8   Left leg  (Hip_Pitch_L, Hip_Roll_L, Hip_Yaw_L, Knee_Pitch_L, Ankle_Pitch_L, Ankle_Roll_L)
   9..14  Right leg (Hip_Pitch_R, Hip_Roll_R, Hip_Yaw_R, Knee_Pitch_R, Ankle_Pitch_R, Ankle_Roll_R)
   15..21 Left arm  (Shoulder_Pitch_L .. Wrist_Pitch_L)
   22..28 Right arm (Shoulder_Pitch_R .. Wrist_Pitch_R)
   29  Neck_Yaw
   30  Neck_Pitch
"""

import numpy as np

N_JOINTS = 31

HOME_POSITION = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.13, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
]
HOME_POSE_RAD = np.asarray(HOME_POSITION, dtype=np.float32)

LEFT_ARM_IDS = [15, 16, 17, 18, 19, 20, 21]
RIGHT_ARM_IDS = [22, 23, 24, 25, 26, 27, 28]
WAIST_YAW_ID = 0
ACTIVE_JOINT_IDS = LEFT_ARM_IDS + RIGHT_ARM_IDS + [WAIST_YAW_ID]
FIXED_JOINT_IDS = sorted(set(range(N_JOINTS)) - set(ACTIVE_JOINT_IDS))

HAND_LEFT_IDS = [11, 12, 13, 14, 15, 16]
HAND_RIGHT_IDS = [21, 22, 23, 24, 25, 26]
HAND_MOTOR_IDS = HAND_LEFT_IDS + HAND_RIGHT_IDS

DEFAULT_JOINT_KP = np.asarray([
    200.0, 200.0, 200.0,
    500.0, 200.0,  50.0, 500.0, 300.0, 300.0,
    500.0, 200.0,  50.0, 500.0, 300.0, 300.0,
     75.0, 200.0,  45.0,  45.0,   5.0,   5.0,   5.0,
     75.0, 200.0,  45.0,  45.0,   5.0,   5.0,   5.0,
      2.0,   5.0,
], dtype=np.float32)
DEFAULT_JOINT_KD = np.asarray([
    15.0, 15.0, 15.0,
     3.0,  0.5,  0.5,  3.0,  1.5,  1.5,
     3.0,  0.5,  0.5,  3.0,  1.5,  1.5,
     0.75, 2.0, 0.225, 0.225, 0.1, 0.1, 0.1,
     0.75, 2.0, 0.225, 0.225, 0.1, 0.1, 0.1,
     0.05, 0.1,
], dtype=np.float32)
DEFAULT_HAND_KP = 50.0
DEFAULT_HAND_KD = 2.0

INIT_HAND_LEFT = 0.0
INIT_HAND_RIGHT = 0.0
INIT_WAIST_YAW = float(HOME_POSE_RAD[WAIST_YAW_ID])

INIT_ACTION_17 = np.concatenate([
    HOME_POSE_RAD[LEFT_ARM_IDS],
    HOME_POSE_RAD[RIGHT_ARM_IDS],
    np.array([INIT_HAND_LEFT, INIT_HAND_RIGHT, INIT_WAIST_YAW], dtype=np.float32),
]).astype(np.float32)

INIT_JOINT_31 = HOME_POSE_RAD.copy()

PROPRIO_BODY_Q_DIM = N_JOINTS
PROPRIO_HAND_Q_DIM = len(HAND_MOTOR_IDS)
PROPRIO_BODY_TAU_DIM = N_JOINTS
PROPRIO_HAND_TAU_DIM = len(HAND_MOTOR_IDS)
PROPRIO_STATE_DIM = (
    PROPRIO_BODY_Q_DIM + PROPRIO_HAND_Q_DIM
    + PROPRIO_BODY_TAU_DIM + PROPRIO_HAND_TAU_DIM
)

IGRIS_C_STATE_KEYS = ["body_q_31", "hand_q_12", "body_tau_31", "hand_tau_12"]
