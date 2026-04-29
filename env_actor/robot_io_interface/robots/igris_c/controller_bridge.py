"""
IGRIS_C Controller Bridge.

Talks to the robot via igris_c_sdk (DDS / Cyclone DDS). Uses TWO independent
ChannelFactory instances so that state/cmd traffic (state_domain_id, typically 0
on wired LAN) and camera traffic (camera_domain_id, typically 10 on the
NUC-bridged interface) stay in separate domains. Per the SDK README's
"Multiple Domains In One Process" pattern:

    state_factory  = igc_sdk.ChannelFactory()
    state_factory.Init(state_domain_id, namespace, state_xml)
    camera_factory = igc_sdk.ChannelFactory()
    camera_factory.Init(camera_domain_id, namespace, camera_xml)

Subscribers/publishers take an explicit factory so we never touch the
process-singleton ChannelFactory unless `init_robot_at_startup` is true (the
IgrisC_Client service helper always uses the singleton; verified in
igris_c_sdk/src/igris_c_client.cpp:18).

Action layout (action_dim=17): [left_arm_7, right_arm_7, left_hand_1, right_hand_1, waist_yaw]
The single hand value is broadcast to all 6 finger motors per side. Body joints
not in the action vector are held at HOME_POSE_RAD via per-motor kp/kd.

See /home/user/.claude/plans/analyze-inference-engine-igris-c-sdk-igr-cached-dijkstra.md
for the full design rationale.
"""

import threading
import time

import cv2
import numpy as np

import igris_c_sdk as igc_sdk

from env_actor.runtime_settings_configs.robots.igris_c.init_params import (
    HOME_POSE_RAD,
    N_JOINTS,
    LEFT_ARM_IDS,
    RIGHT_ARM_IDS,
    WAIST_YAW_ID,
    HAND_LEFT_IDS,
    HAND_RIGHT_IDS,
    HAND_MOTOR_IDS,
    INIT_JOINT_31,
    PROPRIO_BODY_Q_DIM,
    PROPRIO_HAND_Q_DIM,
    PROPRIO_BODY_TAU_DIM,
    PROPRIO_HAND_TAU_DIM,
)
from env_actor.runtime_settings_configs.robots.igris_c.inference_runtime_params import RuntimeParams


class _Latest:
    """Thread-safe latest-value cache for callback-based DDS subscribers."""

    __slots__ = ("_v", "_lock")

    def __init__(self):
        self._v = None
        self._lock = threading.Lock()

    def set(self, v):
        with self._lock:
            self._v = v

    def get(self):
        with self._lock:
            return self._v


class ControllerBridge:
    def __init__(self, runtime_params: RuntimeParams, inference_runtime_topics_config):
        self.runtime_params = runtime_params
        topics = runtime_params.dds_topics
        ns = runtime_params.dds_namespace

        self._state_factory = igc_sdk.ChannelFactory()
        self._state_factory.Init(
            runtime_params.dds_state_domain_id,
            ns,
            runtime_params.dds_state_xml or "",
        )
        self._camera_factory = igc_sdk.ChannelFactory()
        self._camera_factory.Init(
            runtime_params.dds_camera_domain_id,
            ns,
            runtime_params.dds_camera_xml or "",
        )

        if runtime_params.init_robot_at_startup:
            igc_sdk.ChannelFactory.Instance().Init(
                runtime_params.dds_state_domain_id,
                ns,
                runtime_params.dds_state_xml or "",
            )
            self._client = igc_sdk.IgrisC_Client()
            self._client.Init()
            self._client.SetTimeout(2.0)
            self._client.InitBms(igc_sdk.BmsInitType.BMS_AND_MOTOR_INIT, 5000)
            self._client.SetTorque(igc_sdk.TorqueType.TORQUE_ON, 5000)
            self._client.SetControlMode(igc_sdk.ControlMode.CONTROL_MODE_LOW_LEVEL, 5000)
            self._client.InitHand(5000)
        else:
            self._client = None

        self._latest_low = _Latest()
        self._latest_hand = _Latest()
        self._latest_img = {
            "head": _Latest(),
            "left": _Latest(),
            "right": _Latest(),
        }

        self._low_sub = igc_sdk.LowStateSubscriber(self._state_factory, topics["lowstate"])
        self._low_sub.init(lambda msg: self._latest_low.set(msg))

        self._hand_sub = igc_sdk.HandStateSubscriber(self._state_factory, topics["handstate"])
        self._hand_sub.init(lambda msg: self._latest_hand.set(msg))

        for key, topic_key in (
            ("head", "head_camera"),
            ("left", "left_camera"),
            ("right", "right_camera"),
        ):
            sub = igc_sdk.CompressedMessageSubscriber(self._camera_factory, topics[topic_key])
            cache = self._latest_img[key]
            sub.init(lambda msg, c=cache: c.set(bytes(msg.image_data())))
            setattr(self, f"_cam_sub_{key}", sub)

        self._lowcmd_pub = igc_sdk.LowCmdPublisher(self._state_factory, topics["lowcmd"])
        self._lowcmd_pub.init()
        self._handcmd_pub = igc_sdk.HandCmdPublisher(self._state_factory, topics["handcmd"])
        self._handcmd_pub.init()

        self._joint_kp = runtime_params.joint_kp
        self._joint_kd = runtime_params.joint_kd
        self._hand_kp = runtime_params.hand_kp
        self._hand_kd = runtime_params.hand_kd
        assert self._joint_kp.shape == (N_JOINTS,)
        assert self._joint_kd.shape == (N_JOINTS,)

        self._mono_w = runtime_params.mono_img_resize_width
        self._mono_h = runtime_params.mono_img_resize_height

    @property
    def DT(self):
        return 1.0 / self.runtime_params.HZ

    @property
    def policy_update_period(self):
        return self.runtime_params.policy_update_period

    def recorder_rate_controller(self):
        hz = self.runtime_params.HZ

        class _Rate:
            def __init__(self):
                self._dt = 1.0 / hz
                self._t = time.perf_counter()

            def sleep(self):
                self._t += self._dt
                d = self._t - time.perf_counter()
                if d > 0:
                    time.sleep(d)
                else:
                    self._t = time.perf_counter()

        return _Rate()

    def start_state_readers(self):
        deadline = time.time() + 10.0
        needed = [
            self._latest_low,
            self._latest_hand,
            self._latest_img["head"],
            self._latest_img["left"],
            self._latest_img["right"],
        ]
        while time.time() < deadline:
            if all(c.get() is not None for c in needed):
                print("igris_c: state and cameras streaming.")
                return
            time.sleep(0.05)
        raise RuntimeError("Timeout waiting for igris_c lowstate/handstate/cameras.")

    def init_robot_position(self) -> np.ndarray:
        self._publish_lowcmd(target31=HOME_POSE_RAD)
        self._publish_handcmd(left_val=0.0, right_val=0.0)
        return INIT_JOINT_31.copy()

    def read_state(self) -> dict:
        low = self._latest_low.get()
        hand = self._latest_hand.get()

        body_q = np.zeros(PROPRIO_BODY_Q_DIM, dtype=np.float32)
        body_tau = np.zeros(PROPRIO_BODY_TAU_DIM, dtype=np.float32)
        if low is not None:
            ms = low.motor_state()
            for i in range(PROPRIO_BODY_Q_DIM):
                body_q[i] = float(ms[i].q())
                body_tau[i] = float(ms[i].tau_est())

        hand_q = np.zeros(PROPRIO_HAND_Q_DIM, dtype=np.float32)
        hand_tau = np.zeros(PROPRIO_HAND_TAU_DIM, dtype=np.float32)
        if hand is not None:
            id_to_q = {int(m.id()): float(m.q()) for m in hand.motor_state()}
            id_to_tau = {int(m.id()): float(m.tau_est()) for m in hand.motor_state()}
            for k, mid in enumerate(HAND_MOTOR_IDS):
                hand_q[k] = id_to_q.get(mid, 0.0)
                hand_tau[k] = id_to_tau.get(mid, 0.0)

        proprio = np.concatenate([body_q, hand_q, body_tau, hand_tau], dtype=np.float32)

        out = {"proprio": proprio}
        for key in ("head", "left", "right"):
            payload = self._latest_img[key].get()
            target_w = self._mono_w * 2 if key == "head" else self._mono_w
            if payload is None:
                out[key] = np.zeros((3, self._mono_h, target_w), dtype=np.uint8)
                continue
            raw = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if frame is None:
                out[key] = np.zeros((3, self._mono_h, target_w), dtype=np.uint8)
                continue
            frame = cv2.resize(frame, (target_w, self._mono_h), interpolation=cv2.INTER_AREA)
            out[key] = np.transpose(frame, (2, 0, 1))
        return out

    def publish_action(self, action: np.ndarray, prev_joint: np.ndarray):
        a = np.asarray(action, dtype=np.float32)
        assert a.shape == (17,), f"expected (17,), got {a.shape}"

        raw31 = HOME_POSE_RAD.copy()
        raw31[LEFT_ARM_IDS] = a[0:7]
        raw31[RIGHT_ARM_IDS] = a[7:14]
        raw31[WAIST_YAW_ID] = a[16]

        max_d = self.runtime_params.max_delta
        delta = np.clip(raw31 - prev_joint, -max_d, max_d)
        smoothed31 = (prev_joint + delta).astype(np.float32)

        self._publish_lowcmd(target31=smoothed31)

        left_val = float(a[14])
        right_val = float(a[15])
        self._publish_handcmd(left_val=left_val, right_val=right_val)

        hand12 = np.concatenate([
            np.full(6, left_val, dtype=np.float32),
            np.full(6, right_val, dtype=np.float32),
        ])
        return smoothed31, hand12

    def _publish_lowcmd(self, target31: np.ndarray):
        # cmd.motors() returns a Python list of COPIES (pybind11/stl.h converts
        # std::array<MotorCmd, 31> by copy). Build the list and use the setter.
        motors = []
        for i in range(N_JOINTS):
            m = igc_sdk.MotorCmd()
            m.id(i)
            m.q(float(target31[i]))
            m.dq(0.0)
            m.tau(0.0)
            m.kp(float(self._joint_kp[i]))
            m.kd(float(self._joint_kd[i]))
            motors.append(m)

        cmd = igc_sdk.LowCmd()
        cmd.motors(motors)
        cmd.kinematic_modes([igc_sdk.KinematicMode.MS] * 5)
        self._lowcmd_pub.write(cmd)

    def _publish_handcmd(self, left_val: float, right_val: float):
        # Same copy-semantics issue as LowCmd: motor_cmd() (std::vector<MotorCmd>)
        # comes back as a Python list of copies. Build, then setter.
        seq = []
        for mid in HAND_LEFT_IDS:
            mc = igc_sdk.MotorCmd()
            mc.id(mid)
            mc.q(left_val)
            mc.dq(0.0)
            mc.tau(0.0)
            mc.kp(self._hand_kp)
            mc.kd(self._hand_kd)
            seq.append(mc)
        for mid in HAND_RIGHT_IDS:
            mc = igc_sdk.MotorCmd()
            mc.id(mid)
            mc.q(right_val)
            mc.dq(0.0)
            mc.tau(0.0)
            mc.kp(self._hand_kp)
            mc.kd(self._hand_kd)
            seq.append(mc)

        cmd = igc_sdk.HandCmd()
        cmd.motor_cmd(seq)
        self._handcmd_pub.write(cmd)

    def shutdown(self):
        for sub_attr in (
            "_low_sub",
            "_hand_sub",
            "_cam_sub_head",
            "_cam_sub_left",
            "_cam_sub_right",
        ):
            sub = getattr(self, sub_attr, None)
            if sub is not None and hasattr(sub, "stop"):
                try:
                    sub.stop()
                except Exception:
                    pass
        for pub_attr in ("_lowcmd_pub", "_handcmd_pub"):
            pub = getattr(self, pub_attr, None)
            if pub is not None and hasattr(pub, "stop"):
                try:
                    pub.stop()
                except Exception:
                    pass
        for fac_attr in ("_state_factory", "_camera_factory"):
            fac = getattr(self, fac_attr, None)
            if fac is not None and hasattr(fac, "Release"):
                try:
                    fac.Release()
                except Exception:
                    pass
