"""IGRIS_C Controller Bridge — cyclonedds-python direct (no igris_c_sdk).

Bypasses igris_c_sdk because the SDK on this machine was built against a
fork of igris_c_msgs.idl (extra Header struct, kinematic_modes[5]) whose
type-identifier hashes do not match the NUC firmware. SEDP discovery
matches the topic but `type_consistency_enforcement=DISALLOW_TYPE_COERCION`
rejects the writer, so 0 user-data samples reach the reader.

record BE talks to the same NUC successfully using cyclonedds-python and
the schema in `messages.igris_c_msgs`. We mirror that exact approach here.

Action layout (action_dim=17):
    [left_arm_7, right_arm_7, left_hand_1, right_hand_1, waist_yaw]
The single hand value is broadcast to all 6 finger motors per side. Body
joints not in the action vector are held at HOME_POSE_RAD via per-motor
kp/kd. State and commands are PJS (Parallel Joint Space) to match the
recorder. tau_est is read from motor_state (MS), same as recorder.

Hand state's motor_state is positional (no `id` field on MotorState in the
NUC schema); record BE reads it positionally as `hs.motor_state[i].q`,
and we follow.
"""

import os
import threading
import time

import cv2
import numpy as np

from cyclonedds.core import Qos, Policy
from cyclonedds.domain import DomainParticipant
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.util import duration

from env_actor.robot_io_interface.robots.igris_c.messages.igris_c_msgs import (
    LowState,
    LowCmd,
    HandState,
    HandCmd,
    MotorCmd,
    KinematicMode,
    CompressedMessage,
)

from env_actor.runtime_settings_configs.robots.igris_c.init_params import (
    HOME_POSE_RAD,
    N_JOINTS,
    LEFT_ARM_IDS,
    RIGHT_ARM_IDS,
    WAIST_YAW_ID,
    HAND_LEFT_IDS,
    HAND_RIGHT_IDS,
    INIT_JOINT_31,
    PROPRIO_BODY_Q_DIM,
    PROPRIO_HAND_Q_DIM,
    PROPRIO_BODY_TAU_DIM,
    PROPRIO_HAND_TAU_DIM,
)
from env_actor.runtime_settings_configs.robots.igris_c.inference_runtime_params import RuntimeParams


N_HAND_LEFT = len(HAND_LEFT_IDS)   # 6
N_HAND_RIGHT = len(HAND_RIGHT_IDS) # 6
N_HAND = N_HAND_LEFT + N_HAND_RIGHT  # 12 — index-positional in NUC HandState


class _Latest:
    """Thread-safe latest-value cache for poll-thread DDS readers.

    `count` 와 `last_seen` 은 옵션 헬스 모니터(`enable_health_log`)가 토픽
    유입을 추적할 수 있도록 노출.
    """

    __slots__ = ("_v", "_lock", "count", "last_seen")

    def __init__(self):
        self._v = None
        self._lock = threading.Lock()
        self.count = 0
        self.last_seen = 0.0

    def set(self, v):
        now = time.time()
        with self._lock:
            self._v = v
            self.count += 1
            self.last_seen = now

    def get(self):
        with self._lock:
            return self._v


def _read_xml(path: str) -> str:
    if not path:
        return ""
    with open(path, "r") as f:
        return f.read()


class ControllerBridge:
    def __init__(self, runtime_params: RuntimeParams, inference_runtime_topics_config):
        self.runtime_params = runtime_params
        topics = runtime_params.dds_topics

        # cyclonedds-python doesn't have a per-participant config arg the way
        # the C++ SDK does — DomainParticipant reads CYCLONEDDS_URI from env.
        # Set it BEFORE constructing the participant. We use the state XML;
        # camera XML is a separate domain on the same NIC in our setup, so
        # this works for both.
        state_xml_content = _read_xml(runtime_params.dds_state_xml)
        if state_xml_content and not os.environ.get("CYCLONEDDS_URI"):
            os.environ["CYCLONEDDS_URI"] = state_xml_content

        # QoS — matches record BE dds_manager (Reliable for sensors, BestEffort
        # for cameras at KeepLast(1) since stale frames are useless).
        sensor_qos = Qos(
            Policy.Reliability.Reliable(duration(seconds=1)),
            Policy.History.KeepLast(10),
        )
        camera_qos = Qos(
            Policy.Reliability.BestEffort,
            Policy.History.KeepLast(1),
        )

        self._state_dp = DomainParticipant(runtime_params.dds_state_domain_id)
        self._camera_dp = DomainParticipant(runtime_params.dds_camera_domain_id)

        # Latest caches (poll threads write, recording loop reads)
        self._latest_low = _Latest()
        self._latest_hand = _Latest()
        self._latest_img = {
            "head": _Latest(),
            "left": _Latest(),
            "right": _Latest(),
        }

        # Poll-thread infrastructure (cyclonedds-python uses take(), no
        # on_data_available callback, so we run per-reader poll threads
        # exactly like record BE's DDSParticipantContext does).
        self._running = threading.Event()
        self._running.set()
        self._poll_threads = []

        def _spawn(reader, on_msg, name):
            def loop():
                while self._running.is_set():
                    try:
                        for sample in reader.take(N=10):
                            if sample is not None:
                                on_msg(sample)
                    except Exception as e:
                        print(f"[bridge] reader {name} error: {e}", flush=True)
                    self._running.wait(timeout=0.001)
            t = threading.Thread(target=loop, daemon=True, name=f"bridge-{name}")
            t.start()
            self._poll_threads.append(t)

        # --- lowstate (q from joint_state PJS, tau from motor_state MS) ---
        self._low_topic = Topic(self._state_dp, topics["lowstate"], LowState, qos=sensor_qos)
        self._low_reader = DataReader(self._state_dp, self._low_topic)

        def on_low(msg: LowState):
            q = np.fromiter((js.q for js in msg.joint_state),
                            dtype=np.float32, count=N_JOINTS)
            tau = np.fromiter((ms.tau_est for ms in msg.motor_state),
                              dtype=np.float32, count=N_JOINTS)
            self._latest_low.set((q, tau))

        _spawn(self._low_reader, on_low, "lowstate")

        # --- handstate (positional: NUC publishes 12 motors in fixed order) ---
        self._hand_topic = Topic(self._state_dp, topics["handstate"], HandState, qos=sensor_qos)
        self._hand_reader = DataReader(self._state_dp, self._hand_topic)

        def on_hand(msg: HandState):
            seq = msg.motor_state
            n = min(len(seq), N_HAND)
            q = np.zeros(N_HAND, dtype=np.float32)
            tau = np.zeros(N_HAND, dtype=np.float32)
            for i in range(n):
                q[i] = seq[i].q
                tau[i] = seq[i].tau_est
            self._latest_hand.set((q, tau))

        _spawn(self._hand_reader, on_hand, "handstate")

        # --- cameras ---
        self._cam_readers = {}
        for key, topic_key in (
            ("head", "head_camera"),
            ("left", "left_camera"),
            ("right", "right_camera"),
        ):
            cam_topic = Topic(self._camera_dp, topics[topic_key], CompressedMessage, qos=camera_qos)
            cam_reader = DataReader(self._camera_dp, cam_topic)
            cache = self._latest_img[key]

            def on_cam(msg, c=cache):
                c.set(bytes(msg.image_data))

            _spawn(cam_reader, on_cam, f"cam-{key}")
            self._cam_readers[key] = cam_reader

        # --- publishers ---
        self._lowcmd_topic = Topic(self._state_dp, topics["lowcmd"], LowCmd, qos=sensor_qos)
        self._lowcmd_writer = DataWriter(self._state_dp, self._lowcmd_topic)
        self._handcmd_topic = Topic(self._state_dp, topics["handcmd"], HandCmd, qos=sensor_qos)
        self._handcmd_writer = DataWriter(self._state_dp, self._handcmd_topic)

        # gains
        self._joint_kp = runtime_params.joint_kp
        self._joint_kd = runtime_params.joint_kd
        self._hand_kp = runtime_params.hand_kp
        self._hand_kd = runtime_params.hand_kd
        assert self._joint_kp.shape == (N_JOINTS,)
        assert self._joint_kd.shape == (N_JOINTS,)

        self._mono_w = runtime_params.mono_img_resize_width
        self._mono_h = runtime_params.mono_img_resize_height

        self._health_thread = None
        self._health_stop = threading.Event()

        # 환경변수로 즉시 켤 수 있게 — `BRIDGE_HEALTH_LOG=1` 또는 period(sec).
        _env = os.environ.get("BRIDGE_HEALTH_LOG", "")
        if _env and _env != "0":
            try:
                period = float(_env) if _env not in ("1", "true", "True") else 1.0
            except ValueError:
                period = 1.0
            self.enable_health_log(period_sec=period)

    def topic_health(self) -> dict:
        """Return per-topic {count, last_seen, age_sec} snapshot.

        Useful for an external watchdog or one-shot diagnostic prints. Thread-
        safe (only reads counters; no lock needed because GIL gives single-
        instruction atomicity for the int/float reads we do here).
        """
        now = time.time()
        out = {}
        for name, cache in (
            ("lowstate",  self._latest_low),
            ("handstate", self._latest_hand),
            ("head_cam",  self._latest_img["head"]),
            ("left_cam",  self._latest_img["left"]),
            ("right_cam", self._latest_img["right"]),
        ):
            ls = cache.last_seen
            out[name] = {
                "count": cache.count,
                "last_seen": ls,
                "age_sec": (now - ls) if ls > 0 else None,
            }
        return out

    def enable_health_log(self, period_sec: float = 1.0) -> None:
        """Start a daemon thread that prints per-topic Hz + staleness.

        Opt-in — main inference loop is silent by default to avoid noise. Call
        this once after construction (e.g. right before the inference loop).
        """
        if self._health_thread is not None:
            return
        period = max(0.1, float(period_sec))

        def _loop():
            prev_counts = {k: 0 for k in self.topic_health()}
            t_prev = time.time()
            while not self._health_stop.wait(period):
                t_now = time.time()
                dt = max(t_now - t_prev, 1e-6)
                snap = self.topic_health()
                parts = []
                for name, info in snap.items():
                    delta = info["count"] - prev_counts.get(name, 0)
                    hz = delta / dt
                    age = info["age_sec"]
                    age_s = "n/a" if age is None else f"{age:5.2f}s"
                    parts.append(f"{name}={hz:5.1f}Hz(age {age_s})")
                    prev_counts[name] = info["count"]
                t_prev = t_now
                print("[bridge-health] " + "  ".join(parts), flush=True)

        self._health_stop.clear()
        self._health_thread = threading.Thread(
            target=_loop, daemon=True, name="bridge-health",
        )
        self._health_thread.start()

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
        print("[Internal] Starting state reading...")
        needed = [
            ("low", self._latest_low),
            ("hand", self._latest_hand),
            ("head", self._latest_img["head"]),
            ("left", self._latest_img["left"]),
            ("right", self._latest_img["right"]),
        ]
        deadline = time.time() + 10.0
        while time.time() < deadline:
            statuses = [(name, c.get() is not None) for name, c in needed]
            print(" ".join(f"{n}={'OK' if ok else '..'}" for n, ok in statuses), flush=True)
            if all(ok for _, ok in statuses):
                print("igris_c: state and cameras streaming.")
                return
            time.sleep(0.2)
        raise RuntimeError("Timeout waiting for igris_c lowstate/handstate/cameras.")

    def init_robot_position(self) -> np.ndarray:
        print("[Internal] Initializing robot position...")
        start_position = self._publish_init_lowcmd()
        self._publish_handcmd(left_val=0.0, right_val=0.0)
        return start_position.copy()

    def read_state(self) -> dict:
        low = self._latest_low.get()
        hand = self._latest_hand.get()

        if low is not None:
            body_q, body_tau = low
        else:
            body_q = np.zeros(PROPRIO_BODY_Q_DIM, dtype=np.float32)
            body_tau = np.zeros(PROPRIO_BODY_TAU_DIM, dtype=np.float32)

        if hand is not None:
            hand_q, hand_tau = hand
        else:
            hand_q = np.zeros(PROPRIO_HAND_Q_DIM, dtype=np.float32)
            hand_tau = np.zeros(PROPRIO_HAND_TAU_DIM, dtype=np.float32)
            print("read_state hand is None !!", flush=True)

        proprio = np.concatenate([body_q, hand_q, body_tau, hand_tau], dtype=np.float32)

        out = {"proprio": proprio}
        for key in ("head", "left", "right"):
            payload = self._latest_img[key].get()
            #target_w = self._mono_w * 2 if key == "head" else self._mono_w
            if payload is None:
                out[key] = np.zeros((3, self._mono_h, self._mono_w), dtype=np.uint8)
                continue
            raw = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if frame is None:
                print(f"{key} image is None !!", flush=True)
                out[key] = np.zeros((3, self._mono_h, self._mono_w), dtype=np.uint8)
                continue
            if key == "head": 
                h, w, c = frame.shape
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                frame = frame[:, :w // 2, :]
            frame = cv2.resize(frame, (self._mono_w, self._mono_h), interpolation=cv2.INTER_AREA)
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
            np.full(N_HAND_LEFT, left_val, dtype=np.float32),
            np.full(N_HAND_RIGHT, right_val, dtype=np.float32),
        ])
        
        return smoothed31, hand12

    def _publish_init_lowcmd(self):
        # NUC schema: LowCmd has a single `kinematic_mode` (PJS) and a fixed
        # 31-element `motors` array. MotorCmd carries `id` so the NUC uses the
        # array index to dispatch — we still set id explicitly to match the
        # recorder's convention (lowcmd_bridge.py does the same).
        
        start_position = np.array([
            -0.0023252665996551514,
            -0.009463071823120117,
            -0.010103017091751099,
            -0.002355217933654785,
            0.0010220259428024292,
            0.00012607872486114502,
            0.0012354850769042969,
            -0.0018252134323120117,
            -0.0004589557647705078,
            -0.007239818572998047,
            0.022650301456451416,
            -0.013188600540161133,
            0.00041747093200683594,
            -0.0012252330780029297,
            -0.0006156563758850098,
            0.03147315979003906,
            0.402606725692749,
            0.28721415996551514,
            -1.8714139461517334,
            0.4543006420135498,
            -0.1899348497390747,
            -0.02036118507385254,
            0.019218623638153076,
            -0.30788683891296387,
            -0.3067989647388458,
            -1.7461726665496826,
            -0.4039989113807678,
            0.14620661735534668,
            0.06965076923370361,
            0.0,
            0.0
        ])
        low = self._latest_low.get()

        if low is not None:
            body_q, _ = low

        t_values = np.linspace(0, 1, 100)
    
        # Interpolate
        interpolated = np.array([
            (1 - t) * body_q + t * start_position for t in t_values
        ])

        dt = 5.0 / 100.0
        start_time = time.time()

        for k in range(100):
            target_time = start_time + k * dt

            motors = []
            for i in range(N_JOINTS):
                motors.append(MotorCmd(
                    id=i,
                    q=float(interpolated[k][i]),
                    dq=0.0,
                    tau=0.0,
                    kp=float(self._joint_kp[i]),
                    kd=float(self._joint_kd[i]),
                ))

            cmd = LowCmd(kinematic_mode=KinematicMode.PJS, motors=motors)
            self._lowcmd_writer.write(cmd)

            # sleep until the scheduled time
            now = time.time()
            sleep_time = target_time - now
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        return start_position

    def _publish_lowcmd(self, target31: np.ndarray):
        # NUC schema: LowCmd has a single `kinematic_mode` (PJS) and a fixed
        # 31-element `motors` array. MotorCmd carries `id` so the NUC uses the
        # array index to dispatch — we still set id explicitly to match the
        # recorder's convention (lowcmd_bridge.py does the same).
        motors = []
        for i in range(N_JOINTS):
            motors.append(MotorCmd(
                id=i,
                q=float(target31[i]),
                dq=0.0,
                tau=0.0,
                kp=float(self._joint_kp[i]),
                kd=float(self._joint_kd[i]),
            ))
        cmd = LowCmd(kinematic_mode=KinematicMode.PJS, motors=motors)
        self._lowcmd_writer.write(cmd)

    def _publish_handcmd(self, left_val: float, right_val: float):
        seq = []
        for mid in HAND_LEFT_IDS:
            seq.append(MotorCmd(
                id=mid, q=left_val, dq=0.0, tau=0.0,
                kp=float(self._hand_kp), kd=float(self._hand_kd),
            ))
        for mid in HAND_RIGHT_IDS:
            seq.append(MotorCmd(
                id=mid, q=right_val, dq=0.0, tau=0.0,
                kp=float(self._hand_kp), kd=float(self._hand_kd),
            ))
        cmd = HandCmd(motor_cmd=seq)
        self._handcmd_writer.write(cmd)

    def shutdown(self):
        self._health_stop.set()
        if self._health_thread is not None:
            self._health_thread.join(timeout=2.0)
            self._health_thread = None
        self._running.clear()
        for t in self._poll_threads:
            t.join(timeout=2.0)
        self._poll_threads.clear()
