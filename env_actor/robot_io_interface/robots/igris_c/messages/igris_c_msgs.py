"""
IGRIS-C DDS Message Types for Python
=====================================
Aligned with IDL-generated types (igris_c_msgs.idl, CycloneDDS v0.10.5).

Critical for cross-process DDS communication:
  - typename must match C++ side ("igris_c.msg.dds.XXX")
  - @annotate.final + @annotate.autoid("sequential") required
  - Exact field types (float32, uint32, int16...) must match IDL

Usage:
    from messages.igris_c_msgs import LowState, HandState, ...
"""

from dataclasses import dataclass
from enum import auto

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

# -------------------------
# Constants
# -------------------------
N_JOINTS = 31  # 고정 DOF 수


# -------------------------
# Enums (IdlEnum — DDS 호환)
# -------------------------
class KinematicMode(idl.IdlEnum, typename="igris_c.msg.dds.KinematicMode", default="MS"):
    """값 해석 공간"""
    MS = auto()
    PJS = auto()


class RelayState(idl.IdlEnum, typename="igris_c.msg.dds.RelayState", default="RELAY_OFF"):
    """전원 릴레이 상태"""
    RELAY_OFF = auto()
    RELAY_ON = auto()


class EStopState(idl.IdlEnum, typename="igris_c.msg.dds.EStopState", default="ESTOP_RELEASED"):
    """E-STOP 상태"""
    ESTOP_RELEASED = auto()
    ESTOP_PRESSED = auto()


class BmsConnState(idl.IdlEnum, typename="igris_c.msg.dds.BmsConnState", default="BMS_DISCONNECTED"):
    """BMS 연결 상태"""
    BMS_DISCONNECTED = auto()
    BMS_CONNECTED = auto()


class BmsInitState(idl.IdlEnum, typename="igris_c.msg.dds.BmsInitState", default="BMS_NOT_INITIALIZED"):
    """BMS 초기화 상태"""
    BMS_NOT_INITIALIZED = auto()
    BMS_INITIALIZED = auto()
    MOTOR_INITIALIZED = auto()
    BOTH_INITIALIZED = auto()


class BmsInitType(idl.IdlEnum, typename="igris_c.msg.dds.BmsInitType", default="BMS_INIT_NONE"):
    """BMS 초기화 / 제어 명령 타입"""
    BMS_INIT_NONE = auto()
    BMS_INIT = auto()
    MOTOR_INIT = auto()
    BMS_AND_MOTOR_INIT = auto()
    BMS_OFF = auto()


class TorqueType(idl.IdlEnum, typename="igris_c.msg.dds.TorqueType", default="TORQUE_NONE"):
    """토크 관련 명령 타입"""
    TORQUE_NONE = auto()
    TORQUE_ON = auto()
    TORQUE_OFF = auto()


class ControlMode(idl.IdlEnum, typename="igris_c.msg.dds.ControlMode", default="CONTROL_MODE_LOW_LEVEL"):
    """시스템 제어 계층 선택"""
    CONTROL_MODE_LOW_LEVEL = auto()
    CONTROL_MODE_HIGH_LEVEL = auto()


# -------------------------
# IMU State
# -------------------------
@dataclass
@annotate.final
@annotate.autoid("sequential")
class IMUState(idl.IdlStruct, typename="igris_c.msg.dds.IMUState"):
    """IMU 상태 데이터"""
    quaternion: types.array[types.float32, 4]      # [w, x, y, z]
    gyroscope: types.array[types.float32, 3]       # rad/s [x, y, z]
    accelerometer: types.array[types.float32, 3]   # m/s^2 [x, y, z]
    rpy: types.array[types.float32, 3]             # rad [roll, pitch, yaw]


# -------------------------
# Motor Command & State
# -------------------------
@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorCmd(idl.IdlStruct, typename="igris_c.msg.dds.MotorCmd"):
    """모터 명령 (HYBRID mode)"""
    id: types.uint16       # motor index
    q: types.float32       # 위치 (rad)
    dq: types.float32      # 속도 (rad/s)
    tau: types.float32     # Feedforward 토크 (Nm)
    kp: types.float32      # 임피던스 강성
    kd: types.float32      # 임피던스 감쇠


@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorState(idl.IdlStruct, typename="igris_c.msg.dds.MotorState"):
    """모터 상태
    단위는 발행하는 토픽에 따라 다름 — docs/record/dds-reference-index.md 참고:
      rt/lowstate   : q=rad,            dq=rad/s,            tau_est=Nm
      rt/handstate  : q=normalized 0~1, dq=raw count/s,      tau_est=current(mA)  ← handstate는 tau_est가 current
    """
    q: types.float32           # 위치 (lowstate: rad, handstate: normalized 0~1)
    dq: types.float32          # 속도 (lowstate: rad/s, handstate: raw Dynamixel count/s)
    tau_est: types.float32     # 추정 토크 (lowstate: Nm, handstate: current mA)
    temperature: types.int16   # 모터 온도 (C)
    status_bits: types.uint32  # 상태 비트 (fault/limit)


# -------------------------
# Joint State
# -------------------------
@dataclass
@annotate.final
@annotate.autoid("sequential")
class JointState(idl.IdlStruct, typename="igris_c.msg.dds.JointState"):
    """조인트 상태 (PJS)"""
    q: types.float32           # PJS 위치 (rad)
    dq: types.float32          # PJS 속도 (rad/s)
    tau_est: types.float32     # PJS 토크 추정 (Nm)
    status_bits: types.uint32  # 상태 비트


# -------------------------
# Low-level Command & State
# -------------------------
@dataclass
@annotate.final
@annotate.autoid("sequential")
class LowCmd(idl.IdlStruct, typename="igris_c.msg.dds.LowCmd"):
    """로우레벨 명령"""
    kinematic_mode: KinematicMode                # MS or PJS
    motors: types.array[MotorCmd, N_JOINTS]      # 고정 길이 배열 (31 DOF)


@dataclass
@annotate.final
@annotate.autoid("sequential")
class LowState(idl.IdlStruct, typename="igris_c.msg.dds.LowState"):
    """로우레벨 상태"""
    sec: types.int64                                   # Unix timestamp (seconds)
    nanosec: types.uint32                              # nanoseconds
    tick: types.uint32                                 # 1ms 틱 카운터
    imu_state: IMUState                                # 베이스 IMU
    motor_state: types.array[MotorState, N_JOINTS]     # MS 원본 상태
    joint_state: types.array[JointState, N_JOINTS]     # PJS 파생 상태


# -------------------------
# Hand Command & State
# -------------------------
@dataclass
@annotate.final
@annotate.autoid("sequential")
class HandCmd(idl.IdlStruct, typename="igris_c.msg.dds.HandCmd"):
    """손 명령"""
    motor_cmd: types.sequence[MotorCmd]   # 손 모터 명령 리스트 (가변 길이)


@dataclass
@annotate.final
@annotate.autoid("sequential")
class HandState(idl.IdlStruct, typename="igris_c.msg.dds.HandState"):
    """손 상태"""
    motor_state: types.sequence[MotorState]   # 손 모터 상태 (가변 길이)
    imu_state: IMUState                       # 손 IMU


# -------------------------
# BMS Messages
# -------------------------
@dataclass
@annotate.final
@annotate.autoid("sequential")
class BmsState(idl.IdlStruct, typename="igris_c.msg.dds.BmsState"):
    """BMS 상태 메시지"""
    tick: types.uint32             # 1ms tick counter
    body_power: RelayState         # 바디 릴레이 상태
    legs_power: RelayState         # 레그 릴레이 상태
    estop: EStopState              # E-STOP 상태
    connect: BmsConnState          # BMS 연결 상태
    battery: types.float32         # 배터리 잔량 (0.0~100.0 [%])
    bms_init_state: BmsInitState   # 초기화 상태


@dataclass
@annotate.final
@annotate.autoid("sequential")
class BmsInitCmd(idl.IdlStruct, typename="igris_c.msg.dds.BmsInitCmd"):
    """BMS 초기화 명령"""
    request_id: str           # 요청 ID (응답 매칭용)
    init: BmsInitType         # 초기화 타입


@dataclass
@annotate.final
@annotate.autoid("sequential")
class TorqueCmd(idl.IdlStruct, typename="igris_c.msg.dds.TorqueCmd"):
    """토크 제어 명령"""
    request_id: str           # 요청 ID
    torque: TorqueType        # 토크 명령 타입


# -------------------------
# Control Mode Messages
# -------------------------
@dataclass
@annotate.final
@annotate.autoid("sequential")
class ControlModeCmd(idl.IdlStruct, typename="igris_c.msg.dds.ControlModeCmd"):
    """제어 모드 전환 명령"""
    request_id: str           # 요청 ID
    mode: ControlMode         # 제어 모드


@dataclass
@annotate.final
@annotate.autoid("sequential")
class ControlModeState(idl.IdlStruct, typename="igris_c.msg.dds.ControlModeState"):
    """제어 모드 상태"""
    tick: types.uint32        # 1ms tick counter
    mode: ControlMode         # 제어 모드


# -------------------------
# Service Response
# -------------------------
@dataclass
@annotate.final
@annotate.autoid("sequential")
class ServiceResponse(idl.IdlStruct, typename="igris_c.msg.dds.ServiceResponse"):
    """서비스 요청에 대한 공통 응답"""
    request_id: str           # 요청 ID
    success: bool             # 성공 여부
    message: str              # 응답 메시지 또는 에러 설명
    error_code: types.int32   # 에러 코드 (0 = success)


# -------------------------
# Camera (igris_c_jetson IDL — image.idl 기반)
# Jetson(Domain 1)에서 publish → NUC route가 Domain 10으로 re-publish
# -------------------------
@dataclass
@annotate.final
@annotate.autoid("sequential")
class Header(idl.IdlStruct, typename="igris_c.msg.dds.Header"):
    """카메라 프레임 헤더"""
    seq: types.uint64
    sec: types.int64
    nanosec: types.uint32
    frame_id: str


@dataclass
@annotate.final
@annotate.autoid("sequential")
class CompressedMessage(idl.IdlStruct, typename="igris_c.msg.dds.CompressedMessage"):
    """카메라 JPEG 압축 프레임 (igris_c_jetson IDL → igris_c.msg.dds)"""
    header: Header
    format: str                               # "jpeg"
    image_data: types.sequence[types.uint8]   # JPEG 바이트


# 하위호환: 기존 코드에서 CameraFrame을 참조하는 곳을 위한 alias
CameraFrame = CompressedMessage


# -------------------------
# Topic Names (SDK 예제에서 확인됨)
# -------------------------
class TopicNames:
    """DDS 토픽 이름 상수 (igris_c_sdk 기준)"""

    # 로봇 상태 (lowlevel_example.cpp 참조)
    LOW_STATE = "rt/lowstate"
    LOW_CMD = "rt/lowcmd"

    # BMS / Hand / 제어 모드 상태
    BMS_STATE = "rt/bmsstate"
    HAND_CMD = "rt/handcmd"
    HAND_STATE = "rt/handstate"
    CONTROL_MODE_STATE = "rt/controlmodestate"

    # 서비스 Request/Response (NAKNAK.ipynb 참조)
    BMS_INIT_REQUEST = "rt/service/bms_init/request"
    BMS_INIT_RESPONSE = "rt/service/bms_init/response"
    TORQUE_REQUEST = "rt/service/torque/request"
    TORQUE_RESPONSE = "rt/service/torque/response"
    CONTROL_MODE_REQUEST = "rt/service/control_mode/request"
    CONTROL_MODE_RESPONSE = "rt/service/control_mode/response"

    # 카메라 (igris_c_jetson — Domain 10, NUC route 경유)
    CAMERA_D435_COLOR = "igris_c/sensor/d435_color"
    CAMERA_D435_DEPTH = "igris_c/sensor/d435_depth"  # 현재 미사용
    CAMERA_EYES_STEREO = "igris_c/sensor/eyes_stereo"
    CAMERA_LEFT_HAND = "igris_c/sensor/left_hand"
    CAMERA_RIGHT_HAND = "igris_c/sensor/right_hand"
    CAMERA_DOMAIN_ID = 1  # 카메라는 Domain 1 (Jetson 직접 수신, br0 브릿지)
