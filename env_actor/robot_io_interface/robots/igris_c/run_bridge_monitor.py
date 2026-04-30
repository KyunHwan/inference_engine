"""Standalone bridge monitor — controller_bridge_copy 만 띄워서 실시간 값 확인.

사용:
    cd /home/robros/Projects/inference_engine
    ./.venv/bin/python -m env_actor.robot_io_interface.robots.igris_c.run_bridge_monitor

Ctrl+C 로 종료. policy / inference 파이프라인 안 거치고 bridge 만 단독 실행.
"""
import json
import time

from env_actor.runtime_settings_configs.robots.igris_c.inference_runtime_params import RuntimeParams
from env_actor.robot_io_interface.robots.igris_c.controller_bridge import ControllerBridge


CFG = "/home/robros/Projects/inference_engine/env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json"


def main() -> None:
    with open(CFG) as f:
        cfg = json.load(f)
    rp = RuntimeParams(cfg)

    print("[monitor] starting bridge ...")
    cb = ControllerBridge(rp, None)
    print("[monitor] bridge up. waiting for streams ...\n")

    try:
        cb.start_state_readers()
    except RuntimeError as e:
        print(f"[monitor] start_state_readers failed: {e}")
        cb.shutdown()
        return

    try:
        while True:
            state = cb.read_state()
            proprio = state["proprio"]
            head = state["head"]
            left = state["left"]
            right = state["right"]
            body_q = proprio[0:31]
            hand_q = proprio[31:43]
            body_tau = proprio[43:74]
            hand_tau = proprio[74:86]
            print(
                f"\rbody_q[0:3]={body_q[0:3].round(3)}  "
                f"hand_q[0:3]={hand_q[0:3].round(3)}  "
                f"body_tau[0]={body_tau[0]:+.3f}  "
                f"head={head.shape}  L={left.shape}  R={right.shape}",
                end="", flush=True,
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[monitor] stopping ...")
    finally:
        cb.shutdown()


if __name__ == "__main__":
    main()
