"""Real-Time Action Chunking (RTC) inference algorithm — local (no-Ray) runtime.

Mirror of `rtc/` that runs the same two-process (control + inference)
RTC pipeline on a single machine without a Ray cluster. The actor file
and the two child-process entry points are mirrored locally with the
ray import / ray.init() blocks removed; the SharedMemory data manager
modules under `rtc_local/data_manager/` are local copies of those in
`rtc/` (zero ray usage there).

Components:
- RTCLocalActor: spawn-context parent that allocates SharedMemory + sync
  primitives and starts the two child processes
- actors/control_loop.start_control: control-side child process
- actors/inference_loop.start_inference: inference-side child process
"""

from .rtc_local_actor import RTCLocalActor

__all__ = ["RTCLocalActor"]
