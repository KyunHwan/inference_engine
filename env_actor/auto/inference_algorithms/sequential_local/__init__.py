"""
Sequential inference algorithm (local, no-Ray runtime).

Mirror of `sequential/` that runs the same control loop without a Ray
cluster. Reuses `sequential/data_manager/*` and all robot interfaces
unchanged via absolute imports.
"""

from .sequential_local_actor import SequentialLocalActor

__all__ = ["SequentialLocalActor"]
