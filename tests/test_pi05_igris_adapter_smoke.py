"""Smoke test for Pi05IgrisVlaAdapter.

Usage:
  VLA_CKPT_DIR=/path/to/ckpt pytest tests/test_pi05_igris_adapter_smoke.py -v

Requires a real vla_ checkpoint directory containing model.safetensors + assets/.
"""

import os
import pytest
import numpy as np

CKPT_DIR = os.environ.get("VLA_CKPT_DIR")


@pytest.mark.skipif(not CKPT_DIR, reason="Set VLA_CKPT_DIR env var to run")
class TestPi05IgrisAdapter:

    @pytest.fixture(scope="class")
    def adapter(self):
        from env_actor.policy.policies.pi05_igris.pi05_igris import Pi05IgrisVlaAdapter
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return Pi05IgrisVlaAdapter(ckpt_dir=CKPT_DIR, device=device)

    @pytest.fixture
    def fake_raw_obs(self):
        """Mimics output of DataManagerInterface.get_raw_obs_arrays()."""
        return {
            "proprio": np.random.randn(1, 48).astype(np.float32),
            "head": np.random.randint(0, 255, (1, 3, 480, 640), dtype=np.uint8),
            "left": np.random.randint(0, 255, (1, 3, 480, 640), dtype=np.uint8),
            "right": np.random.randint(0, 255, (1, 3, 480, 640), dtype=np.uint8),
        }

    def test_predict_returns_ndarray(self, adapter, fake_raw_obs):
        actions = adapter.predict(fake_raw_obs)
        assert isinstance(actions, np.ndarray)

    def test_predict_shape(self, adapter, fake_raw_obs):
        actions = adapter.predict(fake_raw_obs)
        assert actions.shape == (50, 24), f"Expected (50,24), got {actions.shape}"

    def test_predict_dtype(self, adapter, fake_raw_obs):
        actions = adapter.predict(fake_raw_obs)
        assert actions.dtype == np.float32

    def test_predict_finite(self, adapter, fake_raw_obs):
        actions = adapter.predict(fake_raw_obs)
        assert np.isfinite(actions).all(), "Output contains NaN or Inf"

    def test_predict_latency(self, adapter, fake_raw_obs):
        """Ensure inference completes within 5 seconds (generous for torch.compile warmup)."""
        import time
        # Warmup (first call triggers torch.compile)
        adapter.predict(fake_raw_obs)
        # Timed call
        start = time.perf_counter()
        adapter.predict(fake_raw_obs)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Inference took {elapsed:.2f}s (expected < 5s)"
