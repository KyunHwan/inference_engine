"""Tests for PI0Pytorch model with dummy data and random weights (no checkpoint).

Verifies the model can be instantiated and run end-to-end on GPU using the tiny
"dummy" gemma variant. No pretrained weights are needed.

Requirements:
  - torch (with CUDA)
  - transformers==4.53.2 (exact version)
  - transformers_replace files copied into installed transformers package:
      cp -r ./src/openpi/models_pytorch/transformers_replace/* \
        .venv/lib/python3.11/site-packages/transformers/
  - pytest

Run:
  pytest src/openpi/models_pytorch/pi0_pytorch_test.py -v
  # or
  python src/openpi/models_pytorch/pi0_pytorch_test.py
"""

from unittest.mock import patch

import pytest
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Skip the entire module if transformers_replace is not installed correctly
# ---------------------------------------------------------------------------
try:
    from transformers.models.siglip.check import (
        check_whether_transformers_replace_is_installed_correctly,
    )

    _has_transformers_replace = check_whether_transformers_replace_is_installed_correctly()
except ImportError:
    _has_transformers_replace = False

pytestmark = [
    pytest.mark.skipif(
        not _has_transformers_replace,
        reason=(
            "Requires transformers==4.53.2 with transformers_replace files copied into "
            "the installed transformers package."
        ),
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires a CUDA GPU.",
    ),
]

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyObservation:
    """Minimal observation matching the duck-typed interface expected by PI0Pytorch."""

    def __init__(self, batch_size: int, action_dim: int, max_token_len: int, device: torch.device):
        self.images = {
            "base_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device),
            "left_wrist_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device),
            "right_wrist_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device),
        }
        self.image_masks = {
            "base_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
            "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
            "right_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
        }
        self.state = torch.randn(batch_size, action_dim, device=device)
        self.tokenized_prompt = torch.randint(0, 1000, (batch_size, max_token_len), device=device)
        self.tokenized_prompt_mask = torch.ones(batch_size, max_token_len, dtype=torch.bool, device=device)
        self.token_ar_mask = None
        self.token_loss_mask = None


def _tiny_paligemma_init(self, vlm_config, action_expert_config, use_adarms=None, use_vision_film=False, precision="bfloat16"):
    """Patched __init__ for PaliGemmaWithExpertModel that creates a tiny SigLIP
    vision tower whose projection_dim matches the dummy text model's hidden_size.
    """
    if use_adarms is None:
        use_adarms = [False, False]

    nn.Module.__init__(self)

    from transformers import GemmaForCausalLM, PaliGemmaForConditionalGeneration
    from transformers.models.auto import CONFIG_MAPPING

    # -- PaliGemma (VLM) config --
    vlm_config_hf = CONFIG_MAPPING["paligemma"]()
    vlm_config_hf._vocab_size = 257152  # noqa: SLF001
    vlm_config_hf.image_token_index = 257152

    # Text config — from the gemma variant (e.g. "dummy": width=64, depth=4)
    vlm_config_hf.text_config.hidden_size = vlm_config.width
    vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
    vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
    vlm_config_hf.text_config.head_dim = vlm_config.head_dim
    vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
    vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
    vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
    vlm_config_hf.text_config.torch_dtype = "float32"
    vlm_config_hf.text_config.vocab_size = 257152
    vlm_config_hf.text_config.use_adarms = use_adarms[0]
    vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None

    # Vision config — made tiny to match the dummy text model
    vlm_config_hf.vision_config.hidden_size = vlm_config.width
    vlm_config_hf.vision_config.intermediate_size = vlm_config.width * 2
    vlm_config_hf.vision_config.num_hidden_layers = 2
    vlm_config_hf.vision_config.num_attention_heads = 4
    vlm_config_hf.vision_config.projection_dim = vlm_config.width  # must match text hidden_size
    vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
    vlm_config_hf.vision_config.torch_dtype = "float32"

    # -- Action expert (Gemma) config --
    action_expert_config_hf = CONFIG_MAPPING["gemma"](
        head_dim=action_expert_config.head_dim,
        hidden_size=action_expert_config.width,
        intermediate_size=action_expert_config.mlp_dim,
        num_attention_heads=action_expert_config.num_heads,
        num_hidden_layers=action_expert_config.depth,
        num_key_value_heads=action_expert_config.num_kv_heads,
        vocab_size=257152,
        hidden_activation="gelu_pytorch_tanh",
        torch_dtype="float32",
        use_adarms=use_adarms[1],
        adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
    )

    self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
    self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
    self.gemma_expert.model.embed_tokens = None

    # Keep everything in the requested precision
    from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel

    PaliGemmaWithExpertModel.to_bfloat16_for_selected_params(self, precision)

    self._vision_film_enabled = False
    if use_vision_film:
        self.enable_vision_film(vlm_config.width)


def _create_model(dtype="bfloat16"):
    """Instantiate a tiny PI0Pytorch model with random weights on GPU.

    Note: The dummy variant uses very small dimensions (width=64, head_dim=16)
    which can trigger cuBLAS errors with bfloat16 on some GPUs. We default to
    bfloat16 for reliability. For full-size models, bfloat16 works fine.
    """
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    config = Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype=dtype,
    )

    with (
        patch("torch.compile", side_effect=lambda fn, **kwargs: fn),
        patch.object(PaliGemmaWithExpertModel, "__init__", _tiny_paligemma_init),
    ):
        model = PI0Pytorch(config)

    model = model.to(DEVICE)
    return model, config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_forward_loss_shape():
    """Training forward pass returns per-element MSE loss with the correct shape."""
    model, config = _create_model()
    model.eval()

    batch_size = 2
    obs = DummyObservation(batch_size, config.action_dim, config.max_token_len, DEVICE)
    actions = torch.randn(batch_size, config.action_horizon, config.action_dim, device=DEVICE)

    loss = model.forward(obs, actions)

    assert loss.shape == (batch_size, config.action_horizon, config.action_dim), (
        f"Expected loss shape {(batch_size, config.action_horizon, config.action_dim)}, got {loss.shape}"
    )
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss).all(), "Loss contains non-finite values"


def test_sample_actions_shape():
    """Inference sampling returns actions with the correct shape."""
    model, config = _create_model()
    model.eval()

    batch_size = 2
    obs = DummyObservation(batch_size, config.action_dim, config.max_token_len, DEVICE)

    with torch.no_grad():
        actions = model.sample_actions(DEVICE, obs, num_steps=2)

    assert actions.shape == (batch_size, config.action_horizon, config.action_dim), (
        f"Expected actions shape {(batch_size, config.action_horizon, config.action_dim)}, got {actions.shape}"
    )
    assert actions.dtype == torch.float32
    assert torch.isfinite(actions).all(), "Sampled actions contain non-finite values"


def test_backward_pass():
    """Gradients flow through the training forward pass."""
    model, config = _create_model(dtype="float32")
    model.train()

    batch_size = 2
    obs = DummyObservation(batch_size, config.action_dim, config.max_token_len, DEVICE)
    actions = torch.randn(batch_size, config.action_horizon, config.action_dim, device=DEVICE)

    loss = model.forward(obs, actions)
    loss.mean().backward()

    # At least some parameters should have gradients
    params_with_grad = [name for name, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(params_with_grad) > 0, "No parameters received gradients"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PI0Pytorch Dummy Model Test (GPU)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("SKIP: No CUDA GPU available.")
        raise SystemExit(1)

    if not _has_transformers_replace:
        print("SKIP: transformers_replace not installed correctly.")
        print("  Install: pip install transformers==4.53.2")
        print("  Then:    cp -r ./src/openpi/models_pytorch/transformers_replace/* \\")
        print("             .venv/lib/python3.11/site-packages/transformers/")
        raise SystemExit(1)

    print(f"Device: {DEVICE} ({torch.cuda.get_device_name()})")
    print()

    for name, test_fn in [
        ("test_forward_loss_shape", test_forward_loss_shape),
        ("test_sample_actions_shape", test_sample_actions_shape),
        ("test_backward_pass", test_backward_pass),
    ]:
        print(f"Running {name}...", end=" ", flush=True)
        try:
            test_fn()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
            raise

    print()
    print("All tests passed!")
