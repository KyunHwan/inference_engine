"""FiLM wrappers for the SigLIP vision tower."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer


class FiLMSiglipEncoderLayerWrapper(nn.Module):
    """Wraps a SigLIP encoder layer to apply FiLM between attention and MLP."""

    def __init__(self, layer: SiglipEncoderLayer, llm_dim: int):
        super().__init__()
        self.layer = layer
        vision_dim = layer.embed_dim
        self.scale = nn.Linear(llm_dim, vision_dim)
        self.shift = nn.Linear(llm_dim, vision_dim)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)
        target_param = layer.layer_norm1.weight
        self.scale.to(device=target_param.device, dtype=target_param.dtype)
        self.shift.to(device=target_param.device, dtype=target_param.dtype)
        self._film_cond: Optional[torch.Tensor] = None

    def set_film_cond(self, film_cond: Optional[torch.Tensor]) -> None:
        self._film_cond = film_cond

    def _normalize_film_cond(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        film_cond = self._film_cond
        if film_cond is None:
            return None
        if film_cond.ndim == 3:
            film_cond = film_cond.mean(dim=1)
        if film_cond.ndim != 2:
            raise ValueError(f"Expected film_cond to be 2D, got shape {film_cond.shape}")
        return film_cond.to(device=hidden_states.device, dtype=self.scale.weight.dtype)

    @property
    def embed_dim(self) -> int:
        return self.layer.embed_dim

    @property
    def self_attn(self) -> nn.Module:
        return self.layer.self_attn

    @property
    def layer_norm1(self) -> nn.Module:
        return self.layer.layer_norm1

    @property
    def layer_norm2(self) -> nn.Module:
        return self.layer.layer_norm2

    @property
    def mlp(self) -> nn.Module:
        return self.layer.mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> tuple[torch.FloatTensor]:
        film_cond = self._normalize_film_cond(hidden_states)
        if film_cond is None:
            return self.layer(hidden_states, attention_mask, output_attentions=output_attentions)

        residual = hidden_states
        hidden_states = self.layer.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        gamma = self.scale(film_cond)[:, None, :]
        beta = self.shift(film_cond)[:, None, :]
        hidden_states = hidden_states * (1 + gamma) + beta

        residual = hidden_states
        hidden_states = self.layer.layer_norm2(hidden_states)
        hidden_states = self.layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class FiLMSiglipVisionWrapper(nn.Module):
    """Wraps SigLIP vision tower to inject FiLM without editing transformers code."""

    def __init__(self, vision_tower: nn.Module, llm_dim: int):
        super().__init__()
        self.vision_tower = vision_tower
        self.llm_dim = llm_dim
        self.config = getattr(vision_tower, "config", None)
        self._wrap_layers()

    def _get_encoder(self) -> nn.Module:
        if hasattr(self.vision_tower, "vision_model"):
            return self.vision_tower.vision_model.encoder
        return self.vision_tower.encoder

    def _wrap_layers(self) -> None:
        encoder = self._get_encoder()
        if getattr(encoder, "_film_wrapped", False):
            return
        wrapped_layers = []
        for layer in encoder.layers:
            if isinstance(layer, FiLMSiglipEncoderLayerWrapper):
                wrapped_layers.append(layer)
                continue
            if not isinstance(layer, SiglipEncoderLayer):
                raise TypeError(f"Expected SiglipEncoderLayer, got {type(layer)}")
            wrapped_layers.append(FiLMSiglipEncoderLayerWrapper(layer, self.llm_dim))
        encoder.layers = nn.ModuleList(wrapped_layers)
        encoder._film_wrapped = True

    def set_film_cond(self, film_cond: Optional[torch.Tensor]) -> None:
        encoder = self._get_encoder()
        for layer in encoder.layers:
            if hasattr(layer, "set_film_cond"):
                layer.set_film_cond(film_cond)

    def forward(self, pixel_values, **kwargs):
        return self.vision_tower(pixel_values, **kwargs)


def wrap_siglip_vision_tower(vision_tower: nn.Module, llm_dim: int) -> FiLMSiglipVisionWrapper:
    if isinstance(vision_tower, FiLMSiglipVisionWrapper):
        return vision_tower
    return FiLMSiglipVisionWrapper(vision_tower, llm_dim)
