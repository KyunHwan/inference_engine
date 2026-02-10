import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class IgrisInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    base_camera: str = "cam_head"
    left_camera: str = "cam_left"
    right_camera: str = "cam_right"

    def __call__(self, data: dict) -> dict:
        images = data["images"]
        base_image = _parse_image(images[self.base_camera])

        left_raw = images.get(self.left_camera)
        if left_raw is None:
            left_image = np.zeros_like(base_image)
            left_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
        else:
            left_image = _parse_image(left_raw)
            left_mask = np.True_

        right_raw = images.get(self.right_camera)
        if right_raw is None:
            right_image = np.zeros_like(base_image)
            right_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
        else:
            right_image = _parse_image(right_raw)
            right_mask = np.True_

        inputs = {
            "state": np.asarray(data["state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": left_mask,
                "right_wrist_0_rgb": right_mask,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class IgrisOutputs(transforms.DataTransformFn):
    action_dim: int = 24

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
