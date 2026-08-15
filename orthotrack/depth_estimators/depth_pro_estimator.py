"""
orthotrack/depth_estimators/depth_pro_estimator.py
==================================================
Apple Depth Pro monocular depth estimator.

Depth Pro predicts metric depth and focal length from a single image without
any camera metadata. It uses a multi-scale ViT architecture fine-tuned on a
large-scale metric dataset."""

import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np

from orthotrack.depth_estimators.base_depth_estimator import BaseDepthEstimator

# Path to the depth-pro thirdparty repo
_DEPTH_PRO_DIR = Path(__file__).resolve().parents[2] / 'thirdparty' / 'depth-pro'
_DEPTH_PRO_CKPT = _DEPTH_PRO_DIR / 'checkpoints' / 'depth_pro.pt'


class DepthProEstimator(BaseDepthEstimator):
    """Apple Depth Pro metric depth estimator.

    Requires: pip install -e thirdparty/depth-pro  +  depth_pro.pt checkpoint."""

    name: str = "depth_pro"
    is_metric: bool = True
    input_resolution: Optional[int] = 1536  # native Depth Pro resolution

    def __init__(self, device: str = 'cuda'):
        if str(_DEPTH_PRO_DIR / 'src') not in sys.path:
            sys.path.insert(0, str(_DEPTH_PRO_DIR / 'src'))

        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

        if not _DEPTH_PRO_CKPT.exists():
            raise FileNotFoundError(
                f"Depth Pro checkpoint not found at {_DEPTH_PRO_CKPT}.\n"
                f"Run: cd thirdparty/depth-pro && source get_pretrained_models.sh"
            )

        import dataclasses
        config = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT,
                                     checkpoint_uri=str(_DEPTH_PRO_CKPT))

        print(f"Loading Depth Pro from {_DEPTH_PRO_CKPT} ...")
        self.model, self.transform = depth_pro.create_model_and_transforms(
            config=config,
            device=device,
        )
        self.model.eval()
        self.device = device

    def estimate(
        self,
        image: np.ndarray,
        intrinsics: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Args:
            image:      (H, W, 3) uint8 RGB image.
            intrinsics: Not used by Depth Pro (it infers focal length internally).

        Returns:
            depth: (H, W) float32 metric depth in metres."""
        import torch
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image)
        img_tensor = self.transform(pil_img)

        with torch.no_grad():
            prediction = self.model.infer(img_tensor, f_px=None)

        depth = prediction['depth'].squeeze().cpu().numpy().astype(np.float32)
        return depth
