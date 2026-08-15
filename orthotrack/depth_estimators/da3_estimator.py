"""
orthotrack/depth_estimators/da3_estimator.py
============================================
Depth Anything 3 (DA3) monocular depth estimator.

DA3 is a plain ViT-based multi-view depth and pose foundation model from
ByteDance Seed. It supports relative depth, pose estimation, and 3D Gaussians
from one or more images.

We use it in single-image mode for depth estimation only."""

import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_JAX', '0')

from typing import Optional
import numpy as np

from orthotrack.depth_estimators.base_depth_estimator import BaseDepthEstimator

# HuggingFace model IDs for DA3 variants
_HF_IDS = {
    'small':       'depth-anything/DA3-SMALL',
    'base':        'depth-anything/DA3-BASE',
    'large':       'depth-anything/DA3-LARGE-1.1',
    'giant':       'depth-anything/DA3-GIANT-1.1',
    'nested':      'depth-anything/DA3NESTED-GIANT-LARGE-1.1',
    'mono':        'depth-anything/DA3MONO-LARGE',
}


class DepthAnything3Estimator(BaseDepthEstimator):
    """Depth Anything 3 relative depth estimator (single-image mode).

    Requires: pip install -e thirdparty/depth-anything-3  (or installed package)"""

    name: str = "da3"
    is_metric: bool = False
    input_resolution: Optional[int] = None  # DA3 handles its own resizing

    def __init__(self, variant: str = 'large', device: str = 'cuda'):
        """
        Args:
            variant: one of 'small', 'base', 'large', 'giant', 'nested', 'mono'.
                     'large' (0.4B) is recommended for GPU memory constraints.
            device:  Target device."""
        import torch
        from depth_anything_3.api import DepthAnything3

        hf_id = _HF_IDS.get(variant, _HF_IDS['large'])
        print(f"Loading Depth Anything 3 ({variant}) from HuggingFace ({hf_id}) ...")
        self.model = DepthAnything3.from_pretrained(hf_id)
        self.model = self.model.to(device).eval()
        self._torch = torch
        self.device = device

    def estimate(
        self,
        image: np.ndarray,
        intrinsics: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Args:
            image:      (H, W, 3) uint8 RGB image.
            intrinsics: Not used (relative estimator).

        Returns:
            depth: (H, W) float32 relative depth, larger values = farther."""
        import tempfile, os
        from PIL import Image as PILImage
        import torch

        # Save to temp file — DA3 accepts file paths, PIL images, or numpy arrays.
        # Using PIL image to avoid disk I/O overhead.
        pil_img = PILImage.fromarray(image)

        with torch.no_grad():
            prediction = self.model.inference([pil_img])

        # prediction.depth: [N, H, W] float32, larger = farther
        depth = prediction.depth[0]  # (H, W)
        if hasattr(depth, 'cpu'):
            depth = depth.cpu().numpy()
        depth = np.asarray(depth, dtype=np.float32)
        return depth
