"""
orthotrack/depth_estimators/unidepth_estimator.py
=================================================
UniDepth v2 universal metric depth estimator (ETH Zurich).

UniDepth v2 predicts metric depth and camera intrinsics from a single image.
It supports pinhole and fisheye cameras and outputs metric depth in metres."""

from pathlib import Path
from typing import Optional

import numpy as np

from orthotrack.depth_estimators.base_depth_estimator import BaseDepthEstimator


# HuggingFace model IDs for UniDepth v2 variants
_HF_IDS = {
    'vitl14': 'lpiccinelli/unidepth-v2-vitl14',
    'vits14': 'lpiccinelli/unidepth-v2-vits14',
}


class UniDepthEstimator(BaseDepthEstimator):
    """UniDepth v2 metric depth estimator (ETH Zurich).

    Requires: pip install git+https://github.com/lpiccinelli-eth/UniDepth.git"""

    name: str = "unidepth_v2"
    is_metric: bool = True
    input_resolution: Optional[int] = 518

    def __init__(self, variant: str = 'vitl14', device: str = 'cuda'):
        """
        Args:
            variant: Model variant: 'vitl14' (default, large) or 'vits14' (small, faster).
            device:  Target device."""
        hf_id = _HF_IDS.get(variant, _HF_IDS['vitl14'])
        print(f"Loading UniDepth v2 ({variant}) from HuggingFace ({hf_id}) ...")
        try:
            from unidepth.models import UniDepthV2
            self.model = UniDepthV2.from_pretrained(hf_id)
        except ImportError:
            # Fall back to torch hub
            import torch
            self.model = torch.hub.load('lpiccinelli-eth/unidepth', 'UniDepthV2',
                                        version=variant, pretrained=True, trust_repo=True)

        import torch
        self.model = self.model.to(device).eval()
        self.device = device

    def estimate(
        self,
        image: np.ndarray,
        intrinsics: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Args:
            image:      (H, W, 3) uint8 RGB image.
            intrinsics: Optional (3, 3) camera intrinsics matrix.
                        If provided, passed to UniDepth to improve metric accuracy.

        Returns:
            depth: (H, W) float32 metric depth in metres."""
        import torch

        # UniDepth expects (1, 3, H, W) uint8 or float [0, 255]
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        img_tensor = img_tensor.float().to(self.device)

        camera = None
        if intrinsics is not None:
            try:
                from unidepth.utils.camera import Camera
                K_tensor = torch.from_numpy(intrinsics.astype(np.float32)).unsqueeze(0).to(self.device)
                camera = Camera(K=K_tensor)
            except Exception:
                pass  # If Camera construction fails, run without known intrinsics

        with torch.no_grad():
            predictions = self.model.infer(img_tensor, camera=camera)

        # predictions['depth']: (1, 1, H', W') or (1, H', W')
        depth = predictions['depth'].squeeze().cpu().numpy().astype(np.float32)
        return depth
