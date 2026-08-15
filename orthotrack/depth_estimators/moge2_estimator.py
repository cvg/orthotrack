"""
orthotrack/depth_estimators/moge2_estimator.py
==============================================
Microsoft MoGe v2 monocular geometry estimator.

MoGe v2 estimates metric 3D point maps (XYZ in camera space) from a single image.
Depth is extracted from the Z-component of the predicted point map. MoGe v2 also
supports FOV conditioning and produces surface normals."""

import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from orthotrack.depth_estimators.base_depth_estimator import BaseDepthEstimator


class MoGe2Estimator(BaseDepthEstimator):
    """MoGe v2 metric depth estimator (Microsoft).

    Requires: pip install git+https://github.com/microsoft/MoGe.git

    MoGe-2 outputs metric point maps; this wrapper extracts the Z-component
    as depth.  Optionally accepts FOV (horizontal field of view) via intrinsics."""

    name: str = "moge2"
    is_metric: bool = True
    input_resolution: Optional[int] = 518

    # HuggingFace model ID for MoGe-2
    _HF_MODEL_ID = "Ruicheng/moge-2-vitl"

    def __init__(self, device: str = 'cuda', fov: Optional[float] = None):
        """
        Args:
            device: Target device.
            fov:    Optional horizontal FOV in degrees to condition the model.
                    If None, the model predicts FOV internally."""
        try:
            from moge.model.v2 import MoGeModel
        except ImportError:
            raise ImportError(
                "MoGe not installed.\n"
                "  pip install git+https://github.com/microsoft/MoGe.git"
            )

        print(f"Loading MoGe-2 from HuggingFace ({self._HF_MODEL_ID}) ...")
        self.model = MoGeModel.from_pretrained(self._HF_MODEL_ID).to(device).eval()
        self.device = device
        self.fov = fov

    def estimate(
        self,
        image: np.ndarray,
        intrinsics: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Args:
            image:      (H, W, 3) uint8 RGB image.
            intrinsics: Optional (3, 3) camera intrinsics. If provided,
                        the horizontal FOV is derived from fx and image width.

        Returns:
            depth: (H, W) float32 metric depth in metres (Z-component of point map)."""
        import torch
        import torch.nn.functional as F

        H, W = image.shape[:2]

        # Compute FOV from intrinsics if available
        fov = self.fov
        if fov is None and intrinsics is not None:
            fx = float(intrinsics[0, 0])
            fov = float(2 * np.degrees(np.arctan(W / (2 * fx))))

        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)

        with torch.no_grad():
            output = self.model.infer(
                img_tensor,
                fov_x=fov,
                use_fp16=True,
            )

        # output['points']: (1, H', W', 3) XYZ in camera space
        # output['depth']:  (1, H', W') if available, else use Z from points
        if 'depth' in output and output['depth'] is not None:
            depth = output['depth'].squeeze().cpu().numpy().astype(np.float32)
        else:
            pts = output['points'].squeeze(-1).cpu()  # (1, H', W', 3) -> wrong; fix:
            # output['points'] shape: (1, H', W', 3)
            pts = output['points'].squeeze(0).cpu().numpy()  # (H', W', 3)
            depth = pts[..., 2].astype(np.float32)  # Z component

        return depth
