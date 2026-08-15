"""
orthotrack/depth_estimators/metric3d_estimator.py
=================================================
Metric3D v2 monocular depth estimator.

Metric3D uses canonical camera space normalization to produce metric depth
from a single image. It requires an approximate focal length but is robust to
focal length errors. Surface normals are also available in v2."""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from orthotrack.depth_estimators.base_depth_estimator import BaseDepthEstimator

# Optional: path to local Metric3D clone
_METRIC3D_DIR = Path(__file__).resolve().parents[2] / 'thirdparty' / 'Metric3D'


class Metric3DEstimator(BaseDepthEstimator):
    """Metric3D v2 depth estimator.

    Requires focal length information (extracted from intrinsics).
    If intrinsics is None during estimate(), uses a default focal length
    estimate for typical UAV imagery."""

    name: str = "metric3d_v2"
    is_metric: bool = True
    input_resolution: Optional[int] = 616  # Metric3D default

    # Metric3D input size: (H_pad, W_pad) = (616, 1064) for 16:9 images
    # The model internally pads to the nearest multiple of 14.
    _INPUT_SIZE = (616, 1064)

    def __init__(self, variant: str = 'large', device: str = 'cuda',
                 use_local: bool = False):
        """
        Args:
            variant:    'small' | 'large' | 'giant2'. Default 'large'.
            device:     Target device.
            use_local:  If True, loads from thirdparty/Metric3D instead of torch.hub."""
        import torch

        hub_name_map = {
            'small': 'metric3d_vit_small',
            'large': 'metric3d_vit_large',
            'giant2': 'metric3d_vit_giant2',
        }
        hub_name = hub_name_map.get(variant, 'metric3d_vit_large')

        if use_local and _METRIC3D_DIR.exists():
            if str(_METRIC3D_DIR) not in sys.path:
                sys.path.insert(0, str(_METRIC3D_DIR))
            print(f"Loading Metric3D v2 ({variant}) from local clone ...")
            torch.hub.set_dir(str(_METRIC3D_DIR))
            self.model = torch.hub.load(
                str(_METRIC3D_DIR), hub_name, pretrain=True, source='local', trust_repo=True
            )
        else:
            print(f"Loading Metric3D v2 ({variant}) from torch.hub ...")
            self.model = torch.hub.load(
                'yvanyin/metric3d', hub_name, pretrain=True, trust_repo=True
            )

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
            intrinsics: (3, 3) camera intrinsics matrix. Metric3D uses fx and fy
                        to perform canonical camera space normalization.
                        If None, uses a default focal length.

        Returns:
            depth: (H, W) float32 metric depth in metres."""
        import torch
        import torch.nn.functional as F
        import cv2

        H_orig, W_orig = image.shape[:2]

        # Extract focal length from intrinsics (or use default)
        if intrinsics is not None:
            fx = float(intrinsics[0, 0])
            fy = float(intrinsics[1, 1])
        else:
            # Default: assume ~60 deg vertical FOV for typical UAV camera
            fx = fy = float(max(H_orig, W_orig)) * 0.8

        # Metric3D canonical camera: 1000 px focal, 1000x1000 image
        # We scale the image and adjust focal accordingly for canonical space
        input_size = self._INPUT_SIZE  # (H_input, W_input)
        h_scale = input_size[0] / H_orig
        w_scale = input_size[1] / W_orig
        scale = min(h_scale, w_scale)

        # Resize image to Metric3D input size
        img_resized = cv2.resize(image, (input_size[1], input_size[0]))

        # Adjust focal length for rescaled image
        fx_new = fx * (input_size[1] / W_orig)
        fy_new = fy * (input_size[0] / H_orig)

        # Normalize (ImageNet style by default in Metric3D)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std  = np.array([58.395,  57.12,  57.375], dtype=np.float32)
        img_norm = (img_resized.astype(np.float32) - mean) / std
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            pred_depth, confidence, output_dict = self.model.inference({'input': img_tensor})

        # pred_depth: (1, 1, H, W) or (1, H, W) in canonical space
        depth_pred = pred_depth.squeeze().cpu().numpy().astype(np.float32)

        # Rescale depth from canonical focal to actual focal
        # Metric3D uses: depth_metric = depth_canonical * (focal_canonical / focal_actual)
        focal_canonical = 1000.0  # Metric3D canonical focal length
        depth_pred = depth_pred * (focal_canonical / ((fx_new + fy_new) / 2.0))

        # Resize back to original resolution
        depth_pred = cv2.resize(depth_pred, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        depth_pred = np.maximum(depth_pred, 0.0)

        return depth_pred.astype(np.float32)
