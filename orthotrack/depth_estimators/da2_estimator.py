"""
orthotrack/depth_estimators/da2_estimator.py
============================================
Depth Anything V2 monocular depth estimator.

Depth Anything V2 provides both relative (affine-invariant) and metric
(zero-shot) depth estimation at multiple model scales.

Two variants are available:
    - Relative: model trained purely for relative depth quality
      (requires scale alignment for metric comparison).
    - Metric: fine-tuned head for metric outdoor depth estimation."""

import os
# Prevent TensorFlow (compiled against NumPy 1.x) from being imported by
# transformers, which would crash with NumPy 2.x due to binary incompatibility.
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_JAX', '0')
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from orthotrack.depth_estimators.base_depth_estimator import BaseDepthEstimator

# Path to local DA2 clone
_DA2_DIR = Path(__file__).resolve().parents[2] / 'thirdparty' / 'Depth-Anything-V2'


class DepthAnythingV2Estimator(BaseDepthEstimator):
    """Depth Anything V2 relative depth estimator.

    Uses the HuggingFace Transformers pipeline. Outputs affine-invariant
    (relative) depth; requires scale alignment for metric comparison."""

    name: str = "da2_relative"
    is_metric: bool = False
    input_resolution: Optional[int] = 518

    # HuggingFace model IDs for DA2 relative
    _HF_IDS = {
        'small':  'depth-anything/Depth-Anything-V2-Small-hf',
        'base':   'depth-anything/Depth-Anything-V2-Base-hf',
        'large':  'depth-anything/Depth-Anything-V2-Large-hf',
    }

    def __init__(self, variant: str = 'large', device: str = 'cuda'):
        """
        Args:
            variant: 'small' | 'base' | 'large'. Default 'large'.
            device:  Target device."""
        import torch

        hf_id = self._HF_IDS.get(variant, self._HF_IDS['large'])
        print(f"Loading Depth Anything V2 relative ({variant}) from HuggingFace ({hf_id}) ...")
        from transformers import pipeline as hf_pipeline
        self.pipe = hf_pipeline(
            task='depth-estimation',
            model=hf_id,
            device=0 if device == 'cuda' else -1,
        )
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
            depth: (H, W) float32 relative depth where larger values = farther."""
        from PIL import Image as PILImage
        import torch.nn.functional as F

        pil_img = PILImage.fromarray(image)
        result = self.pipe(pil_img)
        # result['predicted_depth'] is a Tensor (H, W) or (1, H, W).
        # HF depth-estimation pipeline returns DPT-style *disparity* (near=high, far=low).
        # Invert to depth (near=low, far=high) so scale alignment works correctly.
        disp = result['predicted_depth'].squeeze()
        if hasattr(disp, 'numpy'):
            disp = disp.numpy()
        disp = disp.astype(np.float32)
        depth = 1.0 / np.maximum(disp, 1e-6)
        return depth


class DepthAnythingV2MetricEstimator(BaseDepthEstimator):
    """Depth Anything V2 metric outdoor depth estimator.

    Uses the metric fine-tuned variant from the local thirdparty clone.
    Outputs depth in metres calibrated for outdoor scenes."""

    name: str = "da2_metric"
    is_metric: bool = True
    input_resolution: Optional[int] = 518

    # HuggingFace model IDs for DA2 metric (outdoor fine-tuned)
    _HF_IDS = {
        'small':  'depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf',
        'base':   'depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf',
        'large':  'depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf',
    }

    def __init__(self, variant: str = 'large', device: str = 'cuda'):
        """
        Args:
            variant: 'small' | 'base' | 'large'. Default 'large'.
            device:  Target device."""
        hf_id = self._HF_IDS.get(variant, self._HF_IDS['large'])
        print(f"Loading Depth Anything V2 metric-outdoor ({variant}) from HuggingFace ({hf_id}) ...")
        # Always use HuggingFace Transformers pipeline (metric outdoor variant).
        self._load_from_hf(hf_id, device)
        self.device = device


    def _load_from_hf(self, hf_id: str, device: str):
        """Load Depth Anything V2 metric via HuggingFace Transformers pipeline."""
        from transformers import pipeline as hf_pipeline
        import torch
        self.pipe = hf_pipeline(
            task='depth-estimation',
            model=hf_id,
            device=0 if device == 'cuda' else -1,
        )
        self._torch = torch
        self._use_pipe = True

    def estimate(
        self,
        image: np.ndarray,
        intrinsics: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Args:
            image:      (H, W, 3) uint8 RGB image.
            intrinsics: Not used.

        Returns:
            depth: (H, W) float32 metric depth in metres."""
        if self._use_pipe:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image)
            result = self.pipe(pil_img)
            depth = result['predicted_depth'].squeeze()
            if hasattr(depth, 'numpy'):
                depth = depth.numpy()
            return depth.astype(np.float32)

        import torch
        import cv2

        # Local model: use the raw forward pass
        img_norm = (image.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) \
                   / np.array([0.229, 0.224, 0.225])
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            depth = self.model(img_tensor)

        depth = depth.squeeze().cpu().numpy().astype(np.float32)
        return depth
