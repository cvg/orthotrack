"""
orthotrack/depth_estimators/marigold_estimator.py
=================================================
Marigold affine-invariant relative depth estimator (ETH Zurich / CVPR 2024).

Marigold fine-tunes a Stable Diffusion v2 backbone for monocular depth and
surface normals. It produces affine-invariant (relative) depth — no metric scale.
The benchmark applies median scale alignment for the scaled table."""

import os
# Prevent TensorFlow (compiled against NumPy 1.x) from being imported by
# transformers/diffusers, which would crash with NumPy 2.x.
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_JAX', '0')

from pathlib import Path
from typing import Optional

import numpy as np

from orthotrack.depth_estimators.base_depth_estimator import BaseDepthEstimator

_HF_ID = "prs-eth/marigold-depth-v1-1"


class MarigoldEstimator(BaseDepthEstimator):
    """Marigold relative depth estimator (diffusion-based).

    Requires: pip install diffusers>=0.28.0 transformers accelerate"""

    name: str = "marigold"
    is_metric: bool = False
    input_resolution: Optional[int] = 768

    def __init__(self, device: str = 'cuda', n_steps: int = 10,
                 ensemble_size: int = 1):
        """
        Args:
            device:        Target device.
            n_steps:       Number of diffusion denoising steps (10 = fast, 50 = best).
            ensemble_size: Number of ensembled predictions (1 = fast, 10 = best)."""
        from diffusers import MarigoldDepthPipeline

        print(f"Loading Marigold from HuggingFace ({_HF_ID}) ...")
        import torch
        self.pipe = MarigoldDepthPipeline.from_pretrained(
            _HF_ID,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            variant='fp16' if device == 'cuda' else None,
        )
        self.pipe = self.pipe.to(device)
        self.n_steps = n_steps
        self.ensemble_size = ensemble_size
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
            depth: (H, W) float32 relative depth in [0, 1] range
                   (larger = farther from camera)."""
        from PIL import Image as PILImage
        import torch

        pil_img = PILImage.fromarray(image)
        with torch.no_grad():
            pipeline_output = self.pipe(
                pil_img,
                num_inference_steps=self.n_steps,
                ensemble_size=self.ensemble_size,
                match_input_resolution=True,
            )

        # pipeline_output.prediction: (1, H, W) float32 in [0, 1]
        pred = pipeline_output.prediction
        # Handle both torch Tensor and numpy ndarray outputs
        if hasattr(pred, 'numpy'):
            depth = pred.squeeze().numpy().astype(np.float32)
        else:
            depth = np.squeeze(np.asarray(pred, dtype=np.float32))
        # MarigoldDepthPipeline outputs affine-invariant depth in [0, 1]
        # where 0 = nearest and 1 = farthest (larger value = farther).
        # Clip to avoid exact zeros.
        depth = np.clip(depth, 1e-4, 1.0)
        return depth
