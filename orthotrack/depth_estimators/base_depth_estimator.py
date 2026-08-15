"""
orthotrack/depth_estimators/base_depth_estimator.py
===================================================
Abstract base class for monocular depth estimators in OrthoTrack.

Any estimator plugged into the depth benchmarking pipeline must implement
the ``estimate()`` method defined here."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseDepthEstimator(ABC):
    """Base class for monocular depth estimators used in the OrthoTrack pipeline.

    Subclasses must implement:
        - ``estimate(image, intrinsics) -> np.ndarray``"""

    # Human-readable name (override in subclasses)
    name: str = "base"

    # Whether the model outputs metric depth (True) or relative/up-to-scale depth (False)
    is_metric: bool = False

    # Optional: expected input resolution (None = flexible)
    input_resolution: Optional[int] = None

    # ------------------------------------------------------------------
    #  Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def estimate(
        self,
        image: np.ndarray,
        intrinsics: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Estimate depth for a single RGB image.

        Args:
            image:      (H, W, 3) uint8 RGB image.
            intrinsics: Optional (3, 3) camera intrinsics matrix.
                        Some metric estimators (Metric3D, UniDepth, Depth Pro)
                        use this to produce better metric-scaled output.
                        Relative estimators typically ignore it.

        Returns:
            depth: (H, W) float32 depth map.
                - For metric estimators: depth in metres.
                - For relative estimators: up-to-scale values (positive).
                The output spatial resolution may differ from the input;
                the benchmark script handles resizing."""
        ...

    # ------------------------------------------------------------------
    #  Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, metric={self.is_metric})"
