"""
Abstract base matcher interface for OrthoTrack.

Any matcher that can be plugged into the tracking pipeline must implement
the ``match()`` and ``match_batch()`` methods defined here.  The existing
``RoMaV2Matcher`` already conforms to this protocol."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from utils.matching import MatchResult


class BaseMatcher(ABC):
    """Base class for all image matchers used in the OrthoTrack pipeline.

    Subclasses *must* implement:
        - ``match(query_image, dop_image, num_matches) -> MatchResult``
        - ``match_batch(query_image, dop_images, num_matches_per_crop) -> List[MatchResult]``"""

    # Human-readable name (override in subclasses)
    name: str = "base"

    # ------------------------------------------------------------------
    #  Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def match(
        self,
        query_image: np.ndarray,
        dop_image: np.ndarray,
        num_matches: int = 5000,
    ) -> MatchResult:
        """Match a query image to a single DOP crop.

        Args:
            query_image: (H, W, 3) uint8 BGR/RGB query image.
            dop_image:   (H, W, 3) uint8 BGR/RGB DOP orthophoto crop.
            num_matches:  Maximum number of correspondences to return.

        Returns:
            ``MatchResult`` with pixel-coordinate keypoints and confidences."""
        ...

    @abstractmethod
    def match_batch(
        self,
        query_image: np.ndarray,
        dop_images: List[np.ndarray],
        num_matches_per_crop: int = 3000,
    ) -> List[MatchResult]:
        """Match a query image to multiple DOP crops.

        Args:
            query_image:         (H, W, 3) uint8 query image.
            dop_images:          List of (H, W, 3) DOP crop images.
            num_matches_per_crop: Matches to sample per pair.

        Returns:
            List of ``MatchResult``, one per DOP crop."""
        ...

    # ------------------------------------------------------------------
    #  Utility (shared)
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
