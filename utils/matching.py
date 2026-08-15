"""
Matching utilities for OrthoTrack.
Handles RANSAC filtering, view reliability computation, and match visualization."""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Result of matching a query image to a DOP crop."""
    # Matched keypoints in query image (N, 2) in pixel coordinates
    kpts_query: np.ndarray
    # Matched keypoints in DOP crop (N, 2) in pixel coordinates  
    kpts_dop: np.ndarray
    # Confidence scores for each match
    confidences: np.ndarray
    # Original sizes
    query_size: Tuple[int, int]  # (H, W)
    dop_size: Tuple[int, int]  # (H, W)


@dataclass
class BatchMatchResult:
    """Result of batch matching against multiple DOP crops."""
    # All matched keypoints in query image (N, 2) in pixel coordinates
    kpts_query: np.ndarray
    # All matched keypoints in UTM coordinates (N, 2) - aggregated from crops
    kpts_utm: np.ndarray
    # Confidence scores for each match
    confidences: np.ndarray
    # Which crop each match came from
    crop_indices: np.ndarray
    # Query image size
    query_size: Tuple[int, int]




def compute_view_reliability(pitch: float) -> float:
    """
    Compute reliability of DOP-based matching based on camera pitch angle.
    
    DOP is nadir (looking straight down), so matching is most reliable when
    drone camera is also looking down. As pitch becomes more oblique (horizontal),
    perspective mismatch increases and matching becomes less reliable.
    
    Args:
        pitch: Camera pitch in degrees (negative = looking down)
               -90° = nadir, 0° = horizontal
               
    Returns:
        reliability: Value in [0, 1], where 1 = fully reliable, 0 = unreliable"""
    # Angle from nadir: 0° = nadir, 90° = horizontal
    angle_from_nadir = 90.0 - abs(pitch)
    
    # Reliability drops off as angle increases
    # - 0-30° from nadir: high reliability (0.8-1.0)
    # - 30-60° from nadir: medium reliability (0.4-0.8)
    # - 60-90° from nadir: low reliability (0.0-0.4)
    
    if angle_from_nadir <= 30:
        reliability = 1.0 - (angle_from_nadir / 30) * 0.2
    elif angle_from_nadir <= 60:
        reliability = 0.8 - ((angle_from_nadir - 30) / 30) * 0.4
    else:
        reliability = 0.4 - ((angle_from_nadir - 60) / 30) * 0.4
    
    return max(0.0, min(1.0, reliability))


def compute_required_inliers(pitch: float, base_inliers: int = 100) -> int:
    """
    Compute required inliers for accepting a keyframe based on view angle.
    
    For oblique views, we should accept fewer inliers since matching is harder.
    For nadir views, we expect more inliers since matching should be good.
    
    Args:
        pitch: Camera pitch in degrees
        base_inliers: Base requirement for nadir views
        
    Returns:
        Required number of inliers"""
    reliability = compute_view_reliability(pitch)
    
    # Scale down requirements for oblique views
    # nadir: require base_inliers
    # oblique: require as low as base_inliers * 0.3
    required = int(base_inliers * (0.3 + 0.7 * reliability))
    
    return max(30, required)  # Minimum 30 inliers






