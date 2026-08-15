"""
Data types for the OrthoTrack tracking pipeline."""

from dataclasses import dataclass


@dataclass
class FrameResult:
    """Result for a single frame."""
    frame_id: int
    is_keyframe: bool
    success: bool
    est_x: float = None
    est_y: float = None
    est_z: float = None
    est_qw: float = None
    est_qx: float = None
    est_qy: float = None
    est_qz: float = None
    gt_x: float = None
    gt_y: float = None
    gt_z: float = None
    gt_qw: float = None
    gt_qx: float = None
    gt_qy: float = None
    gt_qz: float = None
    position_error: float = None  # 3D position error
    horizontal_error: float = None  # 2D horizontal error
    vertical_error: float = None  # Altitude error
    rotation_error: float = None  # Rotation error in degrees (if available)
    reproj_error: float = None  # Reprojection error in pixels
    num_tracked_points: int = 0
    num_inliers: int = 0
    tracked_points_threshold: int = 0  # Dynamic threshold for point loss trigger
    method: str = "none"  # "keyframe", "tracked", "predicted"
    processing_time: float = 0.0
    inference_time: float = 0.0  # Model inference time only (no data loading)
    kf_reason: str = ""  # Keyframe trigger reason (e.g. "reproj_abs", "low_points")
    baseline_reproj: float = 0.0  # Reproj baseline used for threshold computation
