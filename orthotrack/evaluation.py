"""
Error computation and result printing / saving for the tracking pipeline."""

import numpy as np
from typing import Dict, List, Optional

from utils.pose import CameraPose
from orthotrack.types import FrameResult


def compute_errors(est_pos: np.ndarray, gt_pose: CameraPose,
                   est_rotation: np.ndarray = None) -> Dict:
    """Compute position and rotation errors.

    Args:
        est_pos: (3,) estimated position in UTM
        gt_pose: Ground-truth CameraPose, or None when GT is unavailable
        est_rotation: (3,3) estimated camera-to-world rotation (optional)

    Returns:
        Dict with position_error, horizontal_error, vertical_error, rotation_error.
        All values are None when gt_pose is None."""
    if gt_pose is None or gt_pose.x is None:
        return {
            'position_error': None,
            'horizontal_error': None,
            'vertical_error': None,
            'rotation_error': None,
        }
    gt_pos = np.array([gt_pose.x, gt_pose.y, gt_pose.z])
    errors: Dict[str, Optional[float]] = {
        'position_error': float(np.linalg.norm(est_pos - gt_pos)),
        'horizontal_error': float(np.linalg.norm(est_pos[:2] - gt_pos[:2])),
        'vertical_error': float(abs(est_pos[2] - gt_pose.z)),
        'rotation_error': None,
    }

    if est_rotation is not None:
        R_gt = gt_pose.rotation_matrix
        R_diff = est_rotation @ R_gt.T
        trace = np.clip(np.trace(R_diff), -1.0, 3.0)
        angle_rad = np.arccos((trace - 1) / 2)
        errors['rotation_error'] = float(np.rad2deg(angle_rad))

    return errors


def print_summary(results: List[FrameResult]):
    """Print a summary table for the given results."""
    successful = [r for r in results if r.success and r.position_error is not None]
    keyframes = [r for r in results if r.is_keyframe and r.success]
    tracked = [r for r in results if r.method == "tracked"]
    predicted = [r for r in results if r.method == "predicted"]

    print("\n" + "=" * 60)
    print("TRACKING PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total frames: {len(results)}")
    print(f"Keyframes: {len(keyframes)} ({len(keyframes) / len(results) * 100:.1f}%)")
    print(f"Tracked: {len(tracked)} ({len(tracked) / len(results) * 100:.1f}%)")
    print(f"Predicted: {len(predicted)} ({len(predicted) / len(results) * 100:.1f}%)")

    if successful:
        pos_errors = [r.position_error for r in successful]

        print(f"\nPosition Error (m):")
        print(f"  Mean: {np.mean(pos_errors):.2f}")
        print(f"  Median: {np.median(pos_errors):.2f}")
        print(f"  Std: {np.std(pos_errors):.2f}")
        print(f"  Min: {np.min(pos_errors):.2f}")
        print(f"  Max: {np.max(pos_errors):.2f}")

        for method, name in [("keyframe", "Keyframes"), ("tracked", "Tracked"), ("predicted", "Predicted")]:
            method_results = [r for r in successful if r.method == method]
            if method_results:
                method_errors = [r.position_error for r in method_results]
                print(f"\n{name}:")
                print(f"  Mean: {np.mean(method_errors):.2f}m, Median: {np.median(method_errors):.2f}m")

    times = [r.processing_time for r in results]
    print(f"\nProcessing:")
    print(f"  Total: {sum(times):.1f}s")
    print(f"  Mean: {np.mean(times):.3f}s/frame")
    print(f"  FPS: {len(results) / sum(times):.2f}")

    print("=" * 60)
