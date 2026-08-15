"""
Trajectory alignment utilities for comparing predicted and ground-truth camera poses.

Implements Umeyama alignment (7-DoF similarity transform: rotation + translation + scale)
and SE(3) alignment (6-DoF rigid transform: rotation + translation, no scale).

Used to align foundation model outputs (which produce poses in an arbitrary coordinate 
frame) to ground-truth UTM poses before computing localization metrics."""

import numpy as np
from typing import Tuple, Optional


def umeyama_alignment(
    src: np.ndarray,
    tgt: np.ndarray,
    weights: Optional[np.ndarray] = None,
    with_scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the optimal similarity transform (rotation, translation, scale)
    that aligns src points to tgt points using the Umeyama method.

    Minimizes: sum_i w_i * ||s * R @ src_i + t - tgt_i||^2

    Args:
        src: (N, 3) source points (e.g., predicted camera positions).
        tgt: (N, 3) target points (e.g., ground-truth camera positions).
        weights: (N,) optional per-point weights. If None, uniform weights.
        with_scale: If True, estimate scale (7-DoF Sim(3)). 
                    If False, fix scale=1 (6-DoF SE(3) Procrustes).

    Returns:
        R: (3, 3) rotation matrix.
        t: (3,) translation vector.
        s: scalar scale factor. (1.0 if with_scale=False)

    The aligned source points are: s * R @ src_i + t ≈ tgt_i
    Or equivalently: aligned = s * (src @ R.T) + t"""
    assert src.shape == tgt.shape, f"Shape mismatch: {src.shape} vs {tgt.shape}"
    assert src.shape[1] == 3, f"Expected (N, 3), got {src.shape}"
    n = src.shape[0]
    assert n >= 3, f"Need at least 3 points, got {n}"

    if weights is not None:
        assert weights.shape == (n,), f"Weights shape mismatch: {weights.shape} vs ({n},)"
        w = weights / weights.sum()  # Normalize weights
    else:
        w = np.ones(n) / n

    # Weighted centroids
    mu_src = (w[:, None] * src).sum(axis=0)  # (3,)
    mu_tgt = (w[:, None] * tgt).sum(axis=0)  # (3,)

    # Center the points
    src_c = src - mu_src  # (N, 3)
    tgt_c = tgt - mu_tgt  # (N, 3)

    # Weighted covariance matrix
    # Sigma = sum_i w_i * tgt_c_i @ src_c_i^T  => (3, 3)
    sigma = (w[:, None, None] * (tgt_c[:, :, None] @ src_c[:, None, :])).sum(axis=0)

    # SVD
    U, D, Vt = np.linalg.svd(sigma)

    # Ensure proper rotation (det = +1)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt

    if with_scale:
        # Weighted variance of source points
        var_src = (w[:, None] * (src_c ** 2)).sum()
        # Scale
        s = np.trace(np.diag(D) @ S) / var_src
    else:
        s = 1.0

    t = mu_tgt - s * R @ mu_src

    return R, t, s


def align_trajectories(
    positions_pred: np.ndarray,
    positions_gt: np.ndarray,
    rotations_pred: Optional[np.ndarray] = None,
    with_scale: bool = True,
) -> dict:
    """
    Align predicted trajectory to ground-truth using Umeyama, then compute errors.

    Args:
        positions_pred: (N, 3) predicted camera positions.
        positions_gt: (N, 3) ground-truth camera positions.
        rotations_pred: (N, 3, 3) optional predicted rotation matrices. 
                        If provided, will also be aligned and returned.
        with_scale: If True, estimate scale (Sim(3)). If False, rigid (SE(3)).

    Returns:
        dict with:
            'R_align': (3, 3) alignment rotation
            't_align': (3,) alignment translation
            's_align': float alignment scale
            'positions_aligned': (N, 3) aligned predicted positions
            'rotations_aligned': (N, 3, 3) aligned predicted rotations (if rotations_pred given)
            'position_errors': (N,) per-frame position error in meters
            'mean_position_error': float
            'median_position_error': float"""
    R, t, s = umeyama_alignment(positions_pred, positions_gt, with_scale=with_scale)

    # Apply alignment: aligned = s * R @ pred + t
    positions_aligned = s * (positions_pred @ R.T) + t

    result = {
        'R_align': R,
        't_align': t,
        's_align': s,
        'positions_aligned': positions_aligned,
        'position_errors': np.linalg.norm(positions_aligned - positions_gt, axis=1),
    }
    result['mean_position_error'] = float(np.mean(result['position_errors']))
    result['median_position_error'] = float(np.median(result['position_errors']))

    if rotations_pred is not None:
        # Align rotations: R_aligned = R_align @ R_pred
        rotations_aligned = np.einsum('ij,njk->nik', R, rotations_pred)
        result['rotations_aligned'] = rotations_aligned

    return result




def align_first_frame(
    positions_pred: np.ndarray,
    positions_gt: np.ndarray,
    rotations_pred: Optional[np.ndarray] = None,
    rotations_gt: Optional[np.ndarray] = None,
    with_scale: bool = False,
) -> dict:
    """
    Anchor the predicted trajectory so that the first predicted frame exactly matches
    the first GT frame.  Measures trajectory drift (how far the trajectory strays from
    GT when the starting pose is perfect).

    Alignment strategy
    ------------------
    1. Rotation  — derived from the first-frame **C2W** rotations so that the
       predicted camera at t=0 has the same orientation as the GT camera at t=0:
           R_align = R_c2w_gt[0] @ R_c2w_pred[0]^T
       Applied to world-space positions:  pos_aligned[i] = s * R_align @ pos_pred[i] + t
       Applied to C2W rotations:          rot_aligned[i] = R_align @ rot_pred[i]

    2. Scale  (``with_scale=False`` → s=1, ``with_scale=True`` → LS-estimated):
       When with_scale=True, solve for the scalar s that minimises total position
       error *subject to the constraint that the first aligned position == gt[0]*:
           min_s  ‖s·(R_align @ pos_pred − R_align @ pos_pred[0]) − (pos_gt − pos_gt[0])‖²
           ⟹  s = ΣᵢAᵢ·Bᵢ / ΣᵢAᵢ·Aᵢ   (A = rotated_pred − rotated_pred[0],
                                          B = pos_gt − pos_gt[0])
       The translation is then set to enforce the first-frame anchor exactly:
           t = pos_gt[0] − s · (R_align @ pos_pred[0])

    3. Translation  — always chosen to give zero position error at t=0:
           t = pos_gt[0] − s · rotated_pred[0]

    Args:
        positions_pred:  (N, 3) predicted camera centres.
        positions_gt:    (N, 3) ground-truth camera centres.
        rotations_pred:  (N, 3, 3) predicted C2W rotation matrices (optional).
        rotations_gt:    (N, 3, 3) GT C2W rotation matrices (required when rotations_pred
                         is provided, used only for the first-frame rotation alignment).
        with_scale:      If False, s=1 (metric-drift mode).
                         If True,  s solved by LS (non-metric-drift mode).

    Returns:
        dict with keys:
            'R_align'             (3, 3) world-space alignment rotation
            't_align'             (3,)  alignment translation
            's_align'             float scale factor
            'positions_aligned'   (N, 3)
            'rotations_aligned'   (N, 3, 3)  — only if rotations_pred given
            'position_errors'     (N,) per-frame Euclidean position errors (metres)"""
    assert positions_pred.shape == positions_gt.shape, (
        f"Shape mismatch: {positions_pred.shape} vs {positions_gt.shape}")
    n = len(positions_pred)
    assert n >= 1

    # --- 1. Alignment rotation from first C2W frames ---
    if rotations_pred is not None and rotations_gt is not None:
        # R_align maps pred-world to GT-world such that:
        #   rot_aligned[0] = R_align @ rot_c2w_pred[0]  == rot_c2w_gt[0]
        # => R_align = rot_c2w_gt[0] @ rot_c2w_pred[0]^T  ... check:
        #   (rot_c2w_gt[0] @ rot_c2w_pred[0]^T) @ rot_c2w_pred[0]
        #   = rot_c2w_gt[0] @ I
        #   = rot_c2w_gt[0] ✓
        R_align = rotations_gt[0] @ rotations_pred[0].T  # (3, 3)
    else:
        # No rotation info — identity alignment (translation/scale only)
        R_align = np.eye(3)

    # Rotate predicted positions into GT orientation
    rotated_pred = positions_pred @ R_align.T  # (N, 3)  equiv. R_align @ pos_pred[i] per row

    # --- 2. Scale ---
    if with_scale:
        # Solve 1-D scale that minimises total error, anchored at frame 0
        # min_s ‖ s·A − B ‖²  where A[i] = rotated_pred[i] − rotated_pred[0]
        #                            B[i] = pos_gt[i]       − pos_gt[0]
        A = rotated_pred - rotated_pred[0]   # (N, 3)
        B = positions_gt - positions_gt[0]   # (N, 3)
        denom = np.sum(A * A)
        s = float(np.sum(A * B) / denom) if denom > 1e-12 else 1.0
    else:
        s = 1.0

    # --- 3. Translation — exact anchor at frame 0 ---
    t_align = positions_gt[0] - s * rotated_pred[0]  # (3,)

    # --- Apply transform ---
    positions_aligned = s * rotated_pred + t_align   # (N, 3)
    position_errors = np.linalg.norm(positions_aligned - positions_gt, axis=1)  # (N,)

    result: dict = {
        'R_align': R_align,
        't_align': t_align,
        's_align': s,
        'positions_aligned': positions_aligned,
        'position_errors': position_errors,
    }

    if rotations_pred is not None:
        # C2W rotation alignment: rot_aligned[i] = R_align @ rot_pred[i]
        result['rotations_aligned'] = np.einsum('ij,njk->nik', R_align, rotations_pred)

    return result
