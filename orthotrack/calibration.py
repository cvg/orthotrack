"""
Self-calibration of camera intrinsics from 2D-3D correspondences.

Sweeps vertical FoV candidates, evaluates each via PnP RANSAC inlier count
(with angularly-scaled thresholds for fairness), then optionally refines the
winner with cv2.calibrateCamera."""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict

from orthotrack.exceptions import PnPSolverError


def _safe_solvePnPRansac(*args, **kwargs):
    """cv2.solvePnPRansac — raises PnPSolverError on OpenCV failure."""
    try:
        return cv2.solvePnPRansac(*args, **kwargs)
    except cv2.error as e:
        raise PnPSolverError(f"solvePnPRansac: {e}") from e


def calibrate_intrinsics(
    pts_2d: np.ndarray,
    pts_3d: np.ndarray,
    image_size: Tuple[int, int],
    fov_candidates: List[float] = None,
    reproj_threshold: float = 8.0,
    verbose: bool = False,
    prefer_reproj: bool = False,
) -> Optional[Dict]:
    """Calibrate camera intrinsics by sweeping FoV candidates + PnP RANSAC.

    Each candidate FoV is evaluated purely by solvePnPRansac inlier count
    (with an angularly-scaled RANSAC threshold for fairness). The winner is
    selected by max inliers, with a final cv2.calibrateCamera refinement that
    is accepted only if the refined fx stays within +/-8% of the grid fx.

    Parameters
    ----------
    pts_2d : (N, 2)  2D image points.
    pts_3d : (N, 3)  3D world points.
    image_size : (height, width).
    fov_candidates : Vertical FoV values (degrees) to sweep.
    reproj_threshold : PnP RANSAC reprojection threshold in pixels at FoV=60 deg
        reference; scaled per candidate to maintain constant angular tolerance.
    verbose : Print per-candidate info.
    prefer_reproj : If True, select best by min reproj among top-70% inlier candidates.

    Returns
    -------
    dict with 'K', 'fov_vertical', 'fx', 'fy', 'cx', 'cy',
    'reproj_error', 'num_inliers', or None on failure."""
    if fov_candidates is None:
        fov_candidates = range(12, 91)  # 1 deg steps, 12-90 deg

    h, w = image_size
    if len(pts_2d) < 30:
        return None

    centroid = pts_3d.mean(axis=0)
    pts_3d_c = (pts_3d - centroid).astype(np.float64)
    pts_2d_f = pts_2d.astype(np.float64)

    # Reference focal length (FoV=60 deg) for angular-constant RANSAC threshold
    fy_ref_60 = h / (2 * np.tan(np.radians(60) / 2))

    candidates = []

    for fov in fov_candidates:
        fy = h / (2 * np.tan(np.radians(fov) / 2))
        fx = fy
        K = np.array([[fx, 0, w / 2.0], [0, fy, h / 2.0], [0, 0, 1]], dtype=np.float64)

        # Scale threshold to maintain the same angular tolerance across FoVs
        scaled_threshold = reproj_threshold * (fy / fy_ref_60)

        try:
            success, rvec, tvec, inliers = _safe_solvePnPRansac(
                pts_3d_c.reshape(-1, 1, 3),
                pts_2d_f.reshape(-1, 1, 2),
                K, None,
                iterationsCount=5000,
                reprojectionError=scaled_threshold,
                flags=cv2.SOLVEPNP_SQPNP,
            )
        except PnPSolverError:
            try:
                success, rvec, tvec, inliers = _safe_solvePnPRansac(
                    pts_3d_c.reshape(-1, 1, 3),
                    pts_2d_f.reshape(-1, 1, 2),
                    K, None,
                    iterationsCount=5000,
                    reprojectionError=scaled_threshold,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            except PnPSolverError:
                continue

        if not success or inliers is None or len(inliers) < 15:
            continue

        # Recount inliers with a fixed angular threshold (reproj_threshold @ FoV=60)
        proj_all, _ = cv2.projectPoints(pts_3d_c, rvec, tvec, K, None)
        err_all = np.linalg.norm(pts_2d_f - proj_all.reshape(-1, 2), axis=1)
        idx = np.where(err_all < reproj_threshold)[0]
        n_inliers = len(idx)
        if n_inliers < 15:
            continue

        reproj_err = float(np.mean(err_all[idx]))

        R, _ = cv2.Rodrigues(rvec)
        est_z = float((-R.T @ tvec.flatten() + centroid)[2])

        if verbose:
            print(f"    FoV={fov} deg (fx={fx:.0f}): {n_inliers} inliers, "
                  f"reproj={reproj_err:.2f}px, thr={scaled_threshold:.1f}px, z={est_z:.0f}m")

        angular_reproj_err = reproj_err / fy

        candidates.append({
            'fov': float(fov), 'K': K, 'fx': fx, 'fy': fy,
            'cx': w / 2.0, 'cy': h / 2.0,
            'num_inliers': n_inliers, 'reproj_error': reproj_err,
            'angular_reproj': angular_reproj_err,
            'rvec': rvec, 'tvec': tvec, 'inlier_idx': idx,
            'est_z': est_z,
        })

    if not candidates:
        if verbose:
            print("  Calibration: no FoV candidate succeeded")
        return None

    plausible = candidates

    if prefer_reproj:
        max_inliers = max(c['num_inliers'] for c in plausible)
        eligible = [c for c in plausible if c['num_inliers'] >= 0.70 * max_inliers]
        best = min(eligible, key=lambda c: c['reproj_error'])
        if verbose:
            print(f"  Calibration: reproj-first selection "
                  f"({len(eligible)} eligible at >=70% of {max_inliers} inliers)")
    else:
        _max_inl = max(c['num_inliers'] for c in plausible)
        _min_rp = min(c['reproj_error'] for c in plausible)
        _rp_range = max(c['reproj_error'] for c in plausible) - _min_rp or 1.0
        for c in plausible:
            c['combined_score'] = (c['num_inliers'] / _max_inl) * \
                                  (1.0 - (c['reproj_error'] - _min_rp) / _rp_range)
        best = max(plausible, key=lambda c: c['combined_score'])

    if verbose:
        print(f"  Calibration sweep: best FoV={best['fov']:.1f} deg "
              f"(fx={best['fx']:.0f}, {best['num_inliers']} inliers, "
              f"reproj={best['reproj_error']:.2f}px, z={best['est_z']:.0f}m)")

    # Final calibrateCamera refinement on the winner
    idx = best['inlier_idx']
    if len(idx) >= 50:
        obj_pts = pts_3d_c[idx].reshape(-1, 1, 3).astype(np.float32)
        img_pts = pts_2d_f[idx].reshape(-1, 1, 2).astype(np.float32)
        try:
            ret_cal, K_ref, _dist, rvecs_cal, tvecs_cal = cv2.calibrateCamera(
                [obj_pts], [img_pts], (w, h),
                best['K'].copy(), None,
                flags=(cv2.CALIB_USE_INTRINSIC_GUESS |
                       cv2.CALIB_FIX_ASPECT_RATIO |
                       cv2.CALIB_FIX_PRINCIPAL_POINT |
                       cv2.CALIB_ZERO_TANGENT_DIST |
                       cv2.CALIB_FIX_K1 |
                       cv2.CALIB_FIX_K2 |
                       cv2.CALIB_FIX_K3),
            )
            fx_init = best['fx']
            fx_refined = float(K_ref[0, 0])
            if abs(fx_refined / fx_init - 1.0) <= 0.08:
                fov_refined = float(2 * np.degrees(np.arctan(h / (2 * fx_refined))))
                best.update({
                    'K': K_ref.astype(np.float64),
                    'fx': fx_refined, 'fy': float(K_ref[1, 1]),
                    'cx': float(K_ref[0, 2]), 'cy': float(K_ref[1, 2]),
                    'reproj_error': float(ret_cal),
                    'fov': fov_refined,
                })
                if verbose:
                    print(f"  Calibration refined: FoV={fov_refined:.1f} deg, "
                          f"fx={fx_refined:.1f} (delta={fx_refined-fx_init:+.1f}), "
                          f"reproj={ret_cal:.2f}px")
            else:
                if verbose:
                    print(f"  Calibration refinement rejected: "
                          f"fx {fx_init:.0f} -> {fx_refined:.0f} "
                          f"(ratio={fx_refined/fx_init:.3f}, >8% change)")
        except cv2.error as e:
            if verbose:
                print(f"  Calibration refinement failed: {e}")

    return {
        'K': best['K'],
        'fov_vertical': best['fov'],
        'fx': best['fx'],
        'fy': best['fy'],
        'cx': best['cx'],
        'cy': best['cy'],
        'reproj_error': best['reproj_error'],
        'num_inliers': best['num_inliers'],
        'candidates': candidates,
        'rvec': best['rvec'],
        'tvec': best['tvec'],
        'centroid': centroid,
    }
