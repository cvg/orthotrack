"""
Keyframe localization and pose estimation from 2D-3D correspondences.

Contains all matching, lifting-to-3D, PnP, refinement, and expanding-search
logic for keyframe-based geo-localization."""

import numpy as np
import cv2
import torch
from typing import List, Tuple, Optional, Dict

from utils.geo import (
    GeoTIFFHandler, SequenceGeoHandler,
    bilinear_sample_dsm_batch,
)

from orthotrack.matchers.base_matcher import BaseMatcher
from orthotrack.crop_strategy import get_intrinsics
from orthotrack.exceptions import PnPSolverError, InsufficientConfidentMatchesError
from orthotrack.calibration import calibrate_intrinsics  # noqa: F401 — re-exported
from utils.image import downsample_image


# ------------------------------------------------------------------ #
#  PnP wrapper (fail loudly)                                          #
# ------------------------------------------------------------------ #

def _safe_solvePnPRansac(*args, **kwargs):
    """cv2.solvePnPRansac — raises PnPSolverError on OpenCV failure."""
    try:
        return cv2.solvePnPRansac(*args, **kwargs)
    except cv2.error as e:
        raise PnPSolverError(f"solvePnPRansac: {e}") from e


# ------------------------------------------------------------------ #
#  DSM sampling helpers                                                #
# ------------------------------------------------------------------ #



def sample_full_dsm_batch(geo_handler, utm_xs: np.ndarray, utm_ys: np.ndarray) -> np.ndarray:
    """Vectorized elevation sampling from the full-resolution DSM."""
    utm_xs = np.asarray(utm_xs, dtype=np.float64)
    utm_ys = np.asarray(utm_ys, dtype=np.float64)
    if isinstance(geo_handler, SequenceGeoHandler):
        inv = ~geo_handler.dsm_transform
        cols = inv.a * utm_xs + inv.b * utm_ys + inv.c
        rows = inv.d * utm_xs + inv.e * utm_ys + inv.f
        return bilinear_sample_dsm_batch(geo_handler.dsm_data, cols, rows)
    elif isinstance(geo_handler, GeoTIFFHandler):
        geo_handler.preload(is_dsm=True)
        inv = ~geo_handler.dsm_transform
        cols = inv.a * utm_xs + inv.b * utm_ys + inv.c
        rows = inv.d * utm_xs + inv.e * utm_ys + inv.f
        return bilinear_sample_dsm_batch(geo_handler._dsm_data, cols, rows)
    return np.full(len(utm_xs), np.nan)


def get_full_dop_image(geo_handler) -> Optional[np.ndarray]:
    """Return full-resolution DOP image (H, W, 3) if available."""
    if isinstance(geo_handler, SequenceGeoHandler):
        dop = geo_handler.dop_data
    elif isinstance(geo_handler, GeoTIFFHandler):
        geo_handler.preload(is_dsm=False)
        dop_data = geo_handler._dop_data
        if dop_data is None:
            return None
        dop = np.transpose(dop_data, (1, 2, 0)) if dop_data.ndim == 3 else dop_data
    else:
        return None
    if dop is None:
        return None
    # Ensure (H, W, 3) — grayscale DOPs are broadcast to RGB
    if dop.ndim == 2:
        dop = np.stack([dop, dop, dop], axis=-1)
    elif dop.shape[2] == 1:
        dop = np.concatenate([dop, dop, dop], axis=-1)
    return dop



def compute_valid_mask(dop_tile, kpts_d: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """
    Compute a combined validity mask that excludes:
    - Matches on black/no-data DOP pixels (all channels zero)
    - Matches with invalid DSM values (NaN or unreasonable)

    Parameters
    ----------
    dop_tile : GeoTile with .data attribute (H, W, C) uint8
    kpts_d   : (N, 2) DOP-space keypoints (x, y)
    zs       : (N,) DSM Z-values at match locations

    Returns
    -------
    valid : (N,) bool mask"""
    dop_data = dop_tile.data  # (H, W, C) uint8
    dop_h, dop_w = dop_data.shape[:2]
    kd_int = np.round(kpts_d).astype(int)
    kd_int[:, 0] = np.clip(kd_int[:, 0], 0, dop_w - 1)
    kd_int[:, 1] = np.clip(kd_int[:, 1], 0, dop_h - 1)
    dop_pixels = dop_data[kd_int[:, 1], kd_int[:, 0]]  # (N, C)
    dop_valid = dop_pixels.sum(axis=1) > 0  # not all-black

    return ~np.isnan(zs) & (zs > -100) & dop_valid


def match_and_lift(
    image: np.ndarray,
    crop_center: Tuple[float, float],
    crop_size: float,
    geo_handler,
    matcher: BaseMatcher,
    num_matches: int = 5000,
    confidence_thresh: float = 0.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match the UAV image against a DOP crop, then lift to 3D using the
    **full-resolution DSM**.

    Returns
    -------
    pts_2d : (N, 2) float64 – pixel positions in the query image
    pts_3d : (N, 3) float64 – UTM (X, Y, Z) world positions
    confs  : (N,)   float32 – match confidences"""
    empty = (np.zeros((0, 2), dtype=np.float64),
             np.zeros((0, 3), dtype=np.float64),
             np.zeros(0, dtype=np.float32))

    cx, cy = crop_center
    dop_tile = geo_handler.crop_dop(cx, cy, crop_size)
    if dop_tile is None:
        return empty

    match_result = matcher.match(image, dop_tile.data, num_matches=num_matches)

    kpts_q = match_result.kpts_query
    kpts_d = match_result.kpts_dop
    conf = match_result.confidences

    if confidence_thresh > 0 and len(conf) > 0:
        mask = conf >= confidence_thresh
        kpts_q, kpts_d, conf = kpts_q[mask], kpts_d[mask], conf[mask]

    if len(kpts_q) < 10:
        return empty

    utm_xs, utm_ys = dop_tile.pixel_to_utm_batch(kpts_d[:, 0], kpts_d[:, 1])
    zs = sample_full_dsm_batch(geo_handler, utm_xs, utm_ys)

    valid = compute_valid_mask(dop_tile, kpts_d, zs)
    if not valid.any():
        return empty

    pts_2d_out = kpts_q[valid].astype(np.float64)
    pts_3d_out = np.column_stack([utm_xs[valid], utm_ys[valid], zs[valid]]).astype(np.float64)
    conf_out = conf[valid].astype(np.float32)

    if verbose:
        print(f"  match_and_lift: crop=({cx:.0f},{cy:.0f},{crop_size:.0f}m) "
              f"raw={len(kpts_q)} lift={len(pts_3d_out)}")

    return (pts_2d_out, pts_3d_out, conf_out)


def _mean_reproj_error(
    pts_2d: np.ndarray,
    pts_3d: np.ndarray,
    R_c2w: np.ndarray,
    position: np.ndarray,
    K: np.ndarray,
    idx: np.ndarray,
) -> float:
    """Mean reprojection error (px) of the inlier subset."""
    if len(idx) == 0:
        return float('inf')
    p3 = pts_3d[idx]
    p2 = pts_2d[idx]
    # R_c2w is camera-to-world so world-to-camera = R_c2w.T
    R_wc = R_c2w.T
    t_wc = -R_wc @ position
    rvec, _ = cv2.Rodrigues(R_wc)
    proj, _ = cv2.projectPoints(
        p3.reshape(-1, 1, 3).astype(np.float64),
        rvec, t_wc.reshape(3, 1),
        K, np.zeros(4))
    proj = proj.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - p2, axis=1)))



def localize_from_correspondences(
    pts_2d: np.ndarray,
    pts_3d: np.ndarray,
    image_size: Tuple[int, int],
    fov_vertical: float,
    reproj_threshold: float = 4.0,
    pnp_iterations: int = 5000,
    min_inliers: int = 30,
    verbose: bool = False,
    K: np.ndarray = None,
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, int, Optional[np.ndarray], np.ndarray]:
    """
    Run PnP-RANSAC on a set of 2D-3D correspondences.

    Parameters
    ----------
    K : (3,3) intrinsics matrix.  If provided, overrides the FOV-based
        computation (important for cameras with off-centre principal point).

    Returns
    -------
    position   : (3,) camera centre in UTM, or None
    pts_2d_inl : (M, 2) inlier 2D points
    pts_3d_inl : (M, 3) inlier 3D points
    num_inliers: int
    R_c2w      : (3, 3) camera-to-world rotation, or None
    inlier_idx : (M,) indices into the input arrays, or empty array"""
    fail = (None, np.zeros((0, 2)), np.zeros((0, 3)), 0, None, np.zeros(0, dtype=int))

    if len(pts_2d) < min_inliers:
        return fail

    if K is None:
        K = get_intrinsics(image_size, fov_vertical)

    centroid = pts_3d.mean(axis=0)
    pts_3d_c = pts_3d - centroid

    try:
        success, rvec, tvec, inliers = _safe_solvePnPRansac(
            pts_3d_c.reshape(-1, 1, 3),
            pts_2d.reshape(-1, 1, 2),
            K, None,
            iterationsCount=pnp_iterations,
            reprojectionError=reproj_threshold,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except PnPSolverError:
        # SQPNP fails on near-planar terrain (flat airports, etc.).
        # Fall back to ITERATIVE which handles degenerate z-variance.
        try:
            success, rvec, tvec, inliers = _safe_solvePnPRansac(
                pts_3d_c.reshape(-1, 1, 3),
                pts_2d.reshape(-1, 1, 2),
                K, None,
                iterationsCount=pnp_iterations,
                reprojectionError=reproj_threshold,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except PnPSolverError:
            return fail

    if not success or inliers is None or len(inliers) < min_inliers:
        return fail

    idx = inliers.flatten()

    R, _ = cv2.Rodrigues(rvec)
    position = -R.T @ tvec.flatten() + centroid

    # Sanity check: camera should be above the matched 3D points (not below ground)
    # and within a reasonable height above them (< 2 km).
    ground_z = np.median(pts_3d[idx, 2])
    height_above_ground = position[2] - ground_z
    if height_above_ground < 1 or height_above_ground > 2000:
        return fail

    if verbose:
        print(f"  PnP: {len(inliers)} inliers, "
              f"pos=({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})")

    return position, pts_2d[idx], pts_3d[idx], len(inliers), R.T, idx




def full_dop_roi_detection(
    image: np.ndarray,
    geo_handler,
    matcher: BaseMatcher,
    num_matches: int = 3000,
    fov_vertical: float = 0.0,
    verbose: bool = False,
) -> Optional[Dict]:
    """Match the UAV image against the full DOP to detect the ROI.

    Returns a dict with rich information about the detected region,
    including query-image coverage (what fraction of the query matched),
    the matched UTM bounding box, and estimated altitude.

    Returns
    -------
    dict with keys:
        utm_x, utm_y : float — ROI centre in UTM
        est_altitude  : float — estimated AGL from match spread
        query_coverage: float — fraction of query image area covered by matches (0–1)
        utm_bbox      : (min_x, min_y, max_x, max_y) — UTM bounding box of matches
        match_spread_m: float — larger axis of match spread in metres
        num_inliers   : int — number of RANSAC inliers
        kpts_query    : (N,2) — query image keypoints (inliers, full-res coords)
        kpts_dop      : (N,2) — DOP keypoints (inliers, full-res coords)
        confidences   : (N,) — match confidences
    Or None if matching failed."""
    full_dop = get_full_dop_image(geo_handler)
    if full_dop is None:
        if verbose:
            print("  Full-DOP ROI: full DOP not available, skipping")
        return None

    img_small, img_scale = downsample_image(image, max_size=1536)
    dop_small, dop_scale = downsample_image(full_dop, max_size=2560)

    match_result = matcher.match(img_small, dop_small, num_matches=num_matches)
    match_result.kpts_query *= img_scale
    match_result.kpts_dop *= dop_scale

    kpts_q = match_result.kpts_query
    kpts_d = match_result.kpts_dop
    conf = match_result.confidences

    if len(kpts_d) < 20:
        if verbose:
            print(f"  Full-DOP ROI: too few raw matches ({len(kpts_d)})")
        return None

    center_px = np.median(kpts_d, axis=0)
    utm_x, utm_y = geo_handler.pixel_to_utm(center_px[0], center_px[1])

    # ---- Query-image coverage ----
    # Fraction of query image area covered by matched keypoints.
    h_img, w_img = image.shape[:2]
    q_x_spread = (np.percentile(kpts_q[:, 0], 95) -
                  np.percentile(kpts_q[:, 0], 5))
    q_y_spread = (np.percentile(kpts_q[:, 1], 95) -
                  np.percentile(kpts_q[:, 1], 5))
    query_coverage = (q_x_spread * q_y_spread) / (w_img * h_img)
    query_coverage = float(np.clip(query_coverage, 0.0, 1.0))

    # ---- UTM bounding box of matches ----
    dop_gsd = getattr(geo_handler, 'dop_resolution',
                      getattr(geo_handler, 'resolution', 0.1))

    # Convert all inlier DOP keypoints to UTM for bounding box.
    # Use the DOP affine transform directly for speed.
    if hasattr(geo_handler, 'dop_transform'):
        t = geo_handler.dop_transform
        utm_xs = t.a * kpts_d[:, 0] + t.b * kpts_d[:, 1] + t.c
        utm_ys = t.d * kpts_d[:, 0] + t.e * kpts_d[:, 1] + t.f
    elif hasattr(geo_handler, 'pixel_to_utm_batch'):
        utm_xs, utm_ys = geo_handler.pixel_to_utm_batch(kpts_d[:, 0], kpts_d[:, 1])
    else:
        utm_xs = np.array([geo_handler.pixel_to_utm(kp[0], kp[1])[0] for kp in kpts_d])
        utm_ys = np.array([geo_handler.pixel_to_utm(kp[0], kp[1])[1] for kp in kpts_d])

    utm_bbox = (
        float(np.percentile(utm_xs, 2)),
        float(np.percentile(utm_ys, 2)),
        float(np.percentile(utm_xs, 98)),
        float(np.percentile(utm_ys, 98)),
    )

    # ---- Estimate altitude from match spread in DOP ----
    spread_y_px = (np.percentile(kpts_d[:, 1], 95) -
                   np.percentile(kpts_d[:, 1], 5))
    spread_x_px = (np.percentile(kpts_d[:, 0], 95) -
                   np.percentile(kpts_d[:, 0], 5))
    spread_px = max(spread_x_px, spread_y_px)
    spread_m = spread_px * dop_gsd

    est_altitude = 0.0
    if fov_vertical > 0 and len(kpts_d) >= 10:
        half_fov_rad = np.radians(fov_vertical / 2.0)
        if spread_m > 10.0 and half_fov_rad > 0.01:
            est_altitude = spread_m / (2.0 * np.tan(half_fov_rad))
            est_altitude = float(np.clip(est_altitude, 5.0, 2000.0))

    if verbose:
        print(f"  Full-DOP ROI: {len(kpts_d)} raw matches, "
              f"centre=({utm_x:.1f}, {utm_y:.1f}), "
              f"est_alt={est_altitude:.0f}m, "
              f"query_coverage={query_coverage:.1%}, "
              f"spread={spread_m:.0f}m")

    # ---- Build PnP-ready 3D correspondences ----
    # Sample DSM heights at UTM positions → full 3D points.
    zs = sample_full_dsm_batch(geo_handler, utm_xs, utm_ys)
    valid_3d = ~np.isnan(zs) & (zs > -100)
    pts_2d_pnp = kpts_q[valid_3d].astype(np.float64)
    pts_3d_pnp = np.column_stack([
        utm_xs[valid_3d], utm_ys[valid_3d], zs[valid_3d]
    ]).astype(np.float64)
    confs_pnp = conf[valid_3d].astype(np.float32)

    return {
        'utm_x': float(utm_x),
        'utm_y': float(utm_y),
        'est_altitude': float(est_altitude),
        'query_coverage': query_coverage,
        'utm_bbox': utm_bbox,
        'match_spread_m': float(spread_m),
        'num_inliers': len(kpts_d),
        'kpts_query': kpts_q,
        'kpts_dop': kpts_d,
        'confidences': conf,
        'pts_2d': pts_2d_pnp,
        'pts_3d': pts_3d_pnp,
        'confs': confs_pnp,
    }


def generate_dsm_correspondences(
    camera_pos: np.ndarray,
    R_c2w: np.ndarray,
    image_size: Tuple[int, int],
    fov_vertical: float,
    geo_handler,
    dsm_sample_region: Optional[float] = None,
    dsm_sample_step: float = 3.0,
    zbuffer_cell_size: int = 8,
    verbose: bool = False,
    K: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D-3D correspondences by projecting DSM points onto the image
    using the estimated camera pose, with z-buffer visibility filtering.

    For each image-space cell, only the point with the smallest depth (closest
    to camera) is retained, ensuring occluded surface points are discarded.

    Parameters
    ----------
    camera_pos : (3,) camera position in UTM (x, y, z)
    R_c2w      : (3, 3) camera-to-world rotation matrix
    image_size : (height, width) of the image
    fov_vertical : vertical FOV in degrees
    geo_handler: GeoTIFFHandler or SequenceGeoHandler for DSM sampling
    dsm_sample_region : region around camera to sample DSM (meters).
        If None, automatically derived from the camera frustum projection.
    dsm_sample_step   : grid step for DSM sampling (meters)
    zbuffer_cell_size : pixel cell size for z-buffer binning
    verbose : print debug info

    Returns
    -------
    pts_2d : (N, 2) float64 -- pixel positions in the image
    pts_3d : (N, 3) float64 -- UTM (X, Y, Z) world positions"""
    from orthotrack.crop_strategy import get_intrinsics, compute_visible_dop_crop
    from orthotrack.exceptions import VisibleCropError, InvalidGeometryError

    h, w = image_size
    if K is None:
        K = get_intrinsics(image_size, fov_vertical)

    # Auto-derive sample region from the visible frustum projection
    if dsm_sample_region is None:
        try:
            _, _, vis_extent = compute_visible_dop_crop(
                camera_pos, R_c2w, image_size,
                fov_vertical, geo_handler,
                verbose=False, K=K,
            )
            dsm_sample_region = vis_extent * 2.0  # generous margin
        except (VisibleCropError, InvalidGeometryError):
            from orthotrack.crop_strategy import _estimate_dsm_sample_region
            dsm_sample_region = _estimate_dsm_sample_region(
                camera_pos, R_c2w, image_size, K, geo_handler
            )
    # Safety cap: never allocate an absurdly large DSM tile (protects against
    # near-horizontal rays or corrupted tracked poses causing OOM).
    dsm_sample_region = min(float(dsm_sample_region), 8000.0)

    # Build W2C transform
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ camera_pos

    # Sample 3D points from DSM in a grid around camera position
    cam_x, cam_y = camera_pos[0], camera_pos[1]
    half_region = dsm_sample_region / 2
    xs = np.arange(cam_x - half_region, cam_x + half_region, dsm_sample_step)
    ys = np.arange(cam_y - half_region, cam_y + half_region, dsm_sample_step)

    xx, yy = np.meshgrid(xs, ys)
    xx_flat = xx.ravel()
    yy_flat = yy.ravel()

    # Get elevations (vectorized)
    zz_flat = sample_full_dsm_batch(geo_handler, xx_flat, yy_flat)

    # Filter valid elevations
    valid = ~np.isnan(zz_flat) & (zz_flat > -100)
    if valid.sum() < 20:
        if verbose:
            print(f"  DSM correspondences: insufficient DSM points ({valid.sum()})")
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    pts_3d_all = np.stack([xx_flat[valid], yy_flat[valid], zz_flat[valid]], axis=1)

    # Project to camera coordinates
    pts_cam = (R_w2c @ pts_3d_all.T).T + t_w2c  # (N, 3)

    # Filter points behind camera
    in_front = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[in_front]
    pts_3d_all = pts_3d_all[in_front]

    if len(pts_cam) < 20:
        if verbose:
            print(f"  DSM correspondences: too few points in front of camera ({len(pts_cam)})")
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    # Project to image coordinates
    pts_img = (K @ pts_cam.T).T
    pts_img_2d = pts_img[:, :2] / pts_img[:, 2:3]
    depths = pts_cam[:, 2]

    # Filter to points inside image bounds
    margin = 5
    in_bounds = ((pts_img_2d[:, 0] >= margin) & (pts_img_2d[:, 0] < w - margin) &
                 (pts_img_2d[:, 1] >= margin) & (pts_img_2d[:, 1] < h - margin))

    pts_img_2d = pts_img_2d[in_bounds]
    pts_3d_all = pts_3d_all[in_bounds]
    depths = depths[in_bounds]

    if len(pts_3d_all) < 20:
        if verbose:
            print(f"  DSM correspondences: too few points in image ({len(pts_3d_all)})")
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    # ---- Z-buffer visibility filtering ----
    # For each pixel cell, keep only the point with the smallest depth
    cell_cols = (pts_img_2d[:, 0] / zbuffer_cell_size).astype(int)
    cell_rows = (pts_img_2d[:, 1] / zbuffer_cell_size).astype(int)
    num_cell_cols = w // zbuffer_cell_size + 1
    cell_ids = cell_rows * num_cell_cols + cell_cols

    # For each cell, find the point with minimum depth
    unique_cells = np.unique(cell_ids)
    visible_indices = []
    for cell in unique_cells:
        cell_mask = cell_ids == cell
        cell_depths = depths[cell_mask]
        cell_indices = np.where(cell_mask)[0]
        # Keep only the closest point in each cell
        min_idx = cell_depths.argmin()
        visible_indices.append(cell_indices[min_idx])

    visible_indices = np.array(visible_indices)
    result_2d = pts_img_2d[visible_indices].astype(np.float64)
    result_3d = pts_3d_all[visible_indices].astype(np.float64)

    if verbose:
        print(f"  DSM correspondences: {len(result_2d)} visible points "
              f"(from {len(pts_img_2d)} projected, {len(pts_3d_all)} in-bounds)")

    return result_2d, result_3d


def estimate_pose_from_2d3d_corrspondences(
    pts_2d: np.ndarray,
    pts_3d: np.ndarray,
    image_size: Tuple[int, int],
    fov: float,
    verbose: bool = False,
    reproj_threshold: float = 5.0,
    K: np.ndarray = None,
    max_pnp_points: int = 0,
) -> Tuple[Optional[np.ndarray], int, float, Optional[np.ndarray]]:
    """
    Estimate pose from tracked 2D-3D correspondences.

    Returns (position, num_inliers, reproj_error, R_c2w)."""
    if len(pts_2d) < 10:
        return None, 0, float('inf'), None

    if K is None:
        K = get_intrinsics(image_size, fov)

    centroid = pts_3d.mean(axis=0)
    pts_3d_centered = pts_3d - centroid

    # Subsample for RANSAC only when explicitly requested (max_pnp_points > 0).
    # The default is 0 (no cap): subsampling different point subsets each frame
    # causes different RANSAC starting poses → different L-M inlier sets →
    # residual ~0.5 m frame-to-frame jitter even after L-M refinement.
    # Tracked frames typically have 1000–3000 points — well within RANSAC's
    # efficient operating range without any cap.
    n = len(pts_2d)
    if max_pnp_points > 0 and n > max_pnp_points:
        idx = np.random.choice(n, max_pnp_points, replace=False)
        sub_2d = pts_2d[idx]
        sub_3d = pts_3d_centered[idx]
    else:
        sub_2d = pts_2d
        sub_3d = pts_3d_centered

    try:
        success, rvec, tvec, inliers = _safe_solvePnPRansac(
            sub_3d.reshape(-1, 1, 3),
            sub_2d.reshape(-1, 1, 2),
            K, None,
            iterationsCount=500,
            reprojectionError=reproj_threshold,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except PnPSolverError:
        return None, 0, float('inf'), None

    if not success or inliers is None or len(inliers) < 10:
        return None, len(inliers) if inliers is not None else 0, float('inf'), None

    inlier_idx = inliers.flatten()
    pts_3d_inl = sub_3d[inlier_idx]
    pts_2d_inl = sub_2d[inlier_idx]

    # Match the original behavior: report reproj on RANSAC inliers only.
    # This value is used by the keyframe trigger and should represent
    # inlier quality rather than being diluted by outliers.
    projected, _ = cv2.projectPoints(
        pts_3d_inl.reshape(-1, 1, 3), rvec, tvec, K, None)
    projected = projected.reshape(-1, 2)
    reproj_error = float(np.mean(np.linalg.norm(pts_2d_inl - projected, axis=1)))

    R, _ = cv2.Rodrigues(rvec)
    camera_pos = -R.T @ tvec.flatten() + centroid

    return camera_pos, len(inlier_idx), reproj_error, R.T


def should_create_keyframe(
    num_tracked: int,
    frames_since_kf: int,
    keyframe_min_points: int,
    initial_num_pts: int = 0,
    pts_2d: np.ndarray = None,
    reproj_error: float = 0.0,
    baseline_reproj: float = 0.0,
    reproj_growth_factor: float = 2.0,
    reproj_abs_threshold: float = 2.0,
    grace_ramp_frames: int = 15,
    point_drop_ratio: float = 0.40,
    spatial_collapse_frac: float = None,
    spatial_collapse_px: float = 30.0,
    image_shape: tuple = None,
    growth_decay_frames: int = 100,
    min_growth_margin: float = 0.35,
    keyframe_max_interval: int = 0,
    min_kf_interval: int = 5,
) -> Tuple[bool, str]:
    """
    Determine if a new keyframe is needed.

    Returns (need_keyframe, reason).

    Conditions (paper Sec. 3, Eqs. 3-4):
    1. M_t < N_min  (absolute point count below minimum)
    2. M_t < alpha * M_k  (point-drop ratio: too many correspondences lost)
    3. min(sigma_x, sigma_y) < sigma_min  (spatial collapse)
    4a. Absolute reproj threshold (Eq. 3):
          e_abs(dt) = e_base + delta * max(0, 1 - dt/G)
    4b. Relative reproj threshold (Eq. 4, active for dt >= G):
          trigger when e_t > f_bar_t * e_k, where f_bar_t decays from
          f_0 to 1+m over D frames.
    5. Max interval: force a KF if frames_since_kf >= keyframe_max_interval
       (when keyframe_max_interval > 0)."""
    # (0) Max interval — force KF to bound LK drift (not blocked by min_kf_interval)
    if keyframe_max_interval and keyframe_max_interval > 0 and frames_since_kf >= keyframe_max_interval:
        return True, f"max_interval({frames_since_kf}>={keyframe_max_interval})"

    # Warmup: suppress all quality-triggered KFs for min_kf_interval frames after
    # the previous keyframe to avoid cascading re-localizations after a bad KF.
    if min_kf_interval > 0 and frames_since_kf < min_kf_interval:
        return False, ""

    # (1) Absolute point count
    if num_tracked < keyframe_min_points:
        return True, f"low_points({num_tracked}<{keyframe_min_points})"

    # (2) Point-drop ratio: M_t < alpha * M_k with an absolute-count guard.
    if initial_num_pts > 0 and point_drop_ratio > 0:
        drop_abs_floor = keyframe_min_points * 3
        if num_tracked < initial_num_pts * point_drop_ratio and num_tracked < drop_abs_floor:
            return True, (f"point_drop({num_tracked}/{initial_num_pts}="
                          f"{num_tracked/initial_num_pts:.0%}<{point_drop_ratio:.0%},"
                          f" abs<{drop_abs_floor})")

    # (3) Spatial collapse
    if pts_2d is not None and len(pts_2d) >= 10:
        std_x = np.std(pts_2d[:, 0])
        std_y = np.std(pts_2d[:, 1])
        if spatial_collapse_frac is not None and image_shape is not None:
            h, w = image_shape
            thresh_x = w * spatial_collapse_frac
            thresh_y = h * spatial_collapse_frac
        else:
            thresh_x = spatial_collapse_px
            thresh_y = spatial_collapse_px
        if std_x < thresh_x or std_y < thresh_y:
            return True, (f"spatial_collapse(std_x={std_x:.0f}<{thresh_x:.0f} "
                          f"or std_y={std_y:.0f}<{thresh_y:.0f})")

    # (4a) Absolute reproj threshold — Eq. 3:
    #   e_abs(dt) = e_base + delta * max(0, 1 - dt/G)
    grace_margin = 2.0  # delta in the paper
    if grace_ramp_frames > 0:
        proximity_grace = grace_margin * max(0.0, 1.0 - frames_since_kf / grace_ramp_frames)
    else:
        proximity_grace = 0.0
    effective_threshold = reproj_abs_threshold + proximity_grace
    if baseline_reproj > 0 and baseline_reproj > effective_threshold:
        effective_threshold = baseline_reproj + proximity_grace

    if reproj_abs_threshold > 0 and reproj_error > effective_threshold:
        return True, (f"reproj_abs({reproj_error:.1f}px>{effective_threshold:.1f}px, "
                      f"since_kf={frames_since_kf})")

    # (4b) Relative reproj threshold — Eq. 4 (active for dt >= G):
    #   f_bar_t = 1 + (f_0 - 1)(1 - rho) + m * rho
    #   rho = min(1, (dt - G) / D)
    if (baseline_reproj > 0 and reproj_growth_factor > 0
            and frames_since_kf >= grace_ramp_frames):
        extra_frames = frames_since_kf - grace_ramp_frames
        rho = min(1.0, extra_frames / growth_decay_frames)
        initial_margin = reproj_growth_factor - 1.0
        margin = initial_margin * (1.0 - rho) + min_growth_margin * rho
        effective_factor = 1.0 + margin
        if reproj_error > baseline_reproj * effective_factor:
            growth_pct = (reproj_error / baseline_reproj - 1) * 100
            return True, (f"reproj_growth({growth_pct:.0f}%, "
                          f"{baseline_reproj:.1f}->{reproj_error:.1f}px, "
                          f"factor={effective_factor:.2f})")

    return False, ""
