"""
View geometry and DOP crop computation strategies.

All functions are standalone (no pipeline state) so they can be tested
and reused independently."""

import numpy as np
from typing import List, Tuple, Optional

from utils.geo import bilinear_sample_dsm_batch
from orthotrack.exceptions import VisibleCropError, InvalidGeometryError
from utils.matching import compute_view_reliability
from utils.pose import CameraPose


def get_intrinsics(
    image_size: Tuple[int, int],
    fov_vertical: float,
    fx: float = None,
    fy: float = None,
    cx: float = None,
    cy: float = None,
) -> np.ndarray:
    """Compute camera intrinsics matrix from image size and vertical FOV.

    If fx/fy/cx/cy are provided, they override the FOV-based computation.
    This is important for cameras with off-center principal points (e.g.
    UAVScenes with cx≠w/2).

    Args:
        image_size: (height, width)
        fov_vertical: Vertical FOV in degrees.
        fx, fy: Optional focal lengths (pixels). If None, computed from FOV.
        cx, cy: Optional principal point (pixels). If None, uses image center.

    Returns:
        K: (3, 3) intrinsics matrix."""
    h, w = image_size
    _fy = fy if fy is not None else h / (2 * np.tan(np.radians(fov_vertical) / 2))
    _fx = fx if fx is not None else _fy
    _cx = cx if cx is not None else w / 2
    _cy = cy if cy is not None else h / 2
    return np.array([[_fx, 0, _cx], [0, _fy, _cy], [0, 0, 1]], dtype=np.float64)






def _estimate_dsm_sample_region(
    camera_pos: np.ndarray,
    R_c2w: np.ndarray,
    image_size: Tuple[int, int],
    K: np.ndarray,
    geo_handler,
    min_region: float = 100.0,
    max_region: float = 8000.0,
) -> float:
    """Estimate DSM sample region from camera frustum cast onto a ground plane.

    Casts the 4 image corner rays to a flat ground plane at the local DSM
    height below the camera.  Returns the side length (metres) of a square
    centred on the camera nadir that covers the visible ground footprint."""
    from orthotrack.localization import sample_full_dsm_batch

    # --- 1. estimate altitude above ground ---
    ground_z = sample_full_dsm_batch(
        geo_handler, np.array([camera_pos[0]]), np.array([camera_pos[1]])
    )
    if np.isnan(ground_z[0]) or ground_z[0] < -100:
        agl = 200.0          # fallback when DSM unavailable at nadir
    else:
        agl = max(float(camera_pos[2] - ground_z[0]), 10.0)

    # --- 2. cast 4 image corners as rays ---
    h, w = image_size
    corners = np.array([
        [0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1],
    ], dtype=np.float64)
    K_inv = np.linalg.inv(K)
    rays_cam = (K_inv @ corners.T)           # (3, 4) in camera frame
    rays_world = R_c2w @ rays_cam            # (3, 4) in world frame

    ground_level = float(camera_pos[2]) - agl           # ground Z
    max_dist = 0.0
    for i in range(4):
        rz = rays_world[2, i]
        if rz >= -1e-6:
            # ray doesn't point towards ground — conservative fallback
            max_dist = max(max_dist, agl * 10.0)
            continue
        t = (ground_level - float(camera_pos[2])) / rz
        gx = float(camera_pos[0]) + t * rays_world[0, i]
        gy = float(camera_pos[1]) + t * rays_world[1, i]
        d = np.hypot(gx - float(camera_pos[0]), gy - float(camera_pos[1]))
        max_dist = max(max_dist, d)

    # generous margin (2.5×) to account for terrain relief above/below the
    # flat-ground assumption.  Capped at max_region to avoid OOM when corner
    # rays are near-horizontal (e.g. after tracking drift).
    region = max(max_dist * 2.5, min_region)
    return float(min(region, max_region))


def compute_visible_dop_crop(camera_pos: np.ndarray, R_c2w: np.ndarray,
                              image_size: Tuple[int, int], fov_vertical: float,
                              geo_handler,
                              dsm_sample_region: Optional[float] = None,
                              dsm_sample_step: float = 5.0,
                              zbuffer_cell_size: int = 16,
                              verbose: bool = False,
                              K: np.ndarray = None) -> Tuple[float, float, float]:
    """
    Compute DOP crop parameters by projecting DSM points onto the image
    and finding the visible UTM footprint, with occlusion handling.

    Raises
    -------
    VisibleCropError
        If the footprint cannot be determined from DSM + pose + intrinsics."""
    from orthotrack.localization import sample_full_dsm_batch

    h, w = image_size
    if K is None:
        K = get_intrinsics(image_size, fov_vertical)

    # Build W2C transform from R_c2w
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ camera_pos

    # Auto-derive sample region from camera frustum if not provided
    if dsm_sample_region is None:
        dsm_sample_region = _estimate_dsm_sample_region(
            camera_pos, R_c2w, image_size, K, geo_handler
        )
        # Safety cap already applied inside _estimate_dsm_sample_region,
        # but guard here too in case the visible-crop path produced a huge value.
        dsm_sample_region = min(dsm_sample_region, 8000.0)
        if verbose:
            print(f"  Auto-derived dsm_sample_region={dsm_sample_region:.0f}m")

    # Sample 3D points from DSM in a grid around camera position
    cam_x, cam_y = camera_pos[0], camera_pos[1]
    half_region = dsm_sample_region / 2
    xs = np.arange(cam_x - half_region, cam_x + half_region, dsm_sample_step)
    ys = np.arange(cam_y - half_region, cam_y + half_region, dsm_sample_step)

    # Create meshgrid and flatten
    xx, yy = np.meshgrid(xs, ys)
    xx_flat = xx.ravel()
    yy_flat = yy.ravel()

    # Get elevations for all grid points (vectorized)
    zz_flat = sample_full_dsm_batch(geo_handler, xx_flat, yy_flat)

    # Filter valid elevations
    valid = ~np.isnan(zz_flat) & (zz_flat > -100)
    if valid.sum() < 20:
        raise VisibleCropError(
            f"insufficient DSM samples in region ({int(valid.sum())} valid, need >= 20)."
        )

    pts_3d = np.stack([xx_flat[valid], yy_flat[valid], zz_flat[valid]], axis=1)

    # Project to camera coordinates
    pts_cam = (R_w2c @ pts_3d.T).T + t_w2c  # (N, 3)

    # Filter points behind camera
    in_front = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[in_front]
    pts_3d = pts_3d[in_front]

    if len(pts_cam) < 20:
        raise VisibleCropError(
            f"too few DSM points in front of camera ({len(pts_cam)}, need >= 20)."
        )

    # Project to image coordinates
    pts_img = (K @ pts_cam.T).T
    pts_img_2d = pts_img[:, :2] / pts_img[:, 2:3]
    depths = pts_cam[:, 2]  # Depth in camera frame

    # Filter to points inside image bounds (with small margin)
    margin = 10
    in_bounds = ((pts_img_2d[:, 0] >= margin) & (pts_img_2d[:, 0] < w - margin) &
                 (pts_img_2d[:, 1] >= margin) & (pts_img_2d[:, 1] < h - margin))

    pts_img_2d = pts_img_2d[in_bounds]
    pts_3d = pts_3d[in_bounds]
    depths = depths[in_bounds]

    if len(pts_3d) < 5:
        raise VisibleCropError(
            f"too few DSM points project inside the image ({len(pts_3d)}, need >= 5)."
        )

    # ---- Z-buffer occlusion filtering ----
    cell_cols = (pts_img_2d[:, 0] / zbuffer_cell_size).astype(int)
    cell_rows = (pts_img_2d[:, 1] / zbuffer_cell_size).astype(int)

    cell_ids = cell_rows * (w // zbuffer_cell_size + 1) + cell_cols

    visible_mask = np.zeros(len(pts_3d), dtype=bool)
    unique_cells = np.unique(cell_ids)

    for cell in unique_cells:
        cell_mask = cell_ids == cell
        cell_depths = depths[cell_mask]
        min_depth = cell_depths.min()
        depth_tolerance = max(min_depth * 0.10, 2.0)
        close_enough = cell_depths <= min_depth + depth_tolerance
        cell_indices = np.where(cell_mask)[0]
        visible_mask[cell_indices[close_enough]] = True

    visible_pts_3d = pts_3d[visible_mask]
    visible_depths = depths[visible_mask]

    if len(visible_pts_3d) < 3:
        raise VisibleCropError(
            f"too few visible DSM points after occlusion filter ({len(visible_pts_3d)}, need >= 3)."
        )

    # Detect camera tilt: look direction = R_c2w @ [0,0,1], tilt from nadir
    look_dir = R_c2w @ np.array([0.0, 0.0, 1.0])
    cos_tilt = np.clip(-look_dir[2], -1.0, 1.0)
    tilt_from_nadir = np.degrees(np.arccos(cos_tilt))

    # Robust percentiles to exclude outlier points at the edges of the
    # visible region (works for both nadir and oblique views).
    lo_pct, hi_pct = 10.0, 90.0
    margin_factor = 1.15

    # Compute UTM bounding box of visible points
    min_x = np.percentile(visible_pts_3d[:, 0], lo_pct)
    max_x = np.percentile(visible_pts_3d[:, 0], hi_pct)
    min_y = np.percentile(visible_pts_3d[:, 1], lo_pct)
    max_y = np.percentile(visible_pts_3d[:, 1], hi_pct)

    extent_x = max_x - min_x
    extent_y = max_y - min_y

    # Inverse-depth-weighted center: closer ground points get higher weight.
    # For near-nadir views depths are similar so this reduces to a simple mean.
    inv_depth = 1.0 / np.clip(visible_depths, 1.0, None)
    weights = inv_depth / inv_depth.sum()
    center_x = float(np.sum(weights * visible_pts_3d[:, 0]))
    center_y = float(np.sum(weights * visible_pts_3d[:, 1]))

    # Crop size = max of extents with a small margin
    crop_size = max(extent_x, extent_y) * margin_factor

    if verbose:
        print(f"  Visible crop: {len(visible_pts_3d)} pts, "
              f"center=({center_x:.1f}, {center_y:.1f}), "
              f"extent=({extent_x:.1f}x{extent_y:.1f}m), crop={crop_size:.1f}m "
              f"(tilt={tilt_from_nadir:.0f}°)")

    return float(center_x), float(center_y), float(crop_size)
