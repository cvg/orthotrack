from pathlib import Path
from scipy.ndimage import binary_dilation, binary_closing

"""
utils/depth.py
==============
Depth map processing utilities for MovingDrone.

Key function:
    depth_to_normals(t_hit, fx, fy, cx, cy)

        Computes per-pixel surface normals from a ray-distance depth map.
        Validated against Open3D rendered ground-truth normals:
            Mean angular error: ~0.5°  |  Median: ~0.02°  |  99% of pixels < 5°
        (see tests/test_normals_from_depth.py for the full validation)

Coordinate conventions:
    - t_hit  : ray-distance depth (distance along the unit ray from the camera origin,
               NOT the orthographic Z component).
    - normals: stored in world space by Open3D (primitive_normals), pointing OUTWARD
               (away from surface, i.e. away from camera for top-down drone views).
               This function returns normals pointing TOWARD the camera in camera space,
               so negate and rotate if you need outward world-space normals."""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# NumPy implementation (used by dataset loader and tests)
# ---------------------------------------------------------------------------

def depth_to_normals_np(
    t_hit: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Compute per-pixel surface normals from a ray-distance depth map.

    MovingDrone stores *ray-distance* depth (t_hit from Open3D raycasting),
    not plain Z-depth. This function correctly:
      1. Converts ray-distance → 3-D points in camera frame.
      2. Computes normals via central-difference cross-products.

    Args:
        t_hit : (H, W) float32 ray-distance depth in metres.
                Pixels with no hit should be 0.
        fx, fy, cx, cy: Camera intrinsics (pixels).

    Returns:
        normals : (H, W, 3) float32 unit normals in CAMERA space,
                  pointing TOWARD the camera (+Z direction for frontal surfaces).
                  Invalid pixels (t_hit == 0 or border) → [0, 0, 0].

    Notes:
        Open3D primitive_normals are OUTWARD-facing. To match that convention,
        negate the returned array: normals_outward = -depth_to_normals_np(...)"""
    H, W = t_hit.shape

    # Pixel coordinate grids
    uu = np.arange(W, dtype=np.float64)
    vv = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(uu, vv)  # (H, W)

    # Normalised ray directions (unnormalised: [xn, yn, 1])
    xn = (uu - cx) / fx
    yn = (vv - cy) / fy
    ray_len = np.sqrt(xn**2 + yn**2 + 1.0)   # ||[xn, yn, 1]||

    # Convert ray-distance → Z-depth, then back-project to 3-D
    # Z = t_hit / ray_len   (cosine correction)
    Z = t_hit.astype(np.float64) / ray_len
    X = xn * Z
    Y = yn * Z

    pts = np.stack([X, Y, Z], axis=-1).astype(np.float32)   # (H, W, 3)
    pts[t_hit == 0] = 0.0

    # Central differences
    dPdu = np.zeros_like(pts)
    dPdv = np.zeros_like(pts)
    dPdu[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    dPdv[1:-1, :] = pts[2:, :] - pts[:-2, :]

    # Cross product → normals
    n = np.cross(dPdu, dPdv)                      # (H, W, 3)
    mag = np.linalg.norm(n, axis=-1, keepdims=True)

    valid = (mag[..., 0] > 1e-6) & (t_hit > 0)
    n_unit = np.where(valid[..., np.newaxis], n / (mag + 1e-9), 0.0)

    # Normals naturally point toward camera for top-down views.
    # Do NOT force a sign flip here — let the caller decide convention.

    return n_unit.astype(np.float32)


# ---------------------------------------------------------------------------
# PyTorch implementation (used inside DataLoader workers, GPU-ready)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Depth I/O  (uint16 quantized, self-describing NPZ)
# ---------------------------------------------------------------------------



def load_depth_npz(path) -> np.ndarray:
    """
    Load a depth NPZ and return a (H, W) float32 ray-distance array.

    Auto-detects format:
      - New format   : keys 'depth' (uint16) + 'depth_min' + 'depth_max'
      - Legacy format: key 'depth' (float32)

    Invalid / no-hit pixels are returned as 0."""
    d = np.load(path)
    raw = d['depth']

    if raw.dtype == np.uint16:
        # uint16 quantized format
        d_min = float(d['depth_min'])
        d_max = float(d['depth_max'])
        out = np.where(
            raw > 0,
            d_min + (raw.astype(np.float32) - 1.0) / 65534.0 * (d_max - d_min),
            0.0,
        ).astype(np.float32)
    else:
        # Legacy float32 format
        out = raw.astype(np.float32)

    return out


def load_depth_png(png_path, json_path) -> np.ndarray:
    """
    Load a 16-bit PNG depth map and its corresponding JSON metadata.
    
    Formula: Z = (PNG_value / 65535.0) * (depth_max - depth_min) + depth_min
    Zero pixels (0) represent invalid/no-hit depth and are preserved as 0.0."""
    import cv2, json
    raw = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Failed to read PNG image: {png_path}")
    
    with open(json_path, 'r') as f:
        meta = json.load(f)
    
    d_min = float(meta['depth_min_m'])
    d_max = float(meta['depth_max_m'])
    
    # Apply reconstruction formula. Preserve 0 as invalid.
    out = np.where(
        raw > 0,
        d_min + (raw.astype(np.float32) / 65535.0) * (d_max - d_min),
        0.0,
    ).astype(np.float32)
    
    return out

# ------------------------------------------------------------------ #
#  Depth evaluation metrics                                           #
# ------------------------------------------------------------------ #

def _safe_log(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.log(np.maximum(x, eps))


def compute_depth_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    mask=None,
    scale_correct: bool = False,
    min_depth: float = 0.1,
    max_depth: float = 1e6,
) -> dict:
    """Compute standard depth evaluation metrics (abs_rel, sq_rel, rmse, δ1/2/3).

    Args:
        gt:            (H, W) float32 ground-truth depth in metres. Zero = invalid.
        pred:          (H, W) float32 predicted depth (resized to gt if needed).
        mask:          Optional (H, W) bool additional validity mask.
        scale_correct: If True, align pred to gt via median scale.
        min_depth:     Minimum GT depth (metres) to include.
        max_depth:     Maximum GT depth (metres) to include.

    Returns:
        dict with abs_rel, sq_rel, rmse, rmse_log, delta_1/2/3, scale, n_pixels."""
    import cv2

    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

    valid = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt)
    if mask is not None:
        valid = valid & mask
    valid = valid & (pred > 0) & np.isfinite(pred)

    n_pixels = int(valid.sum())
    if n_pixels == 0:
        return {
            'abs_rel': np.nan, 'sq_rel': np.nan, 'rmse': np.nan, 'rmse_log': np.nan,
            'delta_1': np.nan, 'delta_2': np.nan, 'delta_3': np.nan,
            'scale': 1.0, 'n_pixels': 0,
        }

    gt_v   = gt[valid].astype(np.float64)
    pred_v = pred[valid].astype(np.float64)

    scale = 1.0
    if scale_correct:
        med_pred = np.median(pred_v)
        med_gt   = np.median(gt_v)
        if med_pred > 1e-6:
            scale = float(med_gt / med_pred)
            pred_v = pred_v * scale

    thresh   = np.maximum(gt_v / pred_v, pred_v / gt_v)
    abs_rel  = float(np.mean(np.abs(gt_v - pred_v) / gt_v))
    sq_rel   = float(np.mean((gt_v - pred_v) ** 2 / gt_v))
    rmse     = float(np.sqrt(np.mean((gt_v - pred_v) ** 2)))
    rmse_log = float(np.sqrt(np.mean((_safe_log(gt_v) - _safe_log(pred_v)) ** 2)))
    delta_1  = float(np.mean(thresh < 1.25))
    delta_2  = float(np.mean(thresh < 1.25 ** 2))
    delta_3  = float(np.mean(thresh < 1.25 ** 3))

    return {
        'abs_rel':  abs_rel,  'sq_rel':   sq_rel,
        'rmse':     rmse,     'rmse_log': rmse_log,
        'delta_1':  delta_1,  'delta_2':  delta_2,  'delta_3': delta_3,
        'scale':    scale,    'n_pixels': n_pixels,
    }


def raydist_to_zdepth(
    t_hit: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Convert ray-distance depth (MovingDrone t_hit) to Z-depth.

    MovingDrone stores depth as the distance along each ray from the camera
    origin (OpenGL raycasting). Most depth estimators output Z-depth instead.
    Converts: Z = t_hit / ||[xn, yn, 1]||."""
    H, W = t_hit.shape
    uu = np.arange(W, dtype=np.float64)
    vv = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(uu, vv)
    xn = (uu - cx) / fx
    yn = (vv - cy) / fy
    ray_len = np.sqrt(xn**2 + yn**2 + 1.0)
    z_depth = np.where(t_hit > 0, t_hit.astype(np.float64) / ray_len, 0.0)
    return z_depth.astype(np.float32)


def aggregate_depth_metrics(per_frame_metrics: list) -> dict:
    """Aggregate per-frame depth metric dicts into a sequence-level summary."""
    keys = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'delta_1', 'delta_2', 'delta_3']
    result = {}
    for k in keys:
        vals = [m[k] for m in per_frame_metrics if np.isfinite(m.get(k, np.nan))]
        result[k] = float(np.mean(vals)) if vals else np.nan
    result['n_frames'] = len(per_frame_metrics)
    result['n_pixels_total'] = int(sum(m.get('n_pixels', 0) for m in per_frame_metrics))
    return result
def compute_camera_space_normals(depth, intrinsics):
    """Compute surface normals in camera space from depth + intrinsics.
    
    This produces much cleaner normals than world-space point-map differentiation
    because camera-space coordinates are small and well-conditioned.
    
    Args:
        depth: (1, H, W) or (H, W) depth map (ray-cast distance or Z-depth)
        intrinsics: (3, 3) camera intrinsics matrix
        
    Returns:
        normals: (H, W, 3) numpy array, camera-space normals (normalized).
                 Zero for invalid pixels."""
    if torch.is_tensor(depth):
        d = depth.cpu().numpy()
    else:
        d = np.asarray(depth)
    if d.ndim == 3:
        d = d[0]  # (1, H, W) -> (H, W)
    d = d.astype(np.float64)

    if torch.is_tensor(intrinsics):
        K = intrinsics.cpu().numpy()
    else:
        K = np.asarray(intrinsics)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    H, W = d.shape
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    xn = (uu - cx) / fx
    yn = (vv - cy) / fy

    # Convert ray-length depth to Z-depth, then to camera-space XYZ
    rl = np.sqrt(xn**2 + yn**2 + 1.0)
    Z = (d / rl).astype(np.float32)
    X = (xn * Z).astype(np.float32)
    Y = (yn * Z).astype(np.float32)
    pts = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)
    pts[d == 0] = 0

    # Central differences
    dx = np.zeros_like(pts)
    dy = np.zeros_like(pts)
    dx[1:-1, 1:-1] = pts[1:-1, 2:] - pts[1:-1, :-2]
    dy[1:-1, 1:-1] = pts[2:, 1:-1] - pts[:-2, 1:-1]

    n = np.cross(dx, dy)  # (H, W, 3)
    nl = np.linalg.norm(n, axis=-1, keepdims=True)
    nl[nl < 1e-8] = 1
    n /= nl

    # Flip normals to face camera (positive Z in camera frame)
    n[n[..., 2] < 0] *= -1

    # Zero out invalid pixels
    n[d == 0] = 0
    return n.astype(np.float32)

def compute_surface_normals_from_pointmap(point_map):
    """Compute surface normals from a point map using finite differences.
    
    Args:
        point_map: (3, H, W) tensor with X, Y, Z world coordinates
        
    Returns:
        normals: (3, H, W) numpy array with surface normals (normalized)
        horizontal_mask: (H, W) boolean array - True for horizontal surfaces (roofs, roads)"""
    if torch.is_tensor(point_map):
        pm = point_map.cpu().numpy()
    else:
        pm = point_map
    
    # Compute gradients in x and y image directions
    # Use central differences for smoother normals
    dx = np.zeros_like(pm)
    dy = np.zeros_like(pm)
    
    # dX/du, dY/du, dZ/du (gradient in image u direction)
    dx[:, :, 1:-1] = (pm[:, :, 2:] - pm[:, :, :-2]) / 2
    dx[:, :, 0] = pm[:, :, 1] - pm[:, :, 0]
    dx[:, :, -1] = pm[:, :, -1] - pm[:, :, -2]
    
    # dX/dv, dY/dv, dZ/dv (gradient in image v direction)
    dy[:, 1:-1, :] = (pm[:, 2:, :] - pm[:, :-2, :]) / 2
    dy[:, 0, :] = pm[:, 1, :] - pm[:, 0, :]
    dy[:, -1, :] = pm[:, -1, :] - pm[:, -2, :]
    
    # Cross product to get normal: dx x dy
    # normal = (dy_Z * dx_Y - dy_Y * dx_Z, dx_Z * dy_X - dx_X * dy_Z, dx_X * dy_Y - dy_X * dx_Y)
    normals = np.zeros_like(pm)
    normals[0] = dy[2] * dx[1] - dy[1] * dx[2]  # nx
    normals[1] = dx[2] * dy[0] - dx[0] * dy[2]  # ny
    normals[2] = dx[0] * dy[1] - dy[0] * dx[1]  # nz
    
    # Normalize
    norm = np.sqrt(np.sum(normals**2, axis=0, keepdims=True)) + 1e-8
    normals = normals / norm
    
    # Ensure normals point "up" (positive Z in world coordinates)
    # Flip if Z component is negative
    flip_mask = normals[2] < 0
    normals[:, flip_mask] = -normals[:, flip_mask]
    
    # Detect depth discontinuities (edges between surfaces at different depths)
    # These cause unreliable normals due to large finite difference jumps
    z_channel = pm[2]  # Z coordinates
    depth_grad_x = np.abs(np.diff(z_channel, axis=1, prepend=z_channel[:, :1]))
    depth_grad_y = np.abs(np.diff(z_channel, axis=0, prepend=z_channel[:1, :]))
    depth_discontinuity = (depth_grad_x > 2.0) | (depth_grad_y > 2.0)  # >2m jump = edge
    
    # Dilate depth discontinuities to exclude nearby pixels
    depth_discontinuity = binary_dilation(depth_discontinuity, iterations=1)
    
    # Horizontal surfaces have normal pointing mostly up (large Z component)
    # Threshold: |nz| > 0.75 corresponds to surfaces tilted less than ~41 degrees from horizontal
    horizontal_mask = np.abs(normals[2]) > 0.75
    
    # Exclude depth discontinuities from horizontal classification (mark as vertical/unknown)
    horizontal_mask = horizontal_mask & ~depth_discontinuity
    
    # Apply morphological closing to fill small gaps in horizontal regions
    horizontal_mask = binary_closing(horizontal_mask, iterations=2)
    
    return normals, horizontal_mask

def load_depth(depth_path: str, data_dir: Path, sequence: str) -> np.ndarray:
    """Load depth map from npz file."""
    full_path = data_dir / sequence / depth_path
    if not full_path.exists():
        return None
    data = np.load(full_path)
    for key in ('depth', 'arr_0'):
        if key in data:
            return data[key].astype(np.float32)
    return data[list(data.keys())[0]].astype(np.float32)

def load_depth_npz(depth_npz_path: Path) -> np.ndarray:
    """Load (H, W) float32 ray-distance depth from .npz file."""
    data = np.load(depth_npz_path)
    for key in ('depth', 'arr_0'):
        if key in data:
            return data[key].astype(np.float32)
    return data[list(data.keys())[0]].astype(np.float32)

