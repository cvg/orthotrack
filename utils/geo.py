import torch
import zipfile
import shutil
from typing import Dict, List
from scipy.ndimage import binary_closing, binary_dilation
from utils.depth import compute_surface_normals_from_pointmap
# Assuming GeodataManager was imported in create_movingdrone.py


"""
Geospatial utilities for OrthoTrack.
Handles coordinate transformations, GeoTIFF loading, and cropping."""

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
import pyproj
from typing import Tuple, Optional, Union
from dataclasses import dataclass
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # allow large DOP/DSM GeoTIFFs
from pathlib import Path
import json
import xml.etree.ElementTree as ET

from functools import lru_cache

@lru_cache(maxsize=16)
def get_coordinate_grids(H: int, W: int, device: str = 'cpu'):
    """Return cached coordinate grids for a given size."""
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    return v.astype(np.float32), u.astype(np.float32)


def degrade_dop(rgb: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Simulate a coarser-GSD DOP at the same sampling grid.

    scale < 1.0  -> effective GSD multiplied by 1/scale (e.g. scale=0.1 with
                    a 0.2 m baseline yields a 2.0 m effective GSD). Performs
                    INTER_AREA downsample then INTER_LINEAR upsample, per
                    channel, preserving the original (H, W, C) shape and the
                    pixel-to-UTM affine transform."""
    if scale is None or scale >= 1.0 - 1e-6:
        return rgb
    H, W = rgb.shape[:2]
    small_W = max(1, int(round(W * scale)))
    small_H = max(1, int(round(H * scale)))
    coarse = cv2.resize(rgb, (small_W, small_H), interpolation=cv2.INTER_AREA)
    out = cv2.resize(coarse, (W, H), interpolation=cv2.INTER_LINEAR)
    if rgb.dtype == np.uint8 and out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out.astype(rgb.dtype, copy=False)


def degrade_dsm(height: np.ndarray, scale: float = 1.0, sigma_z: float = 0.0,
                seed: int = 0) -> np.ndarray:
    """Simulate a coarser/noisier DSM as produced by real LiDAR-binning pipelines.

    scale < 1.0  -> effective GSD multiplied by 1/scale (e.g. scale=0.1 with
                    a 0.2 m baseline yields a 2.0 m effective GSD). Spatial
                    coarsening uses block ``max-pool`` (matching how national
                    LiDAR DSMs aggregate point clouds: the cell value is the
                    maximum elevation of all points falling inside it, which
                    preserves rooftops). Up-sampling back to the original grid
                    uses bilinear, matching standard consumer interpolation.
    sigma_z > 0  -> Gaussian vertical noise (metres) added per pixel after
                    resampling, with a fixed seed for reproducibility.
    The returned array has the same shape and dtype as ``height``."""
    if (scale is None or scale >= 1.0 - 1e-6) and (not sigma_z or sigma_z <= 0.0):
        return height
    H, W = height.shape[:2]
    out = height.astype(np.float32, copy=True)
    if scale is not None and scale < 1.0 - 1e-6:
        # Block max-pool (LiDAR-DSM aggregation). Choose the coarse grid as the
        # nearest integer block factor so the operation is exact (no border
        # ambiguity). For scales 0.5/0.2/0.1/0.05/0.02 these factors are
        # 2/5/10/20/50 respectively.
        factor = max(2, int(round(1.0 / scale)))
        new_H = max(1, H // factor)
        new_W = max(1, W // factor)
        cropH = new_H * factor
        cropW = new_W * factor
        block = out[:cropH, :cropW].reshape(new_H, factor, new_W, factor)
        coarse = block.max(axis=(1, 3))
        out = cv2.resize(coarse, (W, H), interpolation=cv2.INTER_LINEAR)
    if sigma_z and sigma_z > 0.0:
        rng = np.random.default_rng(int(seed))
        noise = rng.normal(loc=0.0, scale=float(sigma_z), size=out.shape).astype(np.float32)
        out = out + noise
    return out.astype(height.dtype, copy=False)

def get_visible_footprint(pose_w2c: np.ndarray, K: np.ndarray, height: int, width: int, 
                         plane_z: float = 0.0, max_dist: float = 1000.0) -> Tuple[float, float, float, float]:
    """
    Calculate the bounding box of the camera's visible footprint on the ground plane.
    
    Args:
        pose_w2c: (4, 4) World-to-Camera matrix
        K: (3, 3) Intrinsic matrix
        height: Image height
        width: Image width
        plane_z: Z-coordinate of the ground plane
        max_dist: Maximum distance for horizon rays
        
    Returns:
        (min_x, min_y, max_x, max_y) of the footprint"""
    # Inverse pose to get Camera-to-World
    c2w = np.linalg.inv(pose_w2c)
    R_ws = c2w[:3, :3]
    t_ws = c2w[:3, 3]
    
    # 4 corners of the image + center
    # (u, v)
    corners_px = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height],
        [width/2, height/2]
    ])
    
    # Pixel to Camera rays
    # K_inv * [u, v, 1]
    K_inv = np.linalg.inv(K)
    corners_hom = np.concatenate([corners_px, np.ones((5, 1))], axis=1) # (5, 3)
    rays_cam = (K_inv @ corners_hom.T).T # (5, 3)
    
    # Transform rays to World
    # ray_world = R_ws * ray_cam
    rays_world = (R_ws @ rays_cam.T).T # (5, 3)
    
    # Normalize rays
    rays_world = rays_world / np.linalg.norm(rays_world, axis=1, keepdims=True)
    
    # Ray-Plane Intersection
    # Plane: z = plane_z
    # ray(t) = O + t * D
    # O_z + t * D_z = plane_z => t = (plane_z - O_z) / D_z
    
    O = t_ws
    ground_points = []
    
    for D in rays_world:
        if abs(D[2]) < 1e-6: # Parallel to plane
            continue
            
        t = (plane_z - O[2]) / D[2]
        
        if t < 0: # Intersection behind camera
            continue
        
        # If t is too large (looking at horizon), clamp it
        if t > max_dist:
             t = max_dist
             
        P = O + t * D
        ground_points.append(P[:2]) # (x, y)
        
    if not ground_points:
        # Fallback: simple radius around camera center
        return O[0]-100, O[1]-100, O[0]+100, O[1]+100
        
    ground_points = np.array(ground_points)
    min_x, min_y = ground_points.min(axis=0)
    max_x, max_y = ground_points.max(axis=0)
    
    return min_x, min_y, max_x, max_y


def bilinear_sample_dsm(dsm_data: np.ndarray, col: float, row: float) -> float:
    """Bilinear interpolation on DSM raster.  Returns np.nan when out of bounds."""
    h, w = dsm_data.shape[:2]
    c0, r0 = int(np.floor(col)), int(np.floor(row))
    c1, r1 = c0 + 1, r0 + 1
    if not (0 <= c0 and c1 < w and 0 <= r0 and r1 < h):
        # fall back to nearest-neighbour at boundary
        ci, ri = int(round(col)), int(round(row))
        if 0 <= ci < w and 0 <= ri < h:
            return float(dsm_data[ri, ci])
        return np.nan
    dx, dy = col - c0, row - r0
    z00 = float(dsm_data[r0, c0])
    z10 = float(dsm_data[r0, c1])
    z01 = float(dsm_data[r1, c0])
    z11 = float(dsm_data[r1, c1])
    if np.isnan(z00) or np.isnan(z10) or np.isnan(z01) or np.isnan(z11):
        return float(dsm_data[int(round(row)), int(round(col))])
    return z00 * (1 - dx) * (1 - dy) + z10 * dx * (1 - dy) + z01 * (1 - dx) * dy + z11 * dx * dy


def bilinear_sample_dsm_batch(dsm_data: np.ndarray, cols: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Vectorized bilinear interpolation on DSM raster for arrays of coordinates.

    Args:
        dsm_data: (H, W) DSM elevation array.
        cols: (N,) float array of column (x-pixel) coordinates.
        rows: (N,) float array of row (y-pixel) coordinates.

    Returns:
        (N,) float64 array of interpolated elevations. NaN where out of bounds."""
    h, w = dsm_data.shape[:2]
    cols = np.asarray(cols, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    N = len(cols)
    result = np.full(N, np.nan, dtype=np.float64)

    c0 = np.floor(cols).astype(np.int64)
    r0 = np.floor(rows).astype(np.int64)
    c1 = c0 + 1
    r1 = r0 + 1

    # Mask for full bilinear (all 4 neighbours in bounds)
    bilinear_ok = (c0 >= 0) & (c1 < w) & (r0 >= 0) & (r1 < h)

    if bilinear_ok.any():
        idx = np.where(bilinear_ok)[0]
        bc0 = c0[idx]; br0 = r0[idx]; bc1 = c1[idx]; br1 = r1[idx]
        dx = cols[idx] - bc0
        dy = rows[idx] - br0

        z00 = dsm_data[br0, bc0].astype(np.float64)
        z10 = dsm_data[br0, bc1].astype(np.float64)
        z01 = dsm_data[br1, bc0].astype(np.float64)
        z11 = dsm_data[br1, bc1].astype(np.float64)

        # Where any corner is NaN, fall back to nearest
        any_nan = np.isnan(z00) | np.isnan(z10) | np.isnan(z01) | np.isnan(z11)
        good = ~any_nan

        vals = z00 * (1 - dx) * (1 - dy) + z10 * dx * (1 - dy) + z01 * (1 - dx) * dy + z11 * dx * dy
        result[idx[good]] = vals[good]

        # Nearest-neighbour fallback for NaN corners
        if any_nan.any():
            nn_idx = idx[any_nan]
            ri_nn = np.round(rows[nn_idx]).astype(np.int64)
            ci_nn = np.round(cols[nn_idx]).astype(np.int64)
            in_b = (ci_nn >= 0) & (ci_nn < w) & (ri_nn >= 0) & (ri_nn < h)
            if in_b.any():
                result[nn_idx[in_b]] = dsm_data[ri_nn[in_b], ci_nn[in_b]].astype(np.float64)

    # Nearest-neighbour fallback for boundary points not covered by bilinear
    need_nn = np.isnan(result) & ~bilinear_ok
    if need_nn.any():
        nn_idx = np.where(need_nn)[0]
        ci = np.round(cols[nn_idx]).astype(np.int64)
        ri = np.round(rows[nn_idx]).astype(np.int64)
        in_b = (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h)
        if in_b.any():
            result[nn_idx[in_b]] = dsm_data[ri[in_b], ci[in_b]].astype(np.float64)

    return result


def unproject_depth_to_world(depth: np.ndarray, K: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """
    Unproject a depth map to 3D world coordinates.

    Args:
        depth: (H, W) depth map (camera Z values). Pixels with depth <= 0 are invalid.
        K: (3, 3) camera intrinsic matrix (must match the depth map resolution).
        w2c: (4, 4) world-to-camera matrix.

    Returns:
        (3, H, W) world XYZ point map. Zeros where depth is invalid."""
    H, W = depth.shape
    v_coords, u_coords = get_coordinate_grids(H, W)

    X_cam = (u_coords - K[0, 2]) * depth / K[0, 0]
    Y_cam = (v_coords - K[1, 2]) * depth / K[1, 1]
    Z_cam = depth

    pts_cam = np.stack([X_cam, Y_cam, Z_cam, np.ones_like(Z_cam)], axis=0)  # (4, H, W)
    c2w = np.linalg.inv(w2c)
    pts_world = np.einsum('ij,jhw->ihw', c2w, pts_cam)  # (4, H, W)

    result = pts_world[:3].astype(np.float32)
    valid = depth > 0
    result[:, ~valid] = 0
    return result


def dsm_to_xyz(dsm_height: np.ndarray, bounds: Tuple[float, float, float, float],
               target_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert a DSM height raster to a (3, H, W) XYZ map in world coordinates.

    Cleans nodata values (< -1000) and resizes to target_size via bilinear interpolation,
    then builds X/Y grids from the geospatial bounds.

    Args:
        dsm_height: (H, W) or (1, H, W) raw DSM height array.
        bounds: (left, bottom, right, top) in world coordinates (e.g. UTM).
        target_size: (H_out, W_out) desired output spatial size.

    Returns:
        (3, H_out, W_out) float32 array with channels [X, Y, Z]."""
    if dsm_height.ndim == 3:
        dsm_height = dsm_height[0]

    # Clean nodata
    dsm_clean = dsm_height.copy()
    dsm_clean[dsm_clean < -1000] = 0

    # Resize height map (skip if already at target size)
    h_out, w_out = target_size
    if dsm_clean.shape[0] == h_out and dsm_clean.shape[1] == w_out:
        dsm_resized = dsm_clean.astype(np.float32)
    else:
        # Use cv2.resize - much faster than PIL for float arrays
        dsm_resized = cv2.resize(dsm_clean.astype(np.float32), (w_out, h_out), interpolation=cv2.INTER_LINEAR)

    # Build coordinate grids
    l, b, r, t = bounds
    xx = np.linspace(l, r, w_out, dtype=np.float32)
    yy = np.linspace(t, b, h_out, dtype=np.float32)  # top→bottom (row 0 = top Y)
    xv, yv = np.meshgrid(xx, yy)

    return np.stack([xv, yv, dsm_resized], axis=0)




@dataclass
class GeoTile:
    """Represents a georeferenced tile from a GeoTIFF."""
    data: np.ndarray  # Image data (H, W, C) or (H, W)
    transform: rasterio.Affine  # Affine transform for this tile
    bounds: Tuple[float, float, float, float]  # (left, bottom, right, top) in CRS coordinates
    crs: str  # Coordinate reference system
    
    @property
    def width(self) -> int:
        return self.data.shape[1]
    
    @property
    def height(self) -> int:
        return self.data.shape[0]
    
    def utm_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        """Convert UTM coordinates to pixel coordinates in this tile."""
        col, row = ~self.transform * (x, y)
        return col, row

    def pixel_to_utm(self, col: float, row: float) -> Tuple[float, float]:
        """Convert pixel coordinates to UTM coordinates."""
        x, y = self.transform * (col, row)
        return x, y

    def pixel_to_utm_batch(self, cols: np.ndarray, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized pixel-to-UTM conversion via affine coefficients."""
        a, b, c, d, e, f = self.transform.a, self.transform.b, self.transform.c, \
                            self.transform.d, self.transform.e, self.transform.f
        xs = a * cols + b * rows + c
        ys = d * cols + e * rows + f
        return xs, ys

    def utm_to_pixel_batch(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized UTM-to-pixel conversion via inverse affine coefficients."""
        inv = ~self.transform
        a, b, c, d, e, f = inv.a, inv.b, inv.c, inv.d, inv.e, inv.f
        cols = a * xs + b * ys + c
        rows = d * xs + e * ys + f
        return cols, rows



class GeoTIFFHandler:
    """Handles loading and cropping of georeferenced TIFF files (DOP and DSM)."""
    
    def __init__(self, dop_path: str = None, dsm_path: str = None):
        self.dop_path = dop_path
        self.dsm_path = dsm_path
        
        self.dop_crs = None
        self.dsm_crs = None
        self._dop_src = None
        self._dsm_src = None
        self._dop_data = None  # Preloaded full tile: (bands, H, W) uint8
        self._dsm_data = None  # Preloaded full tile: (H, W) float32
        
        # Load metadata
        if dop_path:
            with rasterio.open(dop_path) as src:
                self.dop_crs = src.crs
                self.dop_transform = src.transform
                self.dop_bounds = src.bounds
                self.dop_width = src.width
                self.dop_height = src.height
                self.dop_resolution = src.res[0]  # Assume square pixels
            
        if dsm_path:
            with rasterio.open(dsm_path) as src:
                self.dsm_crs = src.crs
                self.dsm_transform = src.transform
                self.dsm_bounds = src.bounds
                self.dsm_width = src.width
                self.dsm_height = src.height
                self.dsm_resolution = src.res[0]
            
        # Transformer for lat/lon to UTM
        # Fallback to DSM CRS if DOP CRS is not available
        target_crs = self.dop_crs if self.dop_crs else self.dsm_crs
        if target_crs:
            self.transformer_wgs84_to_utm = pyproj.Transformer.from_crs(
                'EPSG:4326', str(target_crs), always_xy=True
            )
            self.transformer_utm_to_wgs84 = pyproj.Transformer.from_crs(
                str(target_crs), 'EPSG:4326', always_xy=True
            )
        else:
            self.transformer_wgs84_to_utm = None
            self.transformer_utm_to_wgs84 = None

    def preload(self, is_dsm=False):
        """Read the full tile into memory for fast numpy-based cropping."""
        if is_dsm and self._dsm_data is None and self.dsm_path:
            with rasterio.open(self.dsm_path) as src:
                self._dsm_data = src.read(1)  # (H, W) float
        elif not is_dsm and self._dop_data is None and self.dop_path:
            with rasterio.open(self.dop_path) as src:
                data = src.read()  # (bands, H, W)
                # Ensure DOP data is uint8 (some datasets store as float32)
                if data.dtype != np.uint8:
                    data = np.clip(data, 0, 255).astype(np.uint8)
                # Ensure 3-band (H, W, 3) compatible storage — expand grayscale
                if data.shape[0] == 1:
                    data = np.repeat(data, 3, axis=0)
                elif data.shape[0] > 3:
                    data = data[:3]
                self._dop_data = data

    def _get_src(self, is_dsm=False):
        """Lazy load and cache rasterio source."""
        if is_dsm:
            if self._dsm_src is None and self.dsm_path:
                self._dsm_src = rasterio.open(self.dsm_path)
            return self._dsm_src
        else:
            if self._dop_src is None and self.dop_path:
                self._dop_src = rasterio.open(self.dop_path)
            return self._dop_src

    def close(self):
        """Close opened sources."""
        if self._dop_src:
            self._dop_src.close()
            self._dop_src = None
        if self._dsm_src:
            self._dsm_src.close()
            self._dsm_src = None

    def __del__(self):
        self.close()

    
    def latlon_to_utm(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert WGS84 lat/lon to UTM coordinates."""
        x, y = self.transformer_wgs84_to_utm.transform(lon, lat)
        return x, y
    
    def utm_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """Convert UTM to WGS84 lat/lon."""
        lon, lat = self.transformer_utm_to_wgs84.transform(x, y)
        return lat, lon
    
    def utm_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert UTM coordinates to pixel coordinates in the full DOP."""
        col, row = ~self.dop_transform * (x, y)
        return int(col), int(row)
    
    def pixel_to_utm(self, col: int, row: int) -> Tuple[float, float]:
        """Convert pixel coordinates to UTM coordinates."""
        x, y = self.dop_transform * (col, row)
        return x, y
    
    def get_resolution(self) -> float:
        """Get the resolution of the DOP in meters per pixel."""
        return self.dop_resolution
    
    def crop_dop(self, center_x: float, center_y: float, 
                 crop_size_meters: float) -> GeoTile:
        """
        Crop DOP around a center point.
        
        Args:
            center_x, center_y: Center point in UTM coordinates
            crop_size_meters: Size of the crop in meters (creates a square)
            
        Returns:
            GeoTile with the cropped DOP data, or None if crop is invalid"""
        half_size = crop_size_meters / 2
        
        # Calculate bounds
        left = center_x - half_size
        right = center_x + half_size
        bottom = center_y - half_size
        top = center_y + half_size
        
        # Convert to pixel coordinates
        col_start, row_start = self.utm_to_pixel(left, top)
        col_end, row_end = self.utm_to_pixel(right, bottom)
        
        # Ensure bounds are within the image
        col_start = max(0, col_start)
        row_start = max(0, row_start)
        col_end = min(self.dop_width, col_end)
        row_end = min(self.dop_height, row_end)
        
        width = col_end - col_start
        height = row_end - row_start
        
        # Check for valid crop dimensions
        if width <= 10 or height <= 10:
            # Crop is too small or outside bounds
            return None
        
        # Read the cropped region
        if self._dop_data is not None:
            # Fast path: numpy slice from preloaded data
            data = self._dop_data[:3, row_start:row_end, col_start:col_end]
            if data.dtype != np.uint8:
                data = np.clip(data, 0, 255).astype(np.uint8)
            data = np.transpose(data, (1, 2, 0))
        else:
            src = self._get_src(is_dsm=False)
            if src is None: return None
            
            window = Window(col_start, row_start, width, height)
            data = src.read(window=window)
            # Ensure DOP data is uint8 (some datasets store as float32)
            if data.dtype != np.uint8:
                data = np.clip(data, 0, 255).astype(np.uint8)
            # Expand grayscale to 3-band before transposing
            if data.shape[0] == 1:
                data = np.repeat(data, 3, axis=0)
            elif data.shape[0] > 3:
                data = data[:3]
            # Transpose to HWC format
            data = np.transpose(data, (1, 2, 0))
        
        # Calculate transform for the cropped region
        crop_transform = self.dop_transform * rasterio.Affine.translation(col_start, row_start)
        
        # Actual bounds of the cropped region
        actual_left, actual_top = self.pixel_to_utm(col_start, row_start)
        actual_right, actual_bottom = self.pixel_to_utm(col_end, row_end)
        
        return GeoTile(
            data=data,
            transform=crop_transform,
            bounds=(actual_left, actual_bottom, actual_right, actual_top),
            crs=str(self.dop_crs)
        )

    def crop_dsm(self, center_x: float, center_y: float, 
                 crop_size_meters: float) -> GeoTile:
        """
        Crop DSM around a center point.
        
        Returns:
            GeoTile with the cropped DSM data, or None if crop is invalid"""
        half_size = crop_size_meters / 2
        
        left = center_x - half_size
        right = center_x + half_size
        bottom = center_y - half_size
        top = center_y + half_size
        
        # Convert to pixel coordinates
        col_start, row_start = ~self.dsm_transform * (left, top)
        col_end, row_end = ~self.dsm_transform * (right, bottom)
        
        col_start, row_start = int(col_start), int(row_start)
        col_end, row_end = int(col_end), int(row_end)
        
        # Ensure bounds are within the image
        col_start = max(0, col_start)
        row_start = max(0, row_start)
        col_end = min(self.dsm_width, col_end)
        row_end = min(self.dsm_height, row_end)
        
        width = col_end - col_start
        height = row_end - row_start
        
        # Check for valid dimensions
        if width <= 10 or height <= 10:
            return None
        
        with rasterio.open(self.dsm_path) as src:
            window = Window(col_start, row_start, width, height)
            data = src.read(1, window=window)  # Single band
            crop_transform = rasterio.windows.transform(window, src.transform)
        
        # Actual bounds
        actual_left, actual_top = self.dsm_transform * (col_start, row_start)
        actual_right, actual_bottom = self.dsm_transform * (col_end, row_end)
        
        return GeoTile(
            data=data,
            transform=crop_transform,
            bounds=(actual_left, actual_bottom, actual_right, actual_top),
            crs=str(self.dsm_crs)
        )

    def crop_fixed_pixels(self, center_x: float, center_y: float,
                          width_px: int, height_px: int,
                          is_dsm: bool = False) -> GeoTile:
        """Crop a fixed pixel window around a UTM center; pad out-of-bounds with 0/nodata."""
        transform = self.dsm_transform if is_dsm else self.dop_transform
        path = self.dsm_path if is_dsm else self.dop_path
        crs = self.dsm_crs if is_dsm else self.dop_crs
        full_width = self.dsm_width if is_dsm else self.dop_width
        full_height = self.dsm_height if is_dsm else self.dop_height

        col_c, row_c = ~transform * (center_x, center_y)
        col_start = int(round(col_c - width_px / 2))
        row_start = int(round(row_c - height_px / 2))
        col_end = col_start + width_px
        row_end = row_start + height_px

        v_col_start = max(0, col_start)
        v_row_start = max(0, row_start)
        v_col_end = min(full_width, col_end)
        v_row_end = min(full_height, row_end)
        v_width = v_col_end - v_col_start
        v_height = v_row_end - v_row_start

        channels = 1 if is_dsm else 3
        fill_val = -9999.0 if is_dsm else 0
        if is_dsm:
            buffer = np.full((height_px, width_px), fill_val, dtype=np.float32)
        else:
            buffer = np.zeros((height_px, width_px, channels), dtype=np.uint8)

        if v_width > 0 and v_height > 0:
            try:
                b_col_start = v_col_start - col_start
                b_row_start = v_row_start - row_start
                preloaded = self._dsm_data if is_dsm else self._dop_data
                if preloaded is not None:
                    if is_dsm:
                        buffer[b_row_start:b_row_start + v_height,
                               b_col_start:b_col_start + v_width] = preloaded[v_row_start:v_row_end, v_col_start:v_col_end]
                    else:
                        native_crop = preloaded[:3, v_row_start:v_row_end, v_col_start:v_col_end]
                        buffer[b_row_start:b_row_start + v_height,
                               b_col_start:b_col_start + v_width] = np.transpose(native_crop, (1, 2, 0))
                else:
                    src = self._get_src(is_dsm=is_dsm)
                    if src is not None:
                        window = Window(v_col_start, v_row_start, v_width, v_height)
                        data = src.read(window=window)
                        if is_dsm:
                            buffer[b_row_start:b_row_start + v_height, b_col_start:b_col_start + v_width] = data[0]
                        else:
                            data_hwc = np.transpose(data[:3], (1, 2, 0))
                            buffer[b_row_start:b_row_start + v_height, b_col_start:b_col_start + v_width] = data_hwc
            except Exception as e:
                print(f"Warning: Failed to read fixed pixel crop from {path}: {e}")

        crop_transform = transform * rasterio.Affine.translation(col_start, row_start)
        actual_left, actual_top = transform * (col_start, row_start)
        actual_right, actual_bottom = transform * (col_end, row_end)

        return GeoTile(
            data=buffer,
            transform=crop_transform,
            bounds=(actual_left, actual_bottom, actual_right, actual_top),
            crs=str(crs)
        )

    def load_full_dop(self) -> GeoTile:
        """Load the full DOP image."""
        with rasterio.open(self.dop_path) as src:
            data = src.read()
            data = np.transpose(data, (1, 2, 0))
        
        return GeoTile(
            data=data,
            transform=self.dop_transform,
            bounds=(self.dop_bounds.left, self.dop_bounds.bottom, 
                    self.dop_bounds.right, self.dop_bounds.top),
            crs=str(self.dop_crs)
        )
    
    def load_full_dsm(self) -> GeoTile:
        """Load the full DSM."""
        with rasterio.open(self.dsm_path) as src:
            data = src.read(1)
        
        return GeoTile(
            data=data,
            transform=self.dsm_transform,
            bounds=(self.dsm_bounds.left, self.dsm_bounds.bottom,
                    self.dsm_bounds.right, self.dsm_bounds.top),
            crs=str(self.dsm_crs)
        )
    
    def get_elevation_at_utm(self, dsm_tile: GeoTile, x: float, y: float) -> float:
        """Get elevation at a UTM coordinate from a DSM tile."""
        col, row = dsm_tile.utm_to_pixel(x, y)
        col, row = int(col), int(row)
        
        if 0 <= row < dsm_tile.height and 0 <= col < dsm_tile.width:
            return float(dsm_tile.data[row, col])
        return np.nan
    
    def get_elevation(self, x: float, y: float) -> Optional[float]:
        """
        Get elevation at a UTM coordinate directly from the full DSM.
        
        Args:
            x, y: UTM coordinates
            
        Returns:
            Elevation in meters, or None if out of bounds or invalid"""
        # Convert UTM to DSM pixel coordinates
        col, row = ~self.dsm_transform * (x, y)
        col, row = int(col), int(row)
        
        # Check bounds
        if not (0 <= col < self.dsm_width and 0 <= row < self.dsm_height):
            return None
        
        # Read single pixel from DSM
        with rasterio.open(self.dsm_path) as src:
            window = Window(col, row, 1, 1)
            z = src.read(1, window=window)[0, 0]
        
        if np.isnan(z) or z <= 0:
            return None
            
        return float(z)


class MultiTileGeoTIFFHandler(GeoTIFFHandler):
    """GeoTIFFHandler that transparently mosaics multiple DOP and/or DSM tiles.

    Accepts explicit lists of file paths.
    Crops that span tile boundaries are seamlessly merged via rasterio.merge."""

    def __init__(self, dop_paths=None, dsm_paths=None):
        # Do NOT call super().__init__ — we set attributes manually
        self._dop_tiles = []  # list of (bounds, path)
        self._dsm_tiles = []
        self._dop_data = None
        self._dsm_data = None
        self._dop_src = None
        self._dsm_src = None
        self.dop_path = None
        self.dsm_path = None

        dop_files = self._resolve_paths(dop_paths)
        dsm_files = self._resolve_paths(dsm_paths)

        # --- DOP tiles ---
        if dop_files:
            self.dop_path = dop_files[0]  # keep first for compat
            for p in dop_files:
                with rasterio.open(p) as src:
                    self._dop_tiles.append((src.bounds, p))
            # Compute combined bounding box and uniform transform
            self._init_combined_dop(dop_files)
        else:
            self.dop_crs = None
            self.dop_transform = None
            self.dop_bounds = None
            self.dop_width = 0
            self.dop_height = 0
            self.dop_resolution = 0.0

        # --- DSM tiles ---
        if dsm_files:
            self.dsm_path = dsm_files[0]
            for p in dsm_files:
                with rasterio.open(p) as src:
                    self._dsm_tiles.append((src.bounds, p))
            self._init_combined_dsm(dsm_files)
        else:
            self.dsm_crs = None
            self.dsm_transform = None
            self.dsm_bounds = None
            self.dsm_width = 0
            self.dsm_height = 0
            self.dsm_resolution = 0.0

        # Coordinate transformers
        target_crs = self.dop_crs if self.dop_crs else self.dsm_crs
        if target_crs:
            self.transformer_wgs84_to_utm = pyproj.Transformer.from_crs(
                'EPSG:4326', str(target_crs), always_xy=True)
            self.transformer_utm_to_wgs84 = pyproj.Transformer.from_crs(
                str(target_crs), 'EPSG:4326', always_xy=True)
        else:
            self.transformer_wgs84_to_utm = None
            self.transformer_utm_to_wgs84 = None

    # ---- helpers ----

    @staticmethod
    def _resolve_paths(paths):
        """Accept a string, Path, or list; expand directories to .tif files."""
        if paths is None:
            return []
        if isinstance(paths, (str, Path)):
            paths = [paths]
        result = []
        for p in paths:
            p = Path(p)
            if p.is_dir():
                result.extend(sorted(p.glob('*.tif')) + sorted(p.glob('*.tiff')))
            elif p.is_file():
                result.append(p)
        return [str(f) for f in result]

    def _init_combined_dop(self, files):
        with rasterio.open(files[0]) as src:
            self.dop_crs = src.crs
            self.dop_resolution = src.res[0]
        left = min(b.left for b, _ in self._dop_tiles)
        bottom = min(b.bottom for b, _ in self._dop_tiles)
        right = max(b.right for b, _ in self._dop_tiles)
        top = max(b.top for b, _ in self._dop_tiles)
        res = self.dop_resolution
        self.dop_transform = rasterio.Affine(res, 0.0, left, 0.0, -res, top)
        self.dop_width = int(round((right - left) / res))
        self.dop_height = int(round((top - bottom) / res))
        self.dop_bounds = rasterio.coords.BoundingBox(left, bottom, right, top)

    def _init_combined_dsm(self, files):
        with rasterio.open(files[0]) as src:
            self.dsm_crs = src.crs
            self.dsm_resolution = src.res[0]
        left = min(b.left for b, _ in self._dsm_tiles)
        bottom = min(b.bottom for b, _ in self._dsm_tiles)
        right = max(b.right for b, _ in self._dsm_tiles)
        top = max(b.top for b, _ in self._dsm_tiles)
        res = self.dsm_resolution
        self.dsm_transform = rasterio.Affine(res, 0.0, left, 0.0, -res, top)
        self.dsm_width = int(round((right - left) / res))
        self.dsm_height = int(round((top - bottom) / res))
        self.dsm_bounds = rasterio.coords.BoundingBox(left, bottom, right, top)

    @staticmethod
    def _overlapping(tiles, left, bottom, right, top):
        """Return paths of tiles whose bounds overlap the given rectangle."""
        out = []
        for bounds, path in tiles:
            if bounds.right <= left or bounds.left >= right:
                continue
            if bounds.top <= bottom or bounds.bottom >= top:
                continue
            out.append(path)
        return out

    def _merge_crop(self, tiles, paths, left, bottom, right, top, is_dsm=False):
        """Read and merge the crop region from overlapping tiles."""
        from rasterio.merge import merge as rio_merge

        datasets = [rasterio.open(p) for p in paths]
        try:
            merged, merged_transform = rio_merge(
                datasets, bounds=(left, bottom, right, top),
                res=self.dsm_resolution if is_dsm else self.dop_resolution,
            )
        finally:
            for ds in datasets:
                ds.close()

        if is_dsm:
            data = merged[0]  # (H, W)
        else:
            if merged.shape[0] == 1:
                merged = np.repeat(merged, 3, axis=0)
            elif merged.shape[0] > 3:
                merged = merged[:3]
            if merged.dtype != np.uint8:
                merged = np.clip(merged, 0, 255).astype(np.uint8)
            data = np.transpose(merged, (1, 2, 0))  # (H, W, 3)

        return data, merged_transform

    # ---- overrides ----

    def crop_dop(self, center_x, center_y, crop_size_meters):
        half = crop_size_meters / 2
        left, right = center_x - half, center_x + half
        bottom, top = center_y - half, center_y + half

        paths = self._overlapping(self._dop_tiles, left, bottom, right, top)
        if not paths:
            return None

        if len(paths) == 1:
            # Single tile — use parent's efficient path
            orig_path = self.dop_path
            with rasterio.open(paths[0]) as src:
                self.dop_transform = src.transform
                self.dop_width = src.width
                self.dop_height = src.height
            self.dop_path = paths[0]
            tile = super().crop_dop(center_x, center_y, crop_size_meters)
            self.dop_path = orig_path
            # Restore combined transform
            self._init_combined_dop([p for _, p in self._dop_tiles])
            return tile

        data, transform = self._merge_crop(
            self._dop_tiles, paths, left, bottom, right, top, is_dsm=False)
        if data.shape[0] < 10 or data.shape[1] < 10:
            return None
        actual_left = transform.c
        actual_top = transform.f
        actual_right = actual_left + data.shape[1] * self.dop_resolution
        actual_bottom = actual_top - data.shape[0] * self.dop_resolution
        return GeoTile(
            data=data, transform=transform,
            bounds=(actual_left, actual_bottom, actual_right, actual_top),
            crs=str(self.dop_crs),
        )

    def crop_dsm(self, center_x, center_y, crop_size_meters):
        half = crop_size_meters / 2
        left, right = center_x - half, center_x + half
        bottom, top = center_y - half, center_y + half

        paths = self._overlapping(self._dsm_tiles, left, bottom, right, top)
        if not paths:
            return None

        if len(paths) == 1:
            orig_path = self.dsm_path
            with rasterio.open(paths[0]) as src:
                self.dsm_transform = src.transform
                self.dsm_width = src.width
                self.dsm_height = src.height
            self.dsm_path = paths[0]
            tile = super().crop_dsm(center_x, center_y, crop_size_meters)
            self.dsm_path = orig_path
            self._init_combined_dsm([p for _, p in self._dsm_tiles])
            return tile

        data, transform = self._merge_crop(
            self._dsm_tiles, paths, left, bottom, right, top, is_dsm=True)
        if data.shape[0] < 10 or data.shape[1] < 10:
            return None
        actual_left = transform.c
        actual_top = transform.f
        actual_right = actual_left + data.shape[1] * self.dsm_resolution
        actual_bottom = actual_top - data.shape[0] * self.dsm_resolution
        return GeoTile(
            data=data, transform=transform,
            bounds=(actual_left, actual_bottom, actual_right, actual_top),
            crs=str(self.dsm_crs),
        )

    def load_full_dop(self):
        from rasterio.merge import merge as rio_merge
        datasets = [rasterio.open(p) for _, p in self._dop_tiles]
        try:
            merged, transform = rio_merge(datasets)
        finally:
            for ds in datasets:
                ds.close()
        if merged.shape[0] > 3:
            merged = merged[:3]
        if merged.dtype != np.uint8:
            merged = np.clip(merged, 0, 255).astype(np.uint8)
        data = np.transpose(merged, (1, 2, 0))
        b = self.dop_bounds
        return GeoTile(data=data, transform=transform,
                       bounds=(b.left, b.bottom, b.right, b.top),
                       crs=str(self.dop_crs))

    def load_full_dsm(self):
        from rasterio.merge import merge as rio_merge
        datasets = [rasterio.open(p) for _, p in self._dsm_tiles]
        try:
            merged, transform = rio_merge(datasets)
        finally:
            for ds in datasets:
                ds.close()
        data = merged[0]
        b = self.dsm_bounds
        return GeoTile(data=data, transform=transform,
                       bounds=(b.left, b.bottom, b.right, b.top),
                       crs=str(self.dsm_crs))

    def get_elevation(self, x, y):
        for bounds, path in self._dsm_tiles:
            if bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top:
                with rasterio.open(path) as src:
                    col, row = ~src.transform * (x, y)
                    col, row = int(col), int(row)
                    if 0 <= col < src.width and 0 <= row < src.height:
                        z = src.read(1, window=Window(col, row, 1, 1))[0, 0]
                        if not np.isnan(z) and z > 0:
                            return float(z)
        return None

    def preload(self, is_dsm=False):
        """Load full merged raster into memory for fast numpy-based sampling."""
        if is_dsm and self._dsm_data is None and self._dsm_tiles:
            tile = self.load_full_dsm()
            if tile is not None:
                self._dsm_data = tile.data  # (H, W) float
        elif not is_dsm and self._dop_data is None and self._dop_tiles:
            tile = self.load_full_dop()
            if tile is not None:
                data = tile.data
                # Store as (bands, H, W) to match GeoTIFFHandler convention
                if data.ndim == 3 and data.shape[2] in (3, 4):
                    self._dop_data = np.transpose(data, (2, 0, 1))
                else:
                    self._dop_data = data

    def close(self):
        pass


class SequenceGeoHandler:
    """
    Handles loading and cropping of pre-processed sequence data (DOP/DSM) from a directory.
    Compatible with GeoTIFFHandler interface but works with local caches (dop.jpg, dsm.npz)."""
    
    def __init__(self, sequence_dir: str, dop_year: Union[str, int] = 'last',
                 dsm_scale: float = 1.0, dsm_sigma_z: float = 0.0,
                 dsm_noise_seed: int = 0, dop_scale: float = 1.0):
        self.sequence_dir = Path(sequence_dir)
        self.meta_path = self.sequence_dir / "meta.json"
        self.dsm_scale = float(dsm_scale)
        self.dsm_sigma_z = float(dsm_sigma_z)
        self.dsm_noise_seed = int(dsm_noise_seed)
        self.dop_scale = float(dop_scale)
        
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {self.meta_path}")
            
        with open(self.meta_path, 'r') as f:
            self.meta = json.load(f)

        # --- Discover all available DOPs: list of (year, path, meta_dict) ---
        available_dops = []

        dops_meta = self.meta.get('dops', {})
        if dops_meta:
            for yr_str in sorted(dops_meta.keys(), reverse=True):
                entry = dops_meta[yr_str]
                candidate = self.sequence_dir / entry['file']
                if candidate.exists():
                    available_dops.append((entry.get('year', int(yr_str.split('_')[-1])), candidate, entry))
                    continue
                legacy = self.sequence_dir / f"dop_{entry['year']}.jpg"
                if legacy.exists():
                    available_dops.append((entry['year'], legacy, entry))

        # Legacy fallback: meta['dop'] + dop.jpg
        if not available_dops and self.meta.get('dop'):
            legacy_path = self.sequence_dir / "dop.jpg"
            if legacy_path.exists():
                dm = self.meta['dop']
                available_dops.append((dm.get('year', 0), legacy_path, dm))

        # Fallback: scan dop/<year>.jpg files on disk
        # bounds in dsm.npz are render-origin-relative (same coord system as
        # DOP bounds and poses.csv); keep them as-is.
        if not available_dops:
            dsm_path = self.sequence_dir / "dsm.npz"
            dop_subdir = self.sequence_dir / 'dop'
            if dop_subdir.is_dir() and dsm_path.exists():
                for jpg in sorted(dop_subdir.glob('*.jpg'), reverse=True):
                    try:
                        yr = int(jpg.stem)
                    except ValueError:
                        continue
                    with np.load(dsm_path) as dsm_data:
                        raw_bounds = list(dsm_data['bounds'])
                        dm = {
                            'gsd': float(dsm_data['gsd']),
                            'bounds': raw_bounds,
                            'year': yr,
                            'file': f'dop/{jpg.name}',
                        }
                    available_dops.append((yr, jpg, dm))

        if not available_dops:
            raise FileNotFoundError(f"No DOP image found in {self.sequence_dir}")

        # Sort by year descending, select based on dop_year
        available_dops.sort(key=lambda x: x[0], reverse=True)

        if isinstance(dop_year, int):
            chosen = None
            for yr, p, m in available_dops:
                if yr == dop_year:
                    chosen = (yr, p, m)
                    break
            if chosen is None:
                print(f"Warning: DOP year {dop_year} not found. Using most recent.")
                chosen = available_dops[0]
        else:
            # 'last' (default)
            chosen = available_dops[0]

        self.dop_year_resolved = chosen[0]
        self.dop_path = chosen[1]
        dop_info = chosen[2]
             
        # Load as RGB (opencv loads BGR by default)
        self.dop_data = cv2.cvtColor(cv2.imread(str(self.dop_path)), cv2.COLOR_BGR2RGB)

        # Optional in-memory DOP degradation (rebuttal sensitivity sweep):
        # downsample by dop_scale (area-mean) then upsample. Same shape, same
        # transform; only the photometric resolution changes.
        if self.dop_scale < 1.0 - 1e-6:
            print(f"  [SequenceGeoHandler] degrading DOP: scale={self.dop_scale:.4g} "
                  f"-> effective GSD {float(dop_info['gsd']) / max(self.dop_scale, 1e-6):.3g} m")
            self.dop_data = degrade_dop(self.dop_data, scale=self.dop_scale)

        # Bounds: [min_x, min_y, max_x, max_y]
        self.dop_bounds = dop_info['bounds'] 
        self.dop_gsd = float(dop_info['gsd'])
        self.dop_width = self.dop_data.shape[1]
        self.dop_height = self.dop_data.shape[0]
        
        # Create DOP Transform (Affine)
        # Transform maps (col, row) -> (x, y)
        # x = min_x + gsd * col
        # y = max_y - gsd * row (y decreases as row increases, standard image coords)
        self.dop_transform = rasterio.Affine(
            self.dop_gsd, 0.0, self.dop_bounds[0],
            0.0, -self.dop_gsd, self.dop_bounds[3]
        )
        self.dop_crs = dop_info.get('crs', "EPSG:25833") # Default if missing
        self.dop_resolution = self.dop_gsd
        
        # Load DSM (bounds are in same coordinate system as DOP — both relative
        # to render origin; do NOT apply render_origin offset here since DOP
        # bounds and poses.csv are also in the same relative system)
        self.dsm_path = self.sequence_dir / "dsm.npz"
        if not self.dsm_path.exists():
             raise FileNotFoundError(f"DSM file not found: {self.dsm_path}")
             
        with np.load(self.dsm_path) as data:
            self.dsm_data = data['height'] # (H, W)
            self.dsm_bounds = list(np.array(data.get('bounds', self.dop_bounds), dtype=np.float64))
            self.dsm_gsd = float(data.get('gsd', self.dop_gsd))

        # Optional in-memory DSM degradation (rebuttal sensitivity sweep):
        # downsample by dsm_scale (area-mean) then upsample, and add Gaussian
        # vertical noise of std dsm_sigma_z metres. Same shape, same transform.
        if (self.dsm_scale < 1.0 - 1e-6) or (self.dsm_sigma_z > 0.0):
            print(f"  [SequenceGeoHandler] degrading DSM: scale={self.dsm_scale:.4g} "
                  f"-> effective GSD {self.dsm_gsd / max(self.dsm_scale, 1e-6):.3g} m, "
                  f"sigma_z={self.dsm_sigma_z:.3g} m, seed={self.dsm_noise_seed}")
            self.dsm_data = degrade_dsm(self.dsm_data,
                                        scale=self.dsm_scale,
                                        sigma_z=self.dsm_sigma_z,
                                        seed=self.dsm_noise_seed)
            
        self.dsm_width = self.dsm_data.shape[1]
        self.dsm_height = self.dsm_data.shape[0]
        self.dsm_resolution = self.dsm_gsd
        
        self.dsm_transform = rasterio.Affine(
            self.dsm_gsd, 0.0, self.dsm_bounds[0],
            0.0, -self.dsm_gsd, self.dsm_bounds[3]
        )
        self.dsm_crs = self.dop_crs

    def close(self):
        pass # Nothing to close, data is in memory

    def __del__(self):
        pass

    def latlon_to_utm(self, lat: float, lon: float) -> Tuple[float, float]:
        # Not implemented / strictly needed for sequence tracking if poses are already XYZ
        # But if needed, we'd need pyproj and a reference origin.
        # Assuming poses are already xyz, this might not be called.
        raise NotImplementedError("latlon_to_utm not supported for SequenceGeoHandler (local coords)")
    
    def utm_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        raise NotImplementedError("utm_to_latlon not supported for SequenceGeoHandler")
    
    def utm_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert local/UTM coordinates to pixel coordinates in the DOP."""
        col, row = ~self.dop_transform * (x, y)
        return int(col), int(row)
    
    def pixel_to_utm(self, col: int, row: int) -> Tuple[float, float]:
        """Convert pixel coordinates to local/UTM coordinates."""
        x, y = self.dop_transform * (col, row)
        return x, y
    
    def get_resolution(self) -> float:
        return self.dop_resolution
    
    def crop_dop(self, center_x: float, center_y: float, 
                 crop_size_meters: float) -> Optional[GeoTile]:
        """Crop DOP around a center point."""
        half_size = crop_size_meters / 2
        left = center_x - half_size
        right = center_x + half_size
        bottom = center_y - half_size
        top = center_y + half_size
        
        col_start, row_start = self.utm_to_pixel(left, top)
        col_end, row_end = self.utm_to_pixel(right, bottom)
        
        # Clamp bounds
        col_start = max(0, col_start)
        row_start = max(0, row_start)
        col_end = min(self.dop_width, col_end)
        row_end = min(self.dop_height, row_end)
        
        width = col_end - col_start
        height = row_end - row_start
        
        if width <= 10 or height <= 10:
            return None
        
        # Crop data (H, W, 3)
        data = self.dop_data[row_start:row_end, col_start:col_end].copy()
        
        # Calculate transform for the cropped region
        crop_transform = self.dop_transform * rasterio.Affine.translation(col_start, row_start)
        
        # Actual bounds of the cropped region
        actual_left, actual_top = self.pixel_to_utm(col_start, row_start)
        actual_right, actual_bottom = self.pixel_to_utm(col_end, row_end)
        
        return GeoTile(
            data=data,
            transform=crop_transform,
            bounds=(actual_left, actual_bottom, actual_right, actual_top),
            crs=str(self.dop_crs)
        )

    def crop_dsm(self, center_x: float, center_y: float, 
                 crop_size_meters: float) -> Optional[GeoTile]:
        """Crop DSM around a center point."""
        half_size = crop_size_meters / 2
        left = center_x - half_size
        right = center_x + half_size
        bottom = center_y - half_size
        top = center_y + half_size
        
        # Convert to pixel coordinates using DSM transform
        col_start, row_start = ~self.dsm_transform * (left, top)
        col_end, row_end = ~self.dsm_transform * (right, bottom)
        
        col_start, row_start = int(col_start), int(row_start)
        col_end, row_end = int(col_end), int(row_end)
        
        # Clamp bounds
        col_start = max(0, col_start)
        row_start = max(0, row_start)
        col_end = min(self.dsm_width, col_end)
        row_end = min(self.dsm_height, row_end)
        
        width = col_end - col_start
        height = row_end - row_start
        
        if width <= 10 or height <= 10:
            return None
            
        data = self.dsm_data[row_start:row_end, col_start:col_end].copy()
        
        crop_transform = self.dsm_transform * rasterio.Affine.translation(col_start, row_start)
        
        actual_left, actual_top = self.dsm_transform * (col_start, row_start)
        actual_right, actual_bottom = self.dsm_transform * (col_end, row_end)
        
        return GeoTile(
            data=data,
            transform=crop_transform,
            bounds=(actual_left, actual_bottom, actual_right, actual_top),
            crs=str(self.dsm_crs)
        )
    
    def get_elevation_at_utm(self, dsm_tile: GeoTile, x: float, y: float) -> float:
        """Get elevation at a UTM coordinate from a DSM tile."""
        # Reuse logic from GeoTIFFHandler compatible structure
        col, row = dsm_tile.utm_to_pixel(x, y)
        col, row = int(col), int(row)
        
        if 0 <= row < dsm_tile.height and 0 <= col < dsm_tile.width:
            return float(dsm_tile.data[row, col])
        return np.nan

    def get_elevation(self, x: float, y: float) -> Optional[float]:
        """Get elevation from full DSM."""
        col, row = ~self.dsm_transform * (x, y)
        col, row = int(col), int(row)
        
        if not (0 <= col < self.dsm_width and 0 <= row < self.dsm_height):
            return None
            
        z = self.dsm_data[row, col]
        if np.isnan(z) or z <= -1000:
            return None
            
        return float(z)
def world_to_dop_px(x, y, crop_bounds, img_shape):
    """Convert world UTM (x, y) to DOP image pixel coordinates."""
    c_min_x, c_min_y, c_max_x, c_max_y = crop_bounds
    H, W = img_shape[:2]
    px = (x - c_min_x) / (c_max_x - c_min_x) * (W - 1)
    py = (c_max_y - y) / (c_max_y - c_min_y) * (H - 1)
    return px, py

def compute_frustum_corners(w2c, K, img_h, img_w, z_ground):
    """Unproject 4 image corners onto the ground plane at z_ground.

    Returns list of 4 (x_utm, y_utm) tuples, or None if no valid corners.
    For oblique views where some rays miss the ground plane (parallel or pointing
    away), those corners are clamped to MAX_XY_DISP metres along the XY
    projection.  XY displacement is always capped so the polygon stays inside
    the pre-computed sequence DOP/DSM bbox (which uses the same cap)."""
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    C = -R.T @ t
    K_inv = np.linalg.inv(K[:3, :3])

    # Maximum XY displacement (metres) from camera center.
    # Must match MAX_XY_DISP in compute_sequence_bbox() so the FoV polygon
    # never extends beyond the pre-computed DOP/DSM coverage.
    MAX_XY_DISP = 2000.0

    corners = []
    for u, v in [(0, 0), (img_w, 0), (img_w, img_h), (0, img_h)]:
        d_cam = K_inv @ np.array([u, v, 1.0])
        d_world = R.T @ d_cam
        d_world_norm = np.linalg.norm(d_world)
        if d_world_norm < 1e-12:
            continue
        d_unit = d_world / d_world_norm

        if abs(d_world[2]) > 1e-6:
            t_param = (z_ground - C[2]) / d_world[2]
        else:
            t_param = -1  # ray parallel to ground

        if t_param > 0:
            # Valid ground intersection
            hit_xy = C[:2] + t_param * d_world[:2]
        else:
            # Ray misses ground (parallel or looking upward).
            # Project MAX_XY_DISP along the ray's XY direction.
            d_xy = d_unit[:2]
            d_xy_len = np.linalg.norm(d_xy)
            if d_xy_len < 1e-8:
                continue  # looking straight up/down with no XY component
            hit_xy = C[:2] + (MAX_XY_DISP / d_xy_len) * d_xy

        # Cap XY displacement from camera center
        disp = hit_xy - C[:2]
        disp_len = np.linalg.norm(disp)
        if disp_len > MAX_XY_DISP:
            hit_xy = C[:2] + (MAX_XY_DISP / disp_len) * disp

        corners.append((float(hit_xy[0]), float(hit_xy[1])))

    if len(corners) < 3:
        return None  # need at least a triangle
    return corners

def compute_camera_triangle(w2c, size=5.0):
    """Compute a triangle in world XY representing camera position and orientation.

    The triangle nose points along the camera's forward (optical axis) projected
    onto the XY ground plane. Two wing points sit behind the center perpendicular
    to the forward direction.

    Args:
        w2c: (4, 4) W2C matrix
        size: triangle half-width in meters

    Returns:
        list of 3 (x_utm, y_utm): [nose, left_wing, right_wing], or None"""
    R = w2c[:3, :3]
    t_vec = w2c[:3, 3]
    C = -R.T @ t_vec  # camera center in world

    # Camera forward in world = R^T @ [0, 0, 1] (camera Z axis)
    fwd_world = R.T @ np.array([0.0, 0.0, 1.0])
    # Project onto XY plane
    fwd_xy = fwd_world[:2]
    norm = np.linalg.norm(fwd_xy)
    if norm < 1e-6:
        # Near-nadir: use camera X axis (right) as heading fallback
        right_world = R.T @ np.array([1.0, 0.0, 0.0])
        fwd_xy = right_world[:2]
        norm = np.linalg.norm(fwd_xy)
        if norm < 1e-6:
            return None  # truly degenerate
    fwd_xy = fwd_xy / norm

    # Perpendicular in XY
    perp_xy = np.array([-fwd_xy[1], fwd_xy[0]])

    cx, cy = C[0], C[1]
    # Inverted triangle: tail vertex at camera center (on trajectory),
    # wide base points forward in viewing direction.
    tail = (cx, cy)
    left = (cx + fwd_xy[0] * size + perp_xy[0] * size * 0.6,
            cy + fwd_xy[1] * size + perp_xy[1] * size * 0.6)
    right = (cx + fwd_xy[0] * size - perp_xy[0] * size * 0.6,
             cy + fwd_xy[1] * size - perp_xy[1] * size * 0.6)
    return [tail, left, right]

def compute_visible_dop_mask(point_map, dsm_raw, crop_bounds, w2c, K, img_h, img_w):
    """Compute which DOP pixels are co-visible from the current UAV frame.

    Projects co-visible point_map pixels (horizontal OR DSM-consistent)
    onto the DOP grid and returns a boolean mask in DOP pixel space.

    Args:
        point_map: (3, H, W) tensor — raw (denormalized) XYZ point map
        dsm_raw: (3, H, W) tensor — raw XYZ DSM
        crop_bounds: (min_x, min_y, max_x, max_y) UTM crop bounds
        w2c: (3, 4) or (4, 4) numpy W2C matrix
        K: (3, 3) numpy intrinsics
        img_h, img_w: UAV image dimensions

    Returns:
        visible_dop: (Hd, Wd) boolean mask — True for co-visible DOP pixels"""
    pm_np = point_map.cpu().numpy() if torch.is_tensor(point_map) else point_map
    dsm_np = dsm_raw.cpu().numpy() if torch.is_tensor(dsm_raw) else dsm_raw
    Hd, Wd = dsm_np.shape[1:]
    x_min, y_min, x_max, y_max = crop_bounds

    valid_pm = np.any(pm_np != 0, axis=0)

    # Horizontal normal criterion
    _, horiz = compute_surface_normals_from_pointmap(point_map)

    # DSM-Z consistency criterion
    pm_x, pm_y, pm_z = pm_np[0], pm_np[1], pm_np[2]
    du = (pm_x - x_min) / (x_max - x_min) * Wd
    dv = (y_max - pm_y) / (y_max - y_min) * Hd
    in_b = (du >= 0) & (du < Wd) & (dv >= 0) & (dv < Hd)
    du_i = np.clip(du.astype(int), 0, Wd - 1)
    dv_i = np.clip(dv.astype(int), 0, Hd - 1)
    dsm_z = dsm_np[2, dv_i, du_i]
    z_diff_raw = pm_z - dsm_z
    valid_off = valid_pm & in_b & horiz
    z_off = float(np.median(z_diff_raw[valid_off])) if np.sum(valid_off) > 100 else 0.0
    z_diff = z_diff_raw - z_off
    z_ok = (z_diff < 3.0) & (z_diff > -2.0)
    dsm_con = in_b & z_ok

    covis = valid_pm & (dsm_con | horiz)

    # Project co-visible pixels to DOP
    idx = np.where(covis)
    wx = pm_np[0, idx[0], idx[1]]
    wy = pm_np[1, idx[0], idx[1]]
    dop_u = ((wx - x_min) / (x_max - x_min) * Wd).astype(int)
    dop_v = ((y_max - wy) / (y_max - y_min) * Hd).astype(int)
    ok = (dop_u >= 0) & (dop_u < Wd) & (dop_v >= 0) & (dop_v < Hd)

    visible_dop = np.zeros((Hd, Wd), dtype=bool)
    visible_dop[dop_v[ok], dop_u[ok]] = True
    # Fill sparse projection gaps
    visible_dop = binary_closing(visible_dop, iterations=5)
    visible_dop = binary_dilation(visible_dop, iterations=1)
    return visible_dop

def get_tile_bounds_from_name(tile_name: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Parse tile name to get approximate bounds.
    
    Tile naming conventions:
    - DOP/DSM simple:   32690_5335.tif   -> UTM zone 32, easting ~690km, northing ~5335km
    - LoDv2 simple:     690_5334.gml     -> easting ~690km, northing ~5334km
    - Berlin DSM:       bdom_33397-5802  -> zone 33, easting 397km, northing 5802km
    - Berlin ALS file:  als_33397-5803   -> zone 33, easting 397km, northing 5803km
    - Berlin ALS dir:   3dm_33_397_5802_1_be  -> zone 33, easting 397km, northing 5802km
    - Berlin LoD2:      LoD2_33_390_5820_1_BE -> zone 33, easting 390km, northing 5820km
    - Berlin LoD1:      LoD1_390_5820         -> easting 390km, northing 5820km
    
    Returns:
        (min_x, min_y, max_x, max_y) in UTM, or None if can't parse"""
    import re
    # Remove extension and path
    base = Path(tile_name).stem
    
    # Remove common suffixes
    for suffix in ['_20_DOM', '_DOM', '_DSM']:
        base = base.replace(suffix, '')
    
    # --- Strategy 1: Berlin-style "prefix_ZZEEE-NNNN" (e.g. bdom_33397-5802, als_33397-5803) ---
    m = re.search(r'(\d{2})(\d{3})-(\d{4})', base)
    if m:
        easting_km = int(m.group(2))   # e.g. 397
        northing_km = int(m.group(3))  # e.g. 5802
        return easting_km * 1000, northing_km * 1000, (easting_km + 1) * 1000, (northing_km + 1) * 1000
    
    # --- Strategy 2: Berlin-style "prefix_ZZ_EEE_NNNN_..." (e.g. LoD2_33_390_5820_1_BE, 3dm_33_397_5802_1_be) ---
    m = re.search(r'_33_(\d{3})_(\d{4})', base)
    if m:
        easting_km = int(m.group(1))
        northing_km = int(m.group(2))
        return easting_km * 1000, northing_km * 1000, (easting_km + 1) * 1000, (northing_km + 1) * 1000
    
    # --- Strategy 3: "prefix_EEE_NNNN" without zone (e.g. LoD1_390_5820) ---
    m = re.search(r'_(\d{3})_(\d{4})$', base)
    if not m:
        m = re.search(r'^(\d{3})_(\d{4})', base)
    if m:
        easting_km = int(m.group(1))
        northing_km = int(m.group(2))
        return easting_km * 1000, northing_km * 1000, (easting_km + 1) * 1000, (northing_km + 1) * 1000
    
    # --- Strategy 4: Original format "ZZEEE_NNNN" or "EEE_NNNN" ---
    parts = base.split('_')
    try:
        if len(parts) >= 2:
            if parts[0].startswith('32'):
                easting_km = int(parts[0][2:])
                northing_km = int(parts[1])
            else:
                easting_km = int(parts[0])
                northing_km = int(parts[1])
            
            min_x = easting_km * 1000
            max_x = (easting_km + 1) * 1000
            min_y = northing_km * 1000
            max_y = (northing_km + 1) * 1000
            return min_x, min_y, max_x, max_y
    except (ValueError, IndexError):
        pass
    
    return None

def get_tile_bounds_from_geotiff(tile_path: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    Get tile bounds from GeoTIFF metadata.
    
    Returns:
        (min_x, min_y, max_x, max_y) in the file's CRS"""
    try:
        with rasterio.open(tile_path) as src:
            bounds = src.bounds
            return bounds.left, bounds.bottom, bounds.right, bounds.top
    except Exception:
        return None

def boxes_intersect(box1: Tuple[float, float, float, float], 
                    box2: Tuple[float, float, float, float]) -> bool:
    """Check if two bounding boxes intersect."""
    min_x1, min_y1, max_x1, max_y1 = box1
    min_x2, min_y2, max_x2, max_y2 = box2
    
    return not (max_x1 < min_x2 or max_x2 < min_x1 or 
                max_y1 < min_y2 or max_y2 < min_y1)

def get_tile_bounds_from_gml_content(gml_path: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    Parse GML file content to find actual coordinate bounds.
    Useful when the filename implies a smaller region than the content (e.g. non-standard tiling)."""
    try:
        # Optimization: Use iterparse to avoid loading full tree if possible, 
        # but for bounds we need to see everything. To be safe/fast,
        # we'll look for the first few thousand points or check all depending on file size.
        # Given these are city tiles, they can be large. Let's try efficient scanning.
        
        context = ET.iterparse(gml_path, events=('end',))
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        found_any = False
        ns = {'gml': 'http://www.opengis.net/gml'}
        
        # Limit processing to avoid hanging on massive files if strictly checking bounds
        # But here we need accurate bounds.
        
        for event, elem in context:
            if elem.tag.endswith('posList'):
                try:
                    coords = [float(x) for x in elem.text.split()]
                    pts = np.array(coords).reshape(-1, 3)
                    
                    if len(pts) > 0:
                        min_x = min(min_x, pts[:, 0].min())
                        max_x = max(max_x, pts[:, 0].max())
                        min_y = min(min_y, pts[:, 1].min())
                        max_y = max(max_y, pts[:, 1].max())
                        found_any = True
                except: pass
                elem.clear()
                
        if found_any:
            return min_x, min_y, max_x, max_y
            
    except Exception as e:
        print(f"Warning: Failed to parse GML bounds for {gml_path.name}: {e}")
        
    return None

def find_intersecting_tiles(bbox: Tuple[float, float, float, float], 
                            geodata_dir: Path, gm: Optional['GeodataManager'] = None,
                            dop_dir: Optional[Path] = None, dsm_dir: Optional[Path] = None,
                            lod1_dir: Optional[Path] = None, lod2_dir: Optional[Path] = None,
                            als_dir: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Find all geodata tiles that intersect the given bounding box.
    
    Args:
        bbox: (min_x, min_y, max_x, max_y) in UTM
        geodata_dir: Path to geodata directory containing DOP/, DSM/, LoDv2/
        gm: Optional GeodataManager for automatic re-download of corrupted tiles
        
    Returns:
        Dictionary with 'dop', 'dsm', 'lod2' keys, each containing list of tile paths"""
    result = {'dop': [], 'dsm': [], 'lod1': [], 'lod2': [], 'als': [], 'mesh': []}
    
    # Helper to find directory case-insensitively, with common aliases
    _DIR_ALIASES = {
        'LoDv2': ['LoDv2', 'lodv2', 'LODV2', 'lod2', 'LoD2'],
        'LoDv1': ['LoDv1', 'lodv1', 'LODV1', 'lod1', 'LoD1'],
        'ALS':   ['ALS', 'als'],
        'DOP':   ['DOP', 'dop'],
        'DSM':   ['DSM', 'dsm'],
    }
    def get_dir(parent, name):
        # Check exact, lower, upper first
        for variant in [name, name.lower(), name.upper()]:
            if (parent / variant).exists(): return parent / variant
        # Check known aliases
        for alias in _DIR_ALIASES.get(name, []):
            if (parent / alias).exists(): return parent / alias
        return None

    # Helper to handle decompression and finding image in tile dir
    def find_tile_file(tile_dir: Path, extensions: List[str], data_type: str = 'dop') -> List[Path]:
        files = []
        for ext in extensions:
            files.extend(list(tile_dir.glob(ext)))
            files.extend(list(tile_dir.glob(ext.upper())))
        
        # If no image files but a zip exists, unzip it
        if not files:
            zips = list(tile_dir.glob("*.zip"))
            if zips:
                for zip_p in zips:
                    print(f"   Decompressing geodata tile: {zip_p.name}...")
                    try:
                        with zipfile.ZipFile(zip_p, 'r') as zip_ref:
                            zip_ref.extractall(tile_dir)
                    except Exception as e:
                        print(f"   Warning: Failed to unzip {zip_p.name}: {e}")
                        if gm:
                            print(f"   Attempting to re-download {data_type} tile {tile_dir.name}...")
                            # Remove corrupted dir and re-download
                            shutil.rmtree(tile_dir)
                            if data_type == 'dop':
                                gm.download_dop_tile(tile_dir.name)
                            elif data_type == 'dsm':
                                gm.download_dsm_tile(tile_dir.name)
                            
                            # Search again if successful
                            if tile_dir.exists():
                                for ext in extensions:
                                    files.extend(list(tile_dir.glob(ext)))
                                    files.extend(list(tile_dir.glob(ext.upper())))
                        else:
                            raise ValueError(f"Corrupted geodata ZIP {zip_p} and no GeodataManager available for re-download.")
                    
                    # Search again for unzipped files after first successful unzip
                    if not files:
                        for ext in extensions:
                            files.extend(list(tile_dir.glob(ext)))
                            files.extend(list(tile_dir.glob(ext.upper())))
        return files

    # Check DOP tiles
    dop_dir = dop_dir or get_dir(geodata_dir, 'DOP')
    if dop_dir:
        # DOP can follow a nested structure: DOP/tile_id/tile_id.tif or DOP/tile_id.tif
        # We handle both by globbing subdirs or files
        potential_tiles = []
        for p in dop_dir.iterdir():
            if p.is_dir():
                # Check for images or zips inside
                potential_tiles.extend(find_tile_file(p, ['*.tif', '*.jp2', '*.jpg', '*.png'], 'dop'))
            elif p.suffix.lower() in ['.tif', '.jp2', '*.jpg', '*.png', '.zip']:
                if p.suffix.lower() == '.zip':
                    potential_tiles.extend(find_tile_file(dop_dir, ['*.tif', '*.jp2', '*.jpg', '*.png'], 'dop'))
                else:
                    potential_tiles.append(p)
        
        # Unique and filter by bbox
        for tile_path in set(potential_tiles):
            tile_bounds = get_tile_bounds_from_geotiff(tile_path)
            if tile_bounds is None:
                tile_bounds = get_tile_bounds_from_name(tile_path.name)
            
            if tile_bounds and boxes_intersect(bbox, tile_bounds):
                result['dop'].append(str(tile_path.relative_to(geodata_dir)))
    
    # Check DSM tiles
    dsm_dir = dsm_dir or get_dir(geodata_dir, 'DSM')
    if dsm_dir:
        potential_tiles = []
        for p in dsm_dir.iterdir():
            if p.is_dir():
                potential_tiles.extend(find_tile_file(p, ['*.tif', '*.jp2'], 'dsm'))
            elif p.suffix.lower() in ['.tif', '.jp2', '.zip']:
                if p.suffix.lower() == '.zip':
                    potential_tiles.extend(find_tile_file(dsm_dir, ['*.tif', '*.jp2'], 'dsm'))
                else:
                    potential_tiles.append(p)
        
        for tile_path in set(potential_tiles):
            tile_bounds = get_tile_bounds_from_geotiff(tile_path)
            if tile_bounds is None:
                tile_bounds = get_tile_bounds_from_name(tile_path.name)
            
            if tile_bounds and boxes_intersect(bbox, tile_bounds):
                result['dsm'].append(str(tile_path.relative_to(geodata_dir)))
    
    # Check LoDv2 tiles
    lod2_dir = lod2_dir or get_dir(geodata_dir, 'LoDv2')
    if lod2_dir:
        # LoD2 tiles are usually .gml or .xml
        potential_tiles = []
        for p in lod2_dir.iterdir():
            if p.is_dir():
                potential_tiles.extend(find_tile_file(p, ['*.gml', '*.xml']))
            elif p.suffix.lower() in ['.gml', '.xml', '.zip']:
                if p.suffix.lower() == '.zip':
                    potential_tiles.extend(find_tile_file(lod2_dir, ['*.gml', '*.xml']))
                else:
                    potential_tiles.append(p)

        bbox_extended = (bbox[0]-3000.0, bbox[1]-3000.0, bbox[2]+3000.0, bbox[3]+3000.0)
        
        for tile_path in set(potential_tiles):
            name_bounds = get_tile_bounds_from_name(tile_path.name)
            if name_bounds:
                if boxes_intersect(bbox, name_bounds):
                    result['lod2'].append(str(tile_path.relative_to(geodata_dir)))
                    continue
                
                if boxes_intersect(bbox_extended, name_bounds):
                    content_bounds = get_tile_bounds_from_gml_content(tile_path)
                    if content_bounds and boxes_intersect(bbox, content_bounds):
                        result['lod2'].append(str(tile_path.relative_to(geodata_dir)))

    # Check LoDv1 tiles
    lod1_dir = lod1_dir or get_dir(geodata_dir, 'LoDv1')
    if lod1_dir:
        potential_tiles = []
        for p in lod1_dir.iterdir():
            if p.is_dir():
                potential_tiles.extend(find_tile_file(p, ['*.gml', '*.xml']))
            elif p.suffix.lower() in ['.gml', '.xml', '.zip']:
                if p.suffix.lower() == '.zip':
                    potential_tiles.extend(find_tile_file(lod1_dir, ['*.gml', '*.xml']))
                else:
                    potential_tiles.append(p)
        
        for tile_path in set(potential_tiles):
            name_bounds = get_tile_bounds_from_name(tile_path.name)
            if name_bounds and boxes_intersect(bbox, name_bounds):
                result['lod1'].append(str(tile_path.relative_to(geodata_dir)))

    # Check ALS tiles
    als_dir = als_dir or get_dir(geodata_dir, 'ALS')
    if als_dir:
        potential_tiles = []
        for p in als_dir.iterdir():
            if p.is_dir():
                potential_tiles.extend(find_tile_file(p, ['*.laz', '*.las']))
            elif p.suffix.lower() in ['.laz', '.las', '.zip']:
                if p.suffix.lower() == '.zip':
                    potential_tiles.extend(find_tile_file(als_dir, ['*.laz', '*.las']))
                else:
                    potential_tiles.append(p)
        
        for tile_path in set(potential_tiles):
            name_bounds = get_tile_bounds_from_name(tile_path.name)
            if name_bounds and boxes_intersect(bbox, name_bounds):
                result['als'].append(str(tile_path.relative_to(geodata_dir)))

    # Check mesh tiles (OBJ with textures, ~400m tiles)
    # Naming: {easting/100}_{northing/100}_-002/Mesh_*.obj
    mesh_dir_path = get_dir(geodata_dir, 'mesh')
    if mesh_dir_path:
        MESH_TILE_SIZE = 400  # approximate tile extent in metres
        for p in mesh_dir_path.iterdir():
            if not p.is_dir():
                continue
            # Parse coordinate from directory name: "3728_58131_-002"
            parts = p.name.split('_')
            if len(parts) < 2:
                continue
            try:
                e = int(parts[0]) * 100
                n = int(parts[1]) * 100
            except ValueError:
                continue
            # Check bbox intersection (conservative: tile may extend up to MESH_TILE_SIZE)
            tile_bounds = (e, n, e + MESH_TILE_SIZE, n + MESH_TILE_SIZE)
            if boxes_intersect(bbox, tile_bounds):
                # Verify it has OBJ files
                if list(p.glob('*.obj')):
                    result['mesh'].append(str(p.relative_to(geodata_dir)))

    return result

def compute_sequence_bbox(poses_csv, margin_m=50, depth_dir=None, fx=None, fy=None,
                          cx=None, cy=None, width=None, height=None,
                          subsample_pixels=64, subsample_frames=1,
                          crop_padding_m=150, max_dimension_m=5000):
    """Compute bounding box that covers the FULL UAV visible area for every frame.

    The bbox must be large enough so that at runtime, every per-frame crop
    (centred on the visible point cloud with GSD-dependent size) falls
    entirely inside the pre-saved DOP/DSM.

    Strategy (robust, combined depth + frustum):
      1. For each frame, back-project depth pixels → world XY.
      2. Fit a robust convex hull to these points with outlier removal
         (percentile filtering) to get the inlier visible footprint.
      3. For image-border pixels that lack valid depth (sky, beyond mesh),
         project the camera frustum corners onto an estimated ground plane
         so the bbox still covers the full field of view.
      4. Take the union of all per-frame convex hulls.
      5. Add ``margin_m`` + ``crop_padding_m`` on every side.
         ``crop_padding_m`` accounts for the extra GSD-based padding that
         the dataset crop adds around the visible area at runtime.
      6. Clamp bbox so no side exceeds ``max_dimension_m`` (centred on the
         camera trajectory midpoint).  This prevents oblique frustum
         projections from creating impractically large bboxes that exceed
         available geodata (DSM tiles typically cover 1 km² each).

    Falls back to camera-position ± margin_m when no depth maps are available.

    Args:
        poses_csv: Path to poses CSV (absolute UTM coordinates)
        margin_m: Extra margin in meters around the computed bbox (default 50)
        depth_dir: Path to directory with depth_XXXX.npz files
        fx, fy, cx, cy: Camera intrinsics
        width, height: Image dimensions
        subsample_pixels: Pixel stride for back-projection (speed vs accuracy)
        subsample_frames: Process every Nth depth file (default 1 = all).
            For long sequences, 4-8 is usually sufficient for bbox estimation.
        crop_padding_m: Extra padding in meters to account for per-frame crop
            extent.  At runtime the dataset loader centres a crop on the visible
            area and expands it by a GSD-dependent amount; this padding ensures
            the crop never falls outside the pre-saved geodata.  150 m is
            sufficient for most altitude/GSD combinations.
        max_dimension_m: Maximum bbox side length in metres (default 5000).
            Prevents excessively large bboxes from oblique frustum projections."""
    import pandas as pd
    from scipy.spatial.transform import Rotation
    from tqdm import tqdm

    df = pd.read_csv(poses_csv)

    if depth_dir is not None and fx is not None:
        depth_dir = Path(depth_dir)
        depth_files = sorted(depth_dir.glob('depth_*.npz'))

        if depth_files:
            # Temporal subsampling: process every Nth frame
            if subsample_frames > 1:
                depth_files = depth_files[::subsample_frames]
            print(f"    Computing bbox from depth + frustum ({len(depth_files)} frames, "
                  f"stride={subsample_pixels}, frame_step={subsample_frames})...")

            # ── pixel grid (sub-sampled) for interior back-projection ──
            us = np.arange(0, width, subsample_pixels, dtype=np.float64) + 0.5
            vs = np.arange(0, height, subsample_pixels, dtype=np.float64) + 0.5
            uu, vv = np.meshgrid(us, vs)
            rays_cam = np.stack([
                (uu.ravel() - cx) / fx,
                (vv.ravel() - cy) / fy,
                np.ones(uu.size)
            ], axis=-1)

            # ── dense border pixel rays (every 4 px along image edges) ──
            border_step = 4
            top = np.stack([np.arange(0, width, border_step) + 0.5,
                            np.zeros(len(range(0, width, border_step))) + 0.5], axis=-1)
            bot = np.stack([np.arange(0, width, border_step) + 0.5,
                            np.full(len(range(0, width, border_step)), height - 0.5)], axis=-1)
            lft = np.stack([np.zeros(len(range(0, height, border_step))) + 0.5,
                            np.arange(0, height, border_step) + 0.5], axis=-1)
            rgt = np.stack([np.full(len(range(0, height, border_step)), width - 0.5),
                            np.arange(0, height, border_step) + 0.5], axis=-1)
            border_px = np.concatenate([top, bot, lft, rgt], axis=0)
            border_rays = np.stack([
                (border_px[:, 0] - cx) / fx,
                (border_px[:, 1] - cy) / fy,
                np.ones(len(border_px))
            ], axis=-1)

            # ── image corner rays (4 corners) for frustum projection ──
            corner_px = np.array([
                [0.5, 0.5], [width - 0.5, 0.5],
                [width - 0.5, height - 0.5], [0.5, height - 0.5],
            ])
            corner_rays = np.stack([
                (corner_px[:, 0] - cx) / fx,
                (corner_px[:, 1] - cy) / fy,
                np.ones(4),
            ], axis=-1)

            all_xy = []          # collected world-XY points across all frames
            all_z = []           # Z values for computing global ground level
            frustum_xy = []      # frustum-projected corner points (guaranteed coverage)
            frame_info = []      # per-frame (C, Rm, corner_depth) for 2nd-pass frustum

            # ── PASS 1: depth back-projection + collect Z stats ──
            for depth_file in tqdm(depth_files, desc="    bbox frames (pass 1)",
                                   unit="fr", leave=False):
                idx = int(depth_file.stem.split('_')[1])
                if idx >= len(df):
                    continue
                row = df.iloc[idx]

                depth = np.load(depth_file)['depth']
                depth_sub = depth[::subsample_pixels, ::subsample_pixels].ravel()

                n = min(len(depth_sub), len(rays_cam))
                d_flat = depth_sub[:n]
                rays = rays_cam[:n]

                valid = np.isfinite(d_flat) & (d_flat > 0) & (d_flat < 1e5)

                Rm = Rotation.from_quat([row['qx'], row['qy'], row['qz'], row['qw']]).as_matrix()
                C = np.array([row['x'], row['y'], row['z']])

                # ── 1. depth back-projection (interior) ──
                # Rm is R_c2w (camera-to-world) from the quaternion.
                if valid.sum() > 0:
                    pts_cam = d_flat[valid, None] * rays[valid]
                    pts_world = (Rm @ pts_cam.T).T + C
                    all_xy.append(pts_world[:, :2])
                    all_z.append(pts_world[:, 2])

                # ── 2. border-pixel depth back-projection ──
                # Sample depth at border pixels using nearest-neighbor lookup
                border_u = np.clip(border_px[:, 0].astype(int), 0, width - 1)
                border_v = np.clip(border_px[:, 1].astype(int), 0, height - 1)
                border_depth = depth[border_v, border_u]
                bv = np.isfinite(border_depth) & (border_depth > 0) & (border_depth < 1e5)
                if bv.sum() > 0:
                    bpts_cam = border_depth[bv, None] * border_rays[bv]
                    bpts_world = (Rm @ bpts_cam.T).T + C
                    all_xy.append(bpts_world[:, :2])

                # ── 3a. depth-based corner projection (actual visible surface) ──
                c_u = np.clip(corner_px[:, 0].astype(int), 0, width - 1)
                c_v = np.clip(corner_px[:, 1].astype(int), 0, height - 1)
                c_depth = depth[c_v, c_u]

                for ci in range(4):
                    if np.isfinite(c_depth[ci]) and c_depth[ci] > 0 and c_depth[ci] < 1e5:
                        pt_cam = c_depth[ci] * corner_rays[ci]
                        pt_w = Rm @ pt_cam + C
                        frustum_xy.append(pt_w[:2])

                # Store per-frame camera params for pass 2 (ground-plane frustum)
                frame_info.append((C.copy(), Rm.copy()))

            # ── Compute global ground level from ALL depth back-projections ──
            # Use the median of all back-projected Z values as the ground
            # estimate.  This is more robust than per-frame median (which can
            # sit at roof height for oblique frames) and matches what the
            # visualization code uses (median of DSM Z).
            if all_z:
                all_z_arr = np.concatenate(all_z)
                ground_z = float(np.median(all_z_arr))
                print(f"    Global ground_z = {ground_z:.1f} "
                      f"(median of {len(all_z_arr)} pts, "
                      f"range [{all_z_arr.min():.1f}, {all_z_arr.max():.1f}])")
            else:
                # Fallback: use lowest camera position - 100m
                ground_z = float(df['z'].min()) - 100.0

            # ── PASS 2: frustum-ground projection using global ground_z ──
            # Project ALL 4 image corners onto the estimated ground plane.
            # Handles both ground-hitting rays (t > 0) and sky/horizon rays
            # (t <= 0) consistently with the visualization's
            # compute_frustum_corners().  XY displacement is capped to
            # MAX_XY_DISP metres so the bbox remains practical.
            MAX_XY_DISP = 2000.0
            for (C, Rm) in frame_info:
                for ci in range(4):
                    ray_w = Rm @ corner_rays[ci]
                    ray_norm = np.linalg.norm(ray_w)

                    if abs(ray_w[2]) > 1e-6:
                        t = (ground_z - C[2]) / ray_w[2]
                    else:
                        t = -1  # ray parallel to ground

                    if t > 0:
                        hit = C[:2] + t * ray_w[:2]
                    else:
                        # Sky/horizon ray: project MAX_XY_DISP along XY
                        # (same logic as compute_frustum_corners in viz)
                        d_unit = ray_w / ray_norm if ray_norm > 1e-12 else ray_w
                        d_xy = d_unit[:2]
                        d_xy_len = np.linalg.norm(d_xy)
                        if d_xy_len < 1e-8:
                            continue  # straight up/down
                        hit = C[:2] + (MAX_XY_DISP / d_xy_len) * d_xy

                    # Cap XY displacement from camera center
                    disp = hit - C[:2]
                    disp_len = np.linalg.norm(disp)
                    if disp_len > MAX_XY_DISP:
                        hit = C[:2] + (MAX_XY_DISP / disp_len) * disp

                    frustum_xy.append(hit)

            # ── combine all points ──
            if all_xy:
                all_pts = np.concatenate(all_xy, axis=0)

                # Robust percentile filtering (removes outlier depth glitches)
                lo, hi = 0.5, 99.5
                xlo, xhi = np.percentile(all_pts[:, 0], [lo, hi])
                ylo, yhi = np.percentile(all_pts[:, 1], [lo, hi])

                # Also include frustum corner points (guaranteed coverage)
                if frustum_xy:
                    frustum_arr = np.array(frustum_xy)
                    # Frustum points are NOT filtered – they represent the real
                    # camera FOV and must be inside the bbox.
                    xlo = min(xlo, frustum_arr[:, 0].min())
                    xhi = max(xhi, frustum_arr[:, 0].max())
                    ylo = min(ylo, frustum_arr[:, 1].min())
                    yhi = max(yhi, frustum_arr[:, 1].max())

                pad = margin_m + crop_padding_m
                bbox = (xlo - pad, ylo - pad,
                        xhi + pad, yhi + pad)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]

                # Clamp to max_dimension_m centred on camera trajectory midpoint
                if max_dimension_m is not None and (w > max_dimension_m or h > max_dimension_m):
                    cam_cx = (df['x'].min() + df['x'].max()) / 2
                    cam_cy = (df['y'].min() + df['y'].max()) / 2
                    half = max_dimension_m / 2
                    bbox = (max(bbox[0], cam_cx - half), max(bbox[1], cam_cy - half),
                            min(bbox[2], cam_cx + half), min(bbox[3], cam_cy + half))
                    w_new = bbox[2] - bbox[0]
                    h_new = bbox[3] - bbox[1]
                    print(f"    ⚠ Clamped bbox from {w:.0f}x{h:.0f}m to "
                          f"{w_new:.0f}x{h_new:.0f}m (max_dimension_m={max_dimension_m})")
                    w, h = w_new, h_new

                print(f"    Depth+frustum bbox: ({bbox[0]:.0f}, {bbox[1]:.0f}) to "
                      f"({bbox[2]:.0f}, {bbox[3]:.0f}) [{w:.0f}m x {h:.0f}m]  "
                      f"(margin={margin_m}m + crop_pad={crop_padding_m}m)")
                return bbox

    # Fallback: camera positions + margin
    pad = margin_m + crop_padding_m
    print(f"    Fallback: using camera positions ± {pad}m")
    min_x = df['x'].min() - pad
    max_x = df['x'].max() + pad
    min_y = df['y'].min() - pad
    max_y = df['y'].max() + pad

    return (min_x, min_y, max_x, max_y)



def get_cam_rays(K: np.ndarray, R_c2w: np.ndarray, t_w: np.ndarray, H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate camera rays for each pixel.
    
    Args:
        K: (3, 3) Intrinsic matrix
        R_c2w: (3, 3) Rotation matrix from camera-to-world
        t_w: (3,) Camera origin in world coordinates
        H, W: Image dimensions
        
    Returns:
        (H, W, 3) origins, (H, W, 3) directions
    """
    # Create coordinate grid
    grid_y, grid_x = np.mgrid[0:H, 0:W]
    
    # Inverse intrinsics to get normalized image coordinates
    # x = (u - cx) / fx, y = (v - cy) / fy
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x = (grid_x - cx) / fx
    y = (grid_y - cy) / fy
    z = np.ones_like(x)
    
    # Directions in camera frame
    dirs_c = np.stack([x, y, z], axis=-1) # (H, W, 3)
    
    # Transform to world frame (rotation only)
    # dirs_w = (R_c2w @ dirs_c.T).T
    dirs_w = dirs_c @ R_c2w.T
    
    # Normalize directions
    norms = np.linalg.norm(dirs_w, axis=-1, keepdims=True)
    dirs_w = dirs_w / norms
    
    # Origins are same for all rays (pinhole model)
    origins = np.broadcast_to(t_w, (H, W, 3))
    
    return origins.astype(np.float32), dirs_w.astype(np.float32)


def compute_z_offset_from_rendered_depth(
    depth_dir: Path, 
    poses_df,
    dsm_paths,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    origin_shift: Optional[np.ndarray] = None,
    num_samples: int = 5
) -> Optional[float]:
    """
    Compute the Z-offset between rendered mesh depth and DSM elevation.
    
    This offset accounts for the difference between the trajectory's altitude reference
    (e.g., Google Earth Studio internal terrain model) and the local DSM (Berlin DHHN2016
    orthometric heights).
    
    Both LoD mesh and DSM are from Berlin geoportals (same vertical datum), but the
    trajectory altitudes may use a different reference (e.g., ellipsoidal or Google's
    internal terrain model). This function computes the empirical offset.
    
    The offset is computed by:
    1. Loading rendered depth maps from a few frames
    2. Unprojecting depth to 3D world coordinates using camera pose
    3. Querying DSM at those (x, y) locations
    4. Computing median Z difference: rendered_z - dsm_z
    
    Args:
        depth_dir: Path to depth/ folder containing depth_NNNN.npz files
        poses_df: DataFrame with columns x, y, z, qw, qx, qy, qz
        dsm_paths: List of paths to DSM GeoTIFF tiles
        width, height: Frame dimensions
        fx, fy, cx, cy: Camera intrinsics
        origin_shift: Origin shift applied during rendering (3,) array
        num_samples: Number of frames to sample
        
    Returns:
        Median Z-offset in meters, or None if computation fails
    """
    from scipy.spatial.transform import Rotation
    
    if not depth_dir.exists():
        return None
        
    num_frames = len(poses_df)
    if num_frames == 0:
        return None
        
    # Sample frames evenly across sequence
    sample_indices = np.linspace(0, num_frames - 1, min(num_samples, num_frames), dtype=int)
    
    # Build rasterio sources for DSM lookup
    dsm_sources = []
    for dsm_p in dsm_paths:
        try:
            src = rasterio.open(dsm_p)
            dsm_sources.append(src)
        except Exception as e:
            print(f"   Warning: Could not open DSM {dsm_p}: {e}")
    
    if not dsm_sources:
        return None
    
    all_offsets = []
    
    for frame_idx in sample_indices:
        depth_path = depth_dir / f"depth_{frame_idx:04d}.npz"
        if not depth_path.exists():
            continue
            
        try:
            depth_data = np.load(depth_path)
            depth = depth_data['depth']  # (H, W) ray distances
        except Exception:
            continue
            
        # Get camera pose
        row = poses_df.iloc[frame_idx]
        cam_pos = np.array([row['x'], row['y'], row['z']])
        quat = np.array([row['qx'], row['qy'], row['qz'], row['qw']])
        R_c2w = Rotation.from_quat(quat).as_matrix()
        
        # Apply origin shift if provided (rendered mesh was shifted)
        if origin_shift is not None:
            cam_pos_local = cam_pos - origin_shift
        else:
            cam_pos_local = cam_pos
        
        # Sample valid depth pixels (avoid edges, sample grid)
        valid_mask = (depth > 0) & (depth < 10000)
        if not np.any(valid_mask):
            continue
            
        # Sample on a coarse grid to speed up
        step = 50
        sample_rows = np.arange(step, height - step, step)
        sample_cols = np.arange(step, width - step, step)
        
        for y in sample_rows:
            for x in sample_cols:
                if not valid_mask[y, x]:
                    continue
                    
                d = depth[y, x]
                
                # Unproject to camera frame
                x_norm = (x - cx) / fx
                y_norm = (y - cy) / fy
                dir_cam = np.array([x_norm, y_norm, 1.0])
                dir_cam = dir_cam / np.linalg.norm(dir_cam)
                
                # Transform to world frame
                dir_world = R_c2w @ dir_cam
                
                # 3D point in local coordinates (origin-shifted)
                pt_local = cam_pos_local + d * dir_world
                
                # 3D point in world coordinates
                if origin_shift is not None:
                    pt_world = pt_local + origin_shift
                else:
                    pt_world = pt_local
                
                # Query DSM at (x, y) location
                dsm_z = None
                for src in dsm_sources:
                    try:
                        if (src.bounds.left <= pt_world[0] <= src.bounds.right and
                            src.bounds.bottom <= pt_world[1] <= src.bounds.top):
                            row_idx, col_idx = src.index(pt_world[0], pt_world[1])
                            if 0 <= row_idx < src.height and 0 <= col_idx < src.width:
                                dsm_val = src.read(1, window=((row_idx, row_idx+1), (col_idx, col_idx+1)))[0, 0]
                                if dsm_val > -1000:  # Valid DSM value
                                    dsm_z = dsm_val
                                    break
                    except:
                        continue
                
                if dsm_z is not None:
                    offset = pt_world[2] - dsm_z
                    all_offsets.append(offset)
    
    # Close DSM sources
    for src in dsm_sources:
        src.close()
    
    if not all_offsets:
        return None
    
    # Compute median offset
    median_offset = float(np.median(all_offsets))
    std_offset = float(np.std(all_offsets))
    
    print(f"   Z-offset calibration: {median_offset:.2f}m (std: {std_offset:.2f}m, n={len(all_offsets)})")
    
    # Sanity check: offset should be reasonable (< 100m)
    if abs(median_offset) > 100:
        print(f"   Warning: Z-offset seems too large ({median_offset:.2f}m), not applying")
        return None
    
    return median_offset
