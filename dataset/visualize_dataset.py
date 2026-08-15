#!/usr/bin/env python3
"""Visualize MovingDrone dataset samples: query frame, DOP orthophoto, DSM, and LoD2 overlays."""
import os
import sys
import math
import argparse
import random
import multiprocessing as mp

from pathlib import Path
import threading
import concurrent.futures

# Lock to serialize matplotlib figure creation/save (not thread-safe)
_mpl_lock = threading.Lock()

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — faster rendering
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon as MplPolygon, Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_closing
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torchvision.transforms.functional as TF
import pandas as pd

from datasets import MovingDrone, MovingDrone
from utils.image import unnormalize_image
from utils.tensor_ops import denorm
from utils.pose import get_camera_center_from_w2c, quat_pos_to_w2c
from utils.geo import get_visible_footprint, dsm_to_xyz
from utils.lod import mesh_to_polygons
from utils.lod import get_lod_polygons_cached, draw_lod_polygons
from utils.geo import world_to_dop_px, compute_frustum_corners, compute_camera_triangle, compute_visible_dop_mask
from utils.depth import compute_camera_space_normals, compute_surface_normals_from_pointmap
from utils.tensor_ops import _torch_dict_to_numpy, _numpy_dict_to_torch


# ── Per-sequence polygon cache (keyed by lod2.npz path string) ──────────────
_lod_polygon_cache = {}














def init_persistent_matches(dsm, geodata_mask, num_matches):
    """Initialise empty persistent match slots.

    Actual sampling is deferred to update_matches_for_frame which has
    the per-frame visible DOP mask."""
    return [{'world_xyz': np.zeros(3), 'dop_px': (0, 0), 'active': False}
            for _ in range(num_matches)]


def update_matches_for_frame(persistent, w2c, K, img_h, img_w, dsm, geodata_mask,
                             visible_dop=None):
    """Project persistent matches into the current UAV frame.

    A match is kept only when its DOP pixel falls inside the *visible*
    region (co-visible from the current UAV viewpoint).  Hidden or
    out-of-frame points are replaced by new samples drawn exclusively
    from the visible DOP region.

    Args:
        persistent: list of persistent match dicts (modified in-place)
        w2c: (3, 4) or (4, 4) numpy W2C matrix
        K: (3, 3) numpy intrinsics
        img_h, img_w: UAV image dimensions
        dsm: (3, H, W) tensor — raw XYZ
        geodata_mask: (1, H, W) tensor
        visible_dop: optional (Hd, Wd) bool mask — co-visible DOP pixels

    Returns:
        (persistent, frame_matches)"""
    K3 = K[:3, :3]
    frame_matches = []

    # Build sampling mask: intersection of geodata + visible region
    mask_np = geodata_mask[0].cpu().numpy() > 0.5
    if visible_dop is not None:
        sample_mask = mask_np & visible_dop
    else:
        sample_mask = mask_np
    val_idx = np.where(sample_mask)

    for i, pm in enumerate(persistent):
        needs_replace = False
        if not pm['active']:
            needs_replace = True
        else:
            # Check if the DOP pixel is still in the visible region
            du, dv = pm['dop_px']
            Hd, Wd = mask_np.shape
            if visible_dop is not None:
                if not (0 <= dv < Hd and 0 <= du < Wd and visible_dop[dv, du]):
                    needs_replace = True

            # Also check if point projects into current UAV frame
            if not needs_replace:
                xyz_h = np.append(pm['world_xyz'], 1.0)
                p_cam = w2c @ xyz_h
                if p_cam[2] <= 0:
                    needs_replace = True
                else:
                    uv = K3 @ p_cam[:3]
                    uv = uv[:2] / uv[2]
                    u_px, v_px = float(uv[0]), float(uv[1])
                    if not (0 <= u_px < img_w and 0 <= v_px < img_h):
                        needs_replace = True
                    else:
                        frame_matches.append(
                            {'uav': (u_px, v_px),
                             'dop': (float(du), float(dv))})

        if needs_replace:
            persistent[i] = _sample_replacement(val_idx, dsm, w2c, K3, img_h, img_w)
            if persistent[i]['active']:
                p_cam2 = w2c @ np.append(persistent[i]['world_xyz'], 1.0)
                uv2 = K3 @ p_cam2[:3]
                uv2 = uv2[:2] / uv2[2]
                frame_matches.append(
                    {'uav': (float(uv2[0]), float(uv2[1])),
                     'dop': (float(persistent[i]['dop_px'][0]),
                             float(persistent[i]['dop_px'][1]))})

    return persistent, frame_matches


def _sample_replacement(val_idx, dsm, w2c, K3, img_h, img_w, max_tries=50):
    """Try to sample a replacement match point visible in the current UAV frame.

    Samples only from val_idx which should already be restricted to the
    visible DOP region."""
    if len(val_idx[0]) == 0:
        return {'world_xyz': np.zeros(3), 'dop_px': (0, 0), 'active': False}

    for _ in range(max_tries):
        c = np.random.randint(len(val_idx[0]))
        v, u = val_idx[0][c], val_idx[1][c]
        xyz = dsm[:, v, u].cpu().numpy()
        xyz_h = np.append(xyz, 1.0)
        p_cam = w2c @ xyz_h
        if p_cam[2] <= 0:
            continue
        uv = K3 @ p_cam[:3]
        uv = uv[:2] / uv[2]
        if 0 <= uv[0] < img_w and 0 <= uv[1] < img_h:
            return {'world_xyz': xyz, 'dop_px': (int(u), int(v)), 'active': True}

    return {'world_xyz': np.zeros(3), 'dop_px': (0, 0), 'active': False}









def visualize_sample(sample, output_dir, idx, normalization='01', num_matches=5, augment=False,
                     vis_lod=True, vis_semantics=True, vis_matches=True, vis_lidar=True,
                     vis_point_map=True, vis_depth_map=True, vis_normals=True, vis_dop=True, vis_dsm=True,
                     vis_matching_masks=True,
                     frame_info=None, frustum_corners=None, trajectory_history=None,
                     camera_triangle=None, cmap_ranges=None, dpi=150, paper=False, fig_ext=None):
    """Visualize a sample matching the ortholoc visualization style.
    
    Args:
        cmap_ranges: Optional dict with keys 'point_map', 'depth', 'dsm' mapping to (vmin, vmax) tuples
                     for consistent colormap ranges across frames. If None, computed per-frame."""
    
    # Unpack global colormap ranges if provided (for consistent visualization across frames)
    if cmap_ranges is None:
        cmap_ranges = {}
    
    # Layout Parameters
    if paper:
        WSPACE = 0.02
        CB_WIDTH_RATIO = 0.02
        TITLE_Y = 1.0
        TOP_MARGIN = 0.98
        BOTTOM_MARGIN = 0.01
    else:
        WSPACE = 0.10
        CB_WIDTH_RATIO = 0.03
        TITLE_Y = 0.98
        TOP_MARGIN = 0.93
        BOTTOM_MARGIN = 0.03

    # Extract Masks
    q_mask = sample.get('query_mask')
    geo_mask = sample.get('geodata_mask')
    
    def to_np_mask(m):
        if m is None: return None
        if torch.is_tensor(m): m = m.cpu().numpy()
        if m.ndim == 3: return m[0] > 0.5
        return m > 0.5

    q_mask_np = to_np_mask(q_mask)
    geo_mask_np = to_np_mask(geo_mask)

    # 1. Unnormalize Images
    img_uav = unnormalize_image(sample['query'], normalization)
    img_dop = unnormalize_image(sample['dop'], normalization)

    # 2. Mask Images (white for invalid)
    if q_mask_np is not None:
        img_uav = img_uav.copy()
        img_uav[~q_mask_np] = [255, 255, 255]
    if geo_mask_np is not None:
        img_dop = img_dop.copy()
        img_dop[~geo_mask_np] = [255, 255, 255]

    def mask_array(arr, mask):
        if arr is None or mask is None: return arr
        arr = arr.astype(np.float32).copy()
        arr[~mask] = np.nan
        return arr

    # 3. Extract maps and denormalize to raw meters
    dsm = sample.get('dsm')
    point_map = sample.get('point_map')
    depth = sample.get('depth')

    norm_type = sample.get('norm_type', 'none')
    dsm_scale = sample.get('dsm_scale')
    dsm_offset = sample.get('dsm_offset')

    # Denormalize full tensors to raw meters using the proper inverse for each norm_type
    # If _dsm_is_raw is set, the DSM is already in raw meters (e.g. fixed-crop override).
    if sample.get('_dsm_is_raw', False):
        dsm_raw = dsm
    elif dsm is not None and norm_type not in (None, 'none', 'None'):
        dsm_denorm = denorm(dsm, dsm_scale, dsm_offset, norm_type)
        dsm_raw = torch.from_numpy(dsm_denorm) if not torch.is_tensor(dsm_denorm) else dsm_denorm
    else:
        dsm_raw = dsm
    if point_map is not None and norm_type not in (None, 'none', 'None'):
        pm_denorm = denorm(point_map, dsm_scale, dsm_offset, norm_type)
        pm_raw = torch.from_numpy(pm_denorm) if not torch.is_tensor(pm_denorm) else pm_denorm
    else:
        pm_raw = point_map
    if depth is not None and norm_type not in (None, 'none', 'None'):
        if isinstance(dsm_scale, (torch.Tensor, np.ndarray)) and (
            (torch.is_tensor(dsm_scale) and dsm_scale.numel() >= 3) or 
            (isinstance(dsm_scale, np.ndarray) and dsm_scale.size >= 3)
        ):
            depth_scale_z = dsm_scale[2]
            depth_offset_z = dsm_offset[2]
        else:
            depth_scale_z = dsm_scale
            depth_offset_z = dsm_offset
        
        depth_denorm = denorm(depth, depth_scale_z, depth_offset_z, norm_type)
        depth_raw = torch.from_numpy(depth_denorm) if not torch.is_tensor(depth_denorm) else depth_denorm
    else:
        depth_raw = depth

    # Extract XYZ scale/offset for colorbar dual-labels and spatial axes
    sc_xyz = [1.0, 1.0, 1.0]
    off_xyz = [0.0, 0.0, 0.0]
    for arr, dst in [(dsm_scale, sc_xyz), (dsm_offset, off_xyz)]:
        if arr is not None:
            for i in range(3):
                if isinstance(arr, torch.Tensor):
                    dst[i] = arr[i].item() if arr.numel() >= 3 else arr.item()
                elif isinstance(arr, (list, np.ndarray)):
                    dst[i] = float(arr[i]) if len(arr) >= 3 else float(arr[0] if len(arr) > 0 else 0)
                else:
                    dst[i] = float(arr)
    sc_x, sc_y, sc_z = sc_xyz
    off_x, off_y, off_z = off_xyz

    dsm_data = mask_array(dsm_raw.cpu().numpy()[2] if dsm_raw is not None else None, geo_mask_np)
    point_map_data = mask_array(pm_raw.cpu().numpy()[2] if pm_raw is not None else None, q_mask_np)
    depth_data = mask_array(depth_raw.cpu().numpy()[0] if depth_raw is not None else None, q_mask_np)

    # Extract normals for visualization — compute from depth in camera space
    # (produces much cleaner normals than world-space point map differentiation)
    # IMPORTANT: use the *resized* intrinsics that match the depth resolution,
    # not intrinsics_raw which is for the full-resolution image (1920x1080).
    normals_data = None
    if vis_normals and depth_raw is not None:
        K_resized = sample.get('intrinsics', sample.get('intrinsics_raw'))
        if K_resized is not None:
            normals_cam_hw3 = compute_camera_space_normals(depth_raw, K_resized)  # (H, W, 3)
            # Standard RGB normal mapping: [-1, 1] -> [0, 1]
            normals_rgb = (normals_cam_hw3 * 0.5 + 0.5).clip(0, 1)
            # Mask out zero-normal pixels (no hit / boundary)
            hit_mask = np.any(normals_cam_hw3 != 0, axis=-1)
            if q_mask_np is not None:
                hit_mask = hit_mask & q_mask_np
            normals_rgb[~hit_mask] = 1.0  # white for invalid
            normals_data = normals_rgb

    # 4. LoD overlays (Unified Mesh)
    lod_vertices = sample.get('lod_vertices', np.zeros((0, 3)))
    lod_faces = sample.get('lod_faces', np.zeros((0, 3), dtype=int))
    lod_labels = sample.get('lod_labels', np.array([]))
    lod_uav = sample.get('lod_uav', np.zeros((0, 2)))
    lod_dop = sample.get('lod_dop', np.zeros((0, 2)))
    
    # Define colors for surface types (RGB)
    lod_colors = {
        'roof': (255, 0, 0),        # Red
        'wall': (0, 0, 255),        # Blue
        'wall_interior': (100, 100, 255), # Light Blue
        'ground': (0, 255, 0),      # Green
        'closure': (255, 225, 0),   # Yellow
        'ceiling': (255, 0, 255),   # Magenta
        'ceiling_outer': (255, 150, 255), # Pink
        'floor': (139, 69, 19),     # SaddleBrown
        'floor_outer': (210, 105, 30), # Chocolate
        'door': (165, 42, 42),      # Brown
        'window': (0, 255, 255),    # Cyan
        'other': (210, 180, 140),   # Beige
        'building': (255, 165, 0)   # Orange
    }

    def draw_lod_mesh(img, vertices_3d, vertices_proj, faces, labels,
                      is_elevation=False, alpha=0.4, use_semantics=True, view_dir=None):
        """Render LoD mesh with painter's algorithm and proper back-face culling.
        
        For perspective (UAV) views (view_dir provided):
          - Back-facing triangles are culled from the fill pass (no purple/blue walls)
          - Only silhouette edges are drawn (building outlines only, no interior wireframe)
          - Faces with extreme projections near the horizon are clipped
        For top-down (DOP/elevation) views:
          - All faces rendered as before
          - Boundary + semantic + crease edges drawn"""
        if len(faces) == 0 or len(vertices_proj) == 0: 
            return
        
        # Ensure numpy
        if torch.is_tensor(vertices_proj):
            vertices_proj = vertices_proj.cpu().numpy()
        if torch.is_tensor(faces):
            faces = faces.cpu().numpy()
        if torch.is_tensor(vertices_3d):
            vertices_3d = vertices_3d.cpu().numpy()
        
        h, w = img.shape[:2]
        n_faces = len(faces)
        
        # --- Compute face normals (needed for both culling and edge detection) ---
        f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
        v_a = vertices_3d[f0]; v_b = vertices_3d[f1]; v_c = vertices_3d[f2]
        fnormals = np.cross(v_b - v_a, v_c - v_a)
        fnorms = np.linalg.norm(fnormals, axis=1, keepdims=True)
        valid_n = fnorms.ravel() > 1e-6
        fnormals[valid_n] /= fnorms[valid_n]
        fnormals[~valid_n] = 0
        
        # --- Compute front-facing mask for perspective views ---
        if view_dir is not None:
            face_dots = np.einsum('ij,j->i', fnormals, view_dir)  # (n_faces,)
            front_facing = face_dots < 0  # face normal points toward camera
        else:
            front_facing = np.ones(n_faces, dtype=bool)  # all front-facing for top-down
        
        # --- Vectorized face data extraction ---
        face_verts_2d = vertices_proj[faces, :2]  # (N, 3, 2)
        face_verts_z = vertices_proj[faces, 2] if vertices_proj.shape[1] > 2 else np.zeros((n_faces, 3))
        face_mean_z = face_verts_z.mean(axis=1)  # (N,)
        
        # --- Vectorized frustum culling ---
        face_min_xy = face_verts_2d.min(axis=1)  # (N, 2)
        face_max_xy = face_verts_2d.max(axis=1)
        # For perspective views, use tighter margin to clip faces near the horizon
        # that project to extreme coordinates at the top/bottom of the image
        if not is_elevation:
            margin = 50
            in_front = face_verts_z.min(axis=1) >= 0.1
            # Reject faces whose 2D bounding box spans more than 3x the image
            # (these are near-horizon faces that stretch wildly)
            face_span_x = face_max_xy[:, 0] - face_min_xy[:, 0]
            face_span_y = face_max_xy[:, 1] - face_min_xy[:, 1]
            reasonable_size = (face_span_x < w * 3) & (face_span_y < h * 3)
        else:
            margin = 100
            in_front = np.ones(n_faces, dtype=bool)
            reasonable_size = np.ones(n_faces, dtype=bool)
        
        in_bounds = ((face_max_xy[:, 0] > -margin) & (face_min_xy[:, 0] < w + margin) &
                     (face_max_xy[:, 1] > -margin) & (face_min_xy[:, 1] < h + margin))
        
        # For fill pass: cull back-facing faces in perspective view
        visible = in_bounds & in_front & front_facing & reasonable_size
        
        visible_idx = np.where(visible)[0]
        
        SUBPIXEL_SHIFT = 4
        SUBPIXEL_SCALE = 1 << SUBPIXEL_SHIFT
        ZB_SCALE = 4  # downsample factor for Z-buffer
        zb_h = max(1, h // ZB_SCALE)
        zb_w = max(1, w // ZB_SCALE)
        z_buffer_decoded = None
        z_range = 1.0  # will be updated if z-buffer is built

        # --- Sort visible faces: painter's algorithm (back-to-front) ---
        if len(visible_idx) > 0:
            vis_depths = face_mean_z[visible_idx]
            if is_elevation:
                sort_order = np.lexsort((visible_idx, vis_depths))
            else:
                sort_order = np.lexsort((visible_idx, -vis_depths))
            sorted_idx = visible_idx[sort_order]
            
            # --- Render faces: single fillPoly per color group (maintains painter's order) ---
            overlay = img.copy()
            
            if use_semantics:
                labels_arr = np.array(labels[:n_faces]) if len(labels) >= n_faces else np.pad(
                    np.array(labels), (0, n_faces - len(labels)), constant_values='other')
                sorted_labels = labels_arr[sorted_idx]
                sorted_verts = face_verts_2d[sorted_idx]  # (M, 3, 2)
                
                # Iterate in painter's order but batch consecutive same-color runs
                i = 0
                n_sorted = len(sorted_idx)
                while i < n_sorted:
                    cur_label = sorted_labels[i]
                    j = i + 1
                    while j < n_sorted and sorted_labels[j] == cur_label:
                        j += 1
                    color_bgr = lod_colors.get(cur_label, lod_colors['other'])
                    pts_batch = [(sorted_verts[k] * SUBPIXEL_SCALE).astype(np.int32) for k in range(i, j)]
                    cv2.fillPoly(overlay, pts_batch, color_bgr, shift=SUBPIXEL_SHIFT)
                    i = j
            else:
                color_bgr = (128, 128, 128)
                pts_batch = [(face_verts_2d[fi] * SUBPIXEL_SCALE).astype(np.int32) for fi in sorted_idx]
                cv2.fillPoly(overlay, pts_batch, color_bgr, shift=SUBPIXEL_SHIFT)
            
            # Blend shaded overlay
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            # --- Build Z-buffer for depth-aware edge occlusion testing ---
            # Use a downsampled Z-buffer (1/4 resolution) for speed.
            # Strategy: render faces FRONT-TO-BACK into a uint8 image where higher values
            # (=closer faces, smaller camera-Z) never overwrite lower values.
            # We then decode back to camera-Z for the edge visibility test.
            if not is_elevation and view_dir is not None:
                # Encode face mean_z as uint8: map [z_min, z_max] -> [255, 1]
                # (higher uint8 = closer = smaller Z, so we paint closest last via MAX)
                z_all = face_mean_z[sorted_idx]
                z_min_all = z_all.min() if len(z_all) > 0 else 0.0
                z_max_all = z_all.max() if len(z_all) > 0 else 1.0
                z_range = max(z_max_all - z_min_all, 1e-6)
                # Build Z-buffer: group faces into depth bands, render each band,
                # accumulate via element-wise maximum (closer = higher encoded value).
                # Split visible faces into N_BANDS bands by depth for batch rendering.
                N_BANDS = 32
                z_buffer_f32 = np.zeros((zb_h, zb_w), dtype=np.float32)
                band_edges = np.linspace(0, len(sorted_idx), N_BANDS + 1, dtype=int)
                for b in range(N_BANDS):
                    bstart, bend = band_edges[b], band_edges[b + 1]
                    if bstart >= bend:
                        continue
                    band_idx = sorted_idx[bstart:bend]
                    # Use the minimum (closest) Z of this band as the encoded value
                    # → within the band, all faces get the same representative value.
                    # This is a conservative overestimate (marks pixels as "close enough").
                    z_band = face_mean_z[band_idx].min()
                    val = int(np.clip(254 * (1.0 - (z_band - z_min_all) / z_range) + 1, 1, 255))
                    band_pts = [(face_verts_2d[fi] / ZB_SCALE * SUBPIXEL_SCALE).astype(np.int32)
                                for fi in band_idx]
                    tmp = np.zeros((zb_h, zb_w), dtype=np.uint8)
                    cv2.fillPoly(tmp, band_pts, val, shift=SUBPIXEL_SHIFT)
                    z_buffer_f32 = np.maximum(z_buffer_f32, tmp.astype(np.float32))
                # Decode: val_encoded = 1 + 254*(1 - (z - z_min)/z_range)
                # => z = z_min + (1 - (val-1)/254) * z_range
                z_buffer_decoded = np.where(
                    z_buffer_f32 > 0,
                    z_min_all + (1.0 - (z_buffer_f32 - 1.0) / 254.0) * z_range,
                    np.inf
                )
        
        # --- Feature edge detection (fully vectorized) + batch drawing ---
        all_e = np.concatenate([
            np.stack([f0, f1], axis=1),
            np.stack([f1, f2], axis=1),
            np.stack([f2, f0], axis=1),
        ], axis=0)  # (3N, 2)
        all_e_canonical = np.sort(all_e, axis=1)
        face_ids = np.tile(np.arange(n_faces), 3)

        labels_arr_full = np.array(labels[:n_faces]) if len(labels) >= n_faces else np.pad(
            np.array(labels), (0, n_faces - len(labels)), constant_values='other')
        
        max_vert = int(all_e_canonical.max()) + 1
        edge_keys = all_e_canonical[:, 0].astype(np.int64) * max_vert + all_e_canonical[:, 1].astype(np.int64)
        sort_ek = np.argsort(edge_keys)
        edge_keys_sorted = edge_keys[sort_ek]
        face_ids_sorted = face_ids[sort_ek]
        all_e_sorted = all_e_canonical[sort_ek]
        
        splits = np.where(np.diff(edge_keys_sorted) != 0)[0] + 1
        group_starts = np.concatenate([[0], splits])
        group_ends = np.concatenate([splits, [len(edge_keys_sorted)]])
        group_sizes = group_ends - group_starts
        
        if view_dir is not None:
            # --- Perspective view: silhouette edges only ---
            # Silhouette = edge between one front-facing and one back-facing face
            # This gives clean building outlines without interior wireframe clutter
            
            # 1) Boundary edges: only draw if the single adjacent face is front-facing
            boundary_groups = np.where(group_sizes == 1)[0]
            if len(boundary_groups) > 0:
                boundary_face_ids = face_ids_sorted[group_starts[boundary_groups]]
                boundary_front = front_facing[boundary_face_ids]
                boundary_mask = np.zeros(len(group_starts), dtype=bool)
                boundary_mask[boundary_groups[boundary_front]] = True
            else:
                boundary_mask = np.zeros(len(group_starts), dtype=bool)
            
            # 2) Multi-face edges: silhouette = one front + one back
            multi_mask = group_sizes >= 2
            multi_idx = np.where(multi_mask)[0]
            silhouette_multi = np.zeros(len(multi_idx), dtype=bool)
            
            if len(multi_idx) > 0:
                gs_multi = group_starts[multi_idx]
                fa = face_ids_sorted[gs_multi]
                fb = face_ids_sorted[gs_multi + 1]
                
                # Silhouette: one face front-facing, the other back-facing
                silhouette_multi = front_facing[fa] != front_facing[fb]
            
            feature_edge_mask = boundary_mask.copy()
            if len(multi_idx) > 0:
                feature_edge_mask[multi_idx[silhouette_multi]] = True
        else:
            # --- Top-down (DOP) view: boundary + semantic + crease edges ---
            boundary_mask = group_sizes == 1
            
            multi_mask = group_sizes >= 2
            multi_idx = np.where(multi_mask)[0]
            feature_multi = np.zeros(len(multi_idx), dtype=bool)
            
            if len(multi_idx) > 0:
                gs_multi = group_starts[multi_idx]
                fa = face_ids_sorted[gs_multi]
                fb = face_ids_sorted[gs_multi + 1]
                
                semantic_diff = labels_arr_full[fa] != labels_arr_full[fb]
                
                both_valid = valid_n[fa] & valid_n[fb]
                dots = np.abs(np.einsum('ij,ij->i', fnormals[fa], fnormals[fb]))
                crease = both_valid & (dots < 0.70)
                
                feature_multi = semantic_diff | crease
            
            feature_edge_mask = np.zeros(len(group_starts), dtype=bool)
            feature_edge_mask[boundary_mask] = True
            feature_edge_mask[multi_idx[feature_multi]] = True
        
        feature_gs = group_starts[feature_edge_mask]
        edge_v1 = all_e_sorted[feature_gs, 0]
        edge_v2 = all_e_sorted[feature_gs, 1]
        
        # Batch draw feature edges
        if len(edge_v1) > 0:
            w_color = (0, 255, 0)
            p1s = vertices_proj[edge_v1, :2]  # (E, 2)
            p2s = vertices_proj[edge_v2, :2]  # (E, 2)
            # Clip edges to a reasonable range around the image
            clip_margin = max(w, h) * 1.5
            valid_edges = ((np.abs(p1s[:, 0]) < clip_margin) & (np.abs(p1s[:, 1]) < clip_margin) &
                           (np.abs(p2s[:, 0]) < clip_margin) & (np.abs(p2s[:, 1]) < clip_margin))
            # Also reject edges where either vertex is behind camera
            if vertices_proj.shape[1] > 2:
                z1 = vertices_proj[edge_v1, 2]
                z2 = vertices_proj[edge_v2, 2]
                if not is_elevation:
                    valid_edges = valid_edges & (z1 >= 0.1) & (z2 >= 0.1)
            
            # --- Depth-aware occlusion test using the Z-buffer ---
            # An edge is occluded if both its endpoints are significantly BEHIND
            # the closest face already drawn at those pixels.
            # tolerance: edges within 1 Z-unit of the closest face are considered visible
            # (accounts for Z-buffer quantization and face thickness).
            if z_buffer_decoded is not None and vertices_proj.shape[1] > 2:
                Z_TOLERANCE = (z_range * 0.05) + 0.5  # 5% of depth range + 0.5 units
                z1_all = vertices_proj[edge_v1, 2]
                z2_all = vertices_proj[edge_v2, 2]

                def _zbuf_lookup(px, py):
                    """Look up Z-buffer at pixel coords, clamped to buffer bounds."""
                    bx = np.clip((px / ZB_SCALE).astype(int), 0, zb_w - 1)
                    by = np.clip((py / ZB_SCALE).astype(int), 0, zb_h - 1)
                    return z_buffer_decoded[by, bx]

                # Both endpoints must pass the visibility test (z ≤ zbuf + tol)
                zb1 = _zbuf_lookup(p1s[:, 0], p1s[:, 1])
                zb2 = _zbuf_lookup(p2s[:, 0], p2s[:, 1])
                # Vertex is visible if its camera-Z is not significantly larger than zbuf
                ep1_visible = z1_all <= zb1 + Z_TOLERANCE
                ep2_visible = z2_all <= zb2 + Z_TOLERANCE
                # Keep edge if at least one endpoint is visible
                # (avoids hiding edges at depth discontinuities)
                valid_edges = valid_edges & (ep1_visible | ep2_visible)

            p1s = p1s[valid_edges]
            p2s = p2s[valid_edges]
            if len(p1s) > 0:
                edge_pts = np.stack([p1s, p2s], axis=1)  # (E, 2, 2)
                edge_pts_sub = (edge_pts * SUBPIXEL_SCALE).astype(np.int32)
                cv2.polylines(img, edge_pts_sub, isClosed=False, color=w_color, thickness=1, shift=SUBPIXEL_SHIFT)

    sample_id = sample.get('sample_id', str(idx))
    # print(f"Sample {sample_id}: LoD level {sample.get('lod_level')}, Vertices {len(lod_vertices)}, Faces {len(lod_faces)}")
    
    img_uav_vis = img_uav.copy()
    img_dop_vis = img_dop.copy()
    
    # Compute camera view direction for silhouette-based LoD edge rendering
    w2c_raw = sample.get('extrinsics_raw')
    if w2c_raw is not None:
        if torch.is_tensor(w2c_raw):
            R_cam = w2c_raw[:3, :3].cpu().numpy()
        else:
            R_cam = w2c_raw[:3, :3]
        # Camera looks along +Z in camera frame → world view direction = R^T @ [0,0,1]
        cam_view_dir = R_cam.T @ np.array([0.0, 0.0, 1.0])
    else:
        cam_view_dir = None

    if vis_lod:
        # Reconstruct original polygons from the triangulated mesh.
        # Use the sequence part of sample_id as cache key (mesh is per-sequence).
        seq_name = '_'.join(sample.get('sample_id', str(idx)).split('_')[:-1]) or str(idx)
        lod_polygons = get_lod_polygons_cached(
            np.asarray(lod_vertices), np.asarray(lod_faces), np.asarray(lod_labels),
            cache_key=seq_name
        ) if len(lod_faces) > 0 else []

        # Reproject lod_vertices onto the CURRENT frame's DOP crop.
        # The dataset caches lod_dop from the first frame that loaded the LoD, but
        # each single-frame load may have a different crop_bounds → stale projection
        # causes the LoD to appear to shift/flicker between frames.
        _crop_bounds = sample.get('crop_bounds')
        if lod_vertices is not None and len(lod_vertices) > 0 and _crop_bounds is not None:
            lod_verts_np = np.asarray(lod_vertices)
            cb_min_x, cb_min_y, cb_max_x, cb_max_y = _crop_bounds
            H_dop, W_dop = img_dop_vis.shape[:2]
            _du = (lod_verts_np[:, 0] - cb_min_x) / max(cb_max_x - cb_min_x, 1e-6) * W_dop
            _dv = (cb_max_y - lod_verts_np[:, 1]) / max(cb_max_y - cb_min_y, 1e-6) * H_dop
            lod_dop = np.stack([_du, _dv, lod_verts_np[:, 2]], axis=1).astype(np.float32)

        if lod_polygons:
            draw_lod_polygons(img_uav_vis, lod_vertices, lod_uav, lod_polygons,
                              alpha=0.2, use_semantics=vis_semantics,
                              view_dir=cam_view_dir)
            draw_lod_polygons(img_dop_vis, lod_vertices, lod_dop, lod_polygons,
                              is_elevation=True, alpha=0.2,
                              use_semantics=vis_semantics)
        else:
            draw_lod_mesh(img_uav_vis, lod_vertices, lod_uav, lod_faces, lod_labels, alpha=0.2, use_semantics=vis_semantics, view_dir=cam_view_dir)
            draw_lod_mesh(img_dop_vis, lod_vertices, lod_dop, lod_faces, lod_labels, is_elevation=True, alpha=0.2, use_semantics=vis_semantics)

    # 5. DOP Projection Params
    dop_proj_params = sample.get('dop_proj_params')
    if dop_proj_params is not None and isinstance(dop_proj_params, torch.Tensor):
        dop_proj_params = dop_proj_params.cpu().numpy()

    # 6. Matches — use pre-computed or generate from point_map
    matches = sample.get('matches', [])
    final_matches = list(matches) if matches else []
    
    # Compute co-visible mask: a pixel is co-visible (matchable to DOP) when it is
    # visible from a top-down viewpoint.  We use two complementary criteria and take
    # their UNION so that neither alone is a bottleneck:
    #
    #   A) DSM-consistent: point_map Z is close to the DSM Z at the same (X,Y).
    #      Catches ground, roads, building roofs that are in the DSM.
    #
    #   B) Horizontal normal: the surface normal points mostly upward (|nz| > 0.7).
    #      Catches objects present in the rendered scene but absent / poorly
    #      represented in the DSM (trees, vehicles, low structures) whose tops are
    #      still visible from the orthographic DOP viewpoint.
    #
    # Truly non-matchable pixels (building facades, vertical walls) satisfy NEITHER:
    # their Z is well above the DSM surface AND their normal points sideways.
    covisible_mask = None
    if pm_raw is not None:
        pm_np_for_mask = pm_raw.cpu().numpy() if torch.is_tensor(pm_raw) else pm_raw
        valid_pm_mask = np.any(pm_np_for_mask != 0, axis=0)

        # --- Criterion B: horizontal surface normal ---
        _, horizontal_mask = compute_surface_normals_from_pointmap(pm_raw)

        # --- Criterion A: DSM Z consistency ---
        dsm_consistent_mask = np.zeros_like(valid_pm_mask)  # default: unknown → not consistent
        crop_bounds = sample.get('crop_bounds')
        if dsm_raw is not None and crop_bounds is not None:
            dsm_np = dsm_raw.cpu().numpy() if torch.is_tensor(dsm_raw) else dsm_raw
            Hd, Wd = dsm_np.shape[1:]
            x_min, y_min, x_max, y_max = crop_bounds

            pm_x = pm_np_for_mask[0]
            pm_y = pm_np_for_mask[1]
            pm_z = pm_np_for_mask[2]

            dsm_u_float = (pm_x - x_min) / (x_max - x_min) * Wd
            dsm_v_float = (y_max - pm_y) / (y_max - y_min) * Hd
            in_dsm_bounds = (dsm_u_float >= 0) & (dsm_u_float < Wd) & (dsm_v_float >= 0) & (dsm_v_float < Hd)

            dsm_u = np.clip(dsm_u_float.astype(int), 0, Wd - 1)
            dsm_v = np.clip(dsm_v_float.astype(int), 0, Hd - 1)
            dsm_z_at_pm = dsm_np[2, dsm_v, dsm_u]

            z_diff_raw = pm_z - dsm_z_at_pm

            # Auto-detect Z offset from horizontal surfaces only (exclude facades)
            valid_for_offset = valid_pm_mask & in_dsm_bounds & horizontal_mask
            if np.sum(valid_for_offset) > 100:
                z_offset = np.median(z_diff_raw[valid_for_offset])
            elif np.sum(valid_pm_mask & in_dsm_bounds) > 100:
                z_offset = np.median(z_diff_raw[valid_pm_mask & in_dsm_bounds])
            else:
                z_offset = 0.0

            z_diff = z_diff_raw - z_offset

            # Accept if the point is on or near the DSM surface (not floating high above it)
            z_consistent = (z_diff < 3.0) & (z_diff > -2.0)

            # Exclude only hard building-wall edges (>8 m/px gradient, minimal dilation)
            dsm_z_channel = dsm_np[2]
            dsm_grad_x = np.abs(np.diff(dsm_z_channel, axis=1, prepend=dsm_z_channel[:, :1]))
            dsm_grad_y = np.abs(np.diff(dsm_z_channel, axis=0, prepend=dsm_z_channel[:1, :]))
            dsm_gradient = np.maximum(dsm_grad_x, dsm_grad_y)
            dsm_edge_mask = binary_dilation(dsm_gradient > 8.0, iterations=1)
            not_on_dsm_edge = ~dsm_edge_mask[dsm_v, dsm_u]

            dsm_consistent_mask = in_dsm_bounds & z_consistent & not_on_dsm_edge

        # Union: co-visible if DSM-consistent OR has upward-facing normal
        # This correctly handles objects the DSM doesn't resolve (trees, vehicles):
        # their tops face upward → co-visible; their sides face sideways → facade.
        covisible_mask = valid_pm_mask & (dsm_consistent_mask | horizontal_mask)

    if not final_matches and point_map is not None and num_matches > 0 and dop_proj_params is not None:
        # Sample from co-visible area (horizontal surfaces visible in both UAV and DOP)
        if covisible_mask is not None:
            sample_mask = covisible_mask
        elif q_mask_np is not None and geo_mask_np is not None:
            sample_mask = q_mask_np & geo_mask_np
        elif q_mask_np is not None:
            sample_mask = q_mask_np
        else:
            sample_mask = point_map[2].cpu().numpy() != 0
        val_idx = np.where(sample_mask)
        if len(val_idx[0]) > 0:
            cho = np.random.choice(len(val_idx[0]), min(len(val_idx[0]), num_matches), replace=False)
            sel_y, sel_x = val_idx[0][cho], val_idx[1][cho]
            pts_w = point_map[:, sel_y, sel_x].T
            kx, ky = pts_w[:, 0].numpy(), pts_w[:, 1].numpy()
            if len(dop_proj_params) == 6:
                a, b, c, d, e, f = dop_proj_params
                ud, vd = a * kx + b * ky + c, d * kx + e * ky + f
            else:
                su, ou, sv, ov = dop_proj_params
                ud, vd = kx * su + ou, ky * sv + ov
            for i in range(len(cho)):
                final_matches.append({'uav': (float(sel_x[i]), float(sel_y[i])),
                                      'dop': (float(ud[i]), float(vd[i]))})
    
    if not vis_matches:
        final_matches = []

    # 7. Dynamic Layout Calculation
    # Potential columns: Query, PtMap, Depth, LiDAR, DOP, DSM
    possible_cols = []
    
    # Always show Query (Col 0)
    possible_cols.append(('Query', True)) 
    
    # Check others
    show_ptmap = vis_point_map and point_map_data is not None and np.any(~np.isnan(point_map_data))
    possible_cols.append(('PtMap', show_ptmap))
    
    show_depth = vis_depth_map and depth_data is not None and np.any(~np.isnan(depth_data))
    possible_cols.append(('Depth', show_depth))
    
    show_normals = vis_normals and normals_data is not None
    possible_cols.append(('Normals', show_normals))
    
    lidar_pts = sample.get('lidar_uav')
    show_lidar = vis_lidar and lidar_pts is not None and len(lidar_pts) > 0
    possible_cols.append(('LiDAR', show_lidar))
    
    show_dop = vis_dop and img_dop_vis is not None
    possible_cols.append(('DOP', show_dop))
    
    show_dsm = vis_dsm and dsm_data is not None and np.any(~np.isnan(dsm_data))
    possible_cols.append(('DSM', show_dsm))
    
    active_cols = [name for name, active in possible_cols if active]
    num_cols = len(active_cols)
    col_mapping = {name: i for i, name in enumerate(active_cols)}

    # UAV row: Query, PtMap, Depth, Normals, LiDAR, UAV Mask
    uav_axes_names = ['Query', 'PtMap', 'Depth', 'Normals', 'LiDAR']
    if vis_matching_masks:
        uav_axes_names.append('UAVMask')
    uav_active = [n for n in uav_axes_names if col_mapping.get(n) is not None or n == 'UAVMask']
    
    # Geodata row: DOP, DSM, DOP Mask
    geo_axes_names = ['DOP', 'DSM']
    if vis_matching_masks:
        geo_axes_names.append('DOPMask')
    geo_active = [n for n in geo_axes_names if col_mapping.get(n) is not None or n == 'DOPMask']

    n_cols_uav = len(uav_active)
    n_cols_geo = len(geo_active)
    n_cols_max = max(n_cols_uav, n_cols_geo)

    # To avoid jitter when column counts differ between rows, we work in a
    # fine-grained GridSpec of width = LCM(n_cols_uav, n_cols_geo) so each
    # row can be centred and individual subplot widths stay uniform.
    def _lcm(a, b): return a * b // math.gcd(a, b)
    gs_cols = _lcm(n_cols_uav, n_cols_geo) if (n_cols_uav > 0 and n_cols_geo > 0) else max(n_cols_uav, n_cols_geo, 1)
    uav_span = gs_cols // n_cols_uav if n_cols_uav > 0 else gs_cols
    geo_span = gs_cols // n_cols_geo if n_cols_geo > 0 else gs_cols
    # Centre the shorter row
    uav_offset = (gs_cols - n_cols_uav * uav_span) // 2
    geo_offset = (gs_cols - n_cols_geo * geo_span) // 2

    if paper:
        fig = plt.figure(figsize=(3.2 * n_cols_max, 6.4))
        main_gs = GridSpec(2, gs_cols, wspace=WSPACE, hspace=0.04, figure=fig)
        fig.subplots_adjust(top=TOP_MARGIN, bottom=BOTTOM_MARGIN, left=0.01, right=0.99)
    else:
        fig = plt.figure(figsize=(4 * n_cols_max + 2, 8))
        main_gs = GridSpec(2, gs_cols, wspace=WSPACE, hspace=0.15, figure=fig)
        fig.subplots_adjust(top=TOP_MARGIN, bottom=BOTTOM_MARGIN, left=0.03, right=0.97)

    def prepare_ax(row_idx, col_list, name, title):
        if name not in col_list:
            return None
        pos = col_list.index(name)
        if row_idx == 0:
            col_start = uav_offset + pos * uav_span
            col_end   = col_start + uav_span
        else:
            col_start = geo_offset + pos * geo_span
            col_end   = col_start + geo_span
        ax = fig.add_subplot(main_gs[row_idx, col_start:col_end])
        if paper:
            ax.set_title(title, fontsize=7, pad=1)
        else:
            ax.set_title(title, fontsize=9, pad=2)
        ax.axis('off')
        return ax

    def add_cb(ax, im, vmin, vmax, scale, offset, label, norm_type='none'):
        """Colorbar showing metric values with dual labels (raw meters + normalized)."""
        if paper:
            # Compact colorbar: metric values only, smaller font
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.02)
            cbar = fig.colorbar(im, cax=cax)
            ticks = np.linspace(vmin, vmax, 3)
            cbar.set_ticks(ticks)
            cbar.ax.set_yticklabels([f"{t:.0f}" for t in ticks], fontsize=5)
            cbar.ax.tick_params(length=1, pad=1)
            return
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        ticks = np.linspace(vmin, vmax, 4)
        cbar.set_ticks(ticks)
        if norm_type not in (None, 'none', 'None') and scale != 0:
            def to_norm(t):
                if norm_type == 'minmax_11':
                    return 2 * (t - offset) / scale - 1
                else:  # mean_std or minmax_01
                    return (t - offset) / scale
            cbar.ax.set_yticklabels(
                [f"{t:.1f}m\n[{to_norm(t):.2f}]" for t in ticks],
                fontsize=5
            )
        else:
            cbar.ax.set_yticklabels(
                [f"{t:.1f}m" for t in ticks],
                fontsize=6
            )
        cbar.set_label(label, fontsize=7, labelpad=-6)

    def setup_spatial_axes(ax, crop_bounds, img_shape):
        """Add meter tick labels to DOP/DSM axes."""
        if paper or crop_bounds is None:
            return
        c_min_x, c_min_y, c_max_x, c_max_y = crop_bounds
        H, W = img_shape[:2]
        cx, cy = (c_min_x + c_max_x) / 2, (c_min_y + c_max_y) / 2

        ax.axis('on')
        ax.tick_params(labelsize=5, length=2, pad=1)

        n_ticks = 5
        x_px = np.linspace(0, W - 1, n_ticks)
        x_m = c_min_x + x_px / (W - 1) * (c_max_x - c_min_x)
        x_rel = x_m - cx

        y_px = np.linspace(0, H - 1, n_ticks)
        y_m = c_max_y - y_px / (H - 1) * (c_max_y - c_min_y)
        y_rel = y_m - cy

        x_labels = [f"{r:+.0f}m" for r in x_rel]
        y_labels = [f"{r:+.0f}m" for r in y_rel]

        ax.set_xticks(x_px)
        ax.set_xticklabels(x_labels, fontsize=4, rotation=30, ha='right')
        ax.set_yticks(y_px)
        ax.set_yticklabels(y_labels, fontsize=4)

    # Col 0: UAV Query (+ LoD) + Matches
    title0 = "UAV Query" + (" (+ LoD)" if vis_lod else "")
    ax0 = prepare_ax(0, uav_active, 'Query', title0)
    if ax0: ax0.imshow(img_uav_vis, aspect='equal')

    # Col 1: Point Map (Z)
    ax1 = prepare_ax(0, uav_active, 'PtMap', "Point Map (Z)")
    if ax1:
        if 'point_map' in cmap_ranges:
            v0, v1 = cmap_ranges['point_map']
        else:
            _pm_valid = point_map_data[np.isfinite(point_map_data) & (point_map_data != 0)]
            if len(_pm_valid) > 0:
                v0, v1 = float(np.percentile(_pm_valid, 1)), float(np.percentile(_pm_valid, 99))
            else:
                v0, v1 = np.nanmin(point_map_data), np.nanmax(point_map_data)
        im1 = ax1.imshow(point_map_data, cmap='turbo', vmin=v0, vmax=v1, aspect='equal')
        add_cb(ax1, im1, v0, v1, sc_z, off_z, "Z (m)", norm_type)

    # Col 2: Depth Map
    ax2 = prepare_ax(0, uav_active, 'Depth', "Depth Map")
    if ax2:
        if 'depth' in cmap_ranges:
            v0, v1 = cmap_ranges['depth']
        else:
            # Compute per-frame depth range (camera-space Z, NOT world elevation)
            # combine with lidar z_vals (also camera-space Z) for consistent range
            _depth_vals = []
            if depth_data is not None:
                _dv = depth_data[~np.isnan(depth_data)]
                if len(_dv) > 0:
                    _depth_vals.append(_dv)
            _lidar_z_for_range = sample.get('lidar_uav')
            if _lidar_z_for_range is not None and len(_lidar_z_for_range) > 0:
                if torch.is_tensor(_lidar_z_for_range):
                    _lidar_z_for_range = _lidar_z_for_range.cpu().numpy()
                if len(_lidar_z_for_range) > 0:
                    _depth_vals.append(_lidar_z_for_range[:, 2])
            if _depth_vals:
                _all_depth = np.concatenate(_depth_vals)
                _all_depth = _all_depth[np.isfinite(_all_depth)]
                if len(_all_depth) > 0:
                    v0, v1 = float(np.percentile(_all_depth, 2)), float(np.percentile(_all_depth, 98))
                else:
                    v0, v1 = 0, 1
            else:
                v0, v1 = 0, 1
        im2 = ax2.imshow(depth_data, cmap='inferno_r', vmin=v0, vmax=v1, aspect='equal')
        add_cb(ax2, im2, v0, v1, 1.0, 0.0, "Depth (m)", 'none')  # cam-space depth, no denorm

    # Normals Map (RGB mapped from surface normals)
    ax_normals = prepare_ax(0, uav_active, 'Normals', "Surface Normals")
    if ax_normals:
        ax_normals.imshow(normals_data, aspect='equal')

    # Col 3: LiDAR Visualization (next to Depth)
    ax3_lidar = prepare_ax(0, uav_active, 'LiDAR', "LiDAR on UAV")
    if ax3_lidar:
        ax3_lidar.imshow(img_uav, aspect='equal')
        h_uav, w_uav = img_uav.shape[:2]
        # Clip LiDAR points to image bounds to prevent scatter from extending axes
        in_bounds = ((lidar_pts[:, 0] >= 0) & (lidar_pts[:, 0] < w_uav) &
                     (lidar_pts[:, 1] >= 0) & (lidar_pts[:, 1] < h_uav))
        lidar_vis = lidar_pts[in_bounds]
        # Color code by camera-space depth using inferno_r (dark=close, bright=far)
        z_vals = lidar_vis[:, 2] if len(lidar_vis) > 0 else np.array([])
        # Use same range as depth map for consistent coloring;
        # combined depth+lidar range was computed above in the Depth section.
        if 'depth' in cmap_ranges:
            v0, v1 = cmap_ranges['depth']
        elif show_depth and depth_data is not None:
            # Share range with depth (already computed above)
            _all_z = [depth_data[~np.isnan(depth_data)]] if np.any(~np.isnan(depth_data)) else []
            if len(z_vals) > 0:
                _all_z.append(z_vals)
            if _all_z:
                _combined = np.concatenate(_all_z)
                _combined = _combined[np.isfinite(_combined)]
                if len(_combined) > 0:
                    v0, v1 = float(np.percentile(_combined, 2)), float(np.percentile(_combined, 98))
                else:
                    v0, v1 = 0, 1
            else:
                v0, v1 = 0, 1
        elif len(z_vals) > 0:
            z_finite = z_vals[np.isfinite(z_vals)]
            if len(z_finite) > 0:
                v0, v1 = float(np.percentile(z_finite, 2)), float(np.percentile(z_finite, 98))
            else:
                v0, v1 = 0, 1
        else:
            v0, v1 = 0, 1
        if len(lidar_vis) > 0:
            scatter = ax3_lidar.scatter(lidar_vis[:, 0], lidar_vis[:, 1], c=z_vals, s=1, cmap='inferno_r', vmin=v0, vmax=v1, alpha=0.5)
        else:
            scatter = ax3_lidar.scatter([], [], c=[], s=1, cmap='inferno_r', vmin=v0, vmax=v1)
        # Fix axes extent to image bounds (prevents jittering from out-of-bounds points)
        ax3_lidar.set_xlim(0, w_uav)
        ax3_lidar.set_ylim(h_uav, 0)  # Inverted Y to match image coordinates
        add_cb(ax3_lidar, scatter, v0, v1, 1.0, 0.0, "Depth (m)", 'none')  # cam-space depth

    # UAV Mask Consistency (Surface-based classification)
    ax_uav_mask = prepare_ax(0, uav_active, 'UAVMask', "UAV Mask (Yellow=Horiz, Red=Facade)")
    if ax_uav_mask:
        ax_uav_mask.imshow(img_uav, aspect='equal')
        
        # Prefer surface-normal based mask (more informative for occlusion visualization)
        # Only use RoMa matching masks if explicitly requested and available
        use_matching_masks = False  # Set to True to prefer RoMa matching masks when available
        m_mutual_q = sample.get('mask_query_in_dop') if use_matching_masks else None
        m_oneway_q = sample.get('mask_query_in_dop_oneway') if use_matching_masks else None
        
        if m_mutual_q is not None:
            H, W = m_mutual_q.shape[1:]
            overlay = np.zeros((H, W, 4))
            m_mutual_q_np = m_mutual_q[0].cpu().numpy()
            overlay[m_mutual_q_np > 0.5] = [1, 1, 0, 0.6] # Yellow, higher alpha
            
            num_mut = np.sum(m_mutual_q_np > 0.5)
            num_rej = 0
            if m_oneway_q is not None:
                m_oneway_q_np = m_oneway_q[0].cpu().numpy()
                # Rejected = (In UAV and projects to DOP footprint) BUT (Occluded or Cycle Mismatch)
                m_rejected_q = (m_oneway_q_np > 0.5) & (m_mutual_q_np < 0.5)
                num_rej = np.sum(m_rejected_q)
                overlay[m_rejected_q] = [1, 0, 0, 0.6] # Red for Rejections (Facades + Occlusions)
            
            print(f"  UAV Mask: {num_mut} mutual, {num_rej} rejected")
            ax_uav_mask.imshow(overlay)
        else:
            # Fallback: use pre-computed covisible_mask (includes DSM consistency check)
            # Yellow = co-visible surfaces (horizontal + DSM-consistent) - matchable to DOP
            # Red = non-matchable (facades, DSM-inconsistent) - NOT visible in top-down DOP
            # Transparent = invalid/no depth data
            if pm_raw is not None:
                H, W = pm_raw.shape[1:]
                overlay = np.zeros((H, W, 4))
                
                # Get point_map as numpy
                pm_np = pm_raw.cpu().numpy() if torch.is_tensor(pm_raw) else pm_raw
                
                # Valid mask: pixels where point_map has actual data (not all zeros)
                valid_mask = np.any(pm_np != 0, axis=0)  # (H, W)
                
                # Use covisible_mask if available (includes DSM consistency), else just horizontal
                if covisible_mask is not None:
                    matchable_mask = covisible_mask
                else:
                    _, horizontal_mask = compute_surface_normals_from_pointmap(pm_raw)
                    matchable_mask = valid_mask & horizontal_mask
                
                # Only color valid pixels
                # Co-visible = matchable (Yellow)
                overlay[matchable_mask] = [1, 1, 0, 0.6]
                # Non-matchable = facades/DSM-inconsistent (Red)
                overlay[valid_mask & ~matchable_mask] = [1, 0, 0, 0.6]
                
                num_valid = np.sum(valid_mask)
                num_mut = np.sum(matchable_mask)
                num_rej = np.sum(valid_mask & ~matchable_mask)
                print(f"  UAV Mask: {num_mut} co-visible, {num_rej} non-matchable (of {num_valid} valid)")
                ax_uav_mask.imshow(overlay)
            else:
                ax_uav_mask.text(0.5, 0.5, "Point Map Not Available", ha='center', va='center', transform=ax_uav_mask.transAxes)

    # --- Compute valid-data bounding box from geo_mask for DOP/DSM ---
    # This clips out the white padding (pixels with no geodata) at the edges.
    _valid_xlim_geo = None
    _valid_ylim_geo = None
    if geo_mask_np is not None:
        rows_valid = np.any(geo_mask_np, axis=1)
        cols_valid = np.any(geo_mask_np, axis=0)
        if rows_valid.any() and cols_valid.any():
            r_min, r_max = np.where(rows_valid)[0][[0, -1]]
            c_min, c_max = np.where(cols_valid)[0][[0, -1]]
            _valid_xlim_geo = (c_min, c_max)
            _valid_ylim_geo = (r_max, r_min)  # inverted Y for image coords

    # Col 4: DOP + LoD (no camera annotations here — those go on DOP Mask)
    _native_gsd = sample.get('native_gsd', None)
    _dop_year = sample.get('dop_year', None)
    title4 = "DOP"
    if _dop_year is not None:
        title4 += f" {_dop_year}"
    if vis_lod:
        title4 += " (+ LoD)"
    if _native_gsd is not None:
        title4 += f" — {_native_gsd:.2f} m/px"
    ax4 = prepare_ax(1, geo_active, 'DOP', title4)
    if ax4:
        ax4.imshow(img_dop_vis, aspect='equal')
        h_dop_vis, w_dop_vis = img_dop_vis.shape[:2]
        if _valid_xlim_geo is not None:
            ax4.set_xlim(*_valid_xlim_geo)
            ax4.set_ylim(*_valid_ylim_geo)
        else:
            ax4.set_xlim(0, w_dop_vis - 1)
            ax4.set_ylim(h_dop_vis - 1, 0)
        setup_spatial_axes(ax4, sample.get('crop_bounds'), img_dop_vis.shape)

        # Draw single-frame crop bounding box (what the dataset would extract for this frame alone)
        _sf_crop = sample.get('single_frame_crop')
        _crop_bounds_dop = sample.get('crop_bounds')
        if _sf_crop is not None and _crop_bounds_dop is not None:
            sf_min_x, sf_min_y, sf_max_x, sf_max_y = _sf_crop
            _ref = img_dop_vis.shape
            tl = world_to_dop_px(sf_min_x, sf_max_y, _crop_bounds_dop, _ref)  # top-left (max_y because Y flipped)
            br = world_to_dop_px(sf_max_x, sf_min_y, _crop_bounds_dop, _ref)  # bottom-right
            rect_w = br[0] - tl[0]
            rect_h = br[1] - tl[1]
            from matplotlib.patches import Rectangle as MplRect
            rect = MplRect((tl[0], tl[1]), rect_w, rect_h,
                           linewidth=2.0, edgecolor='blue', facecolor='none', zorder=10)
            ax4.add_patch(rect)
            ax4.legend(
                handles=[MplRect((0, 0), 1, 1, linewidth=2.0, edgecolor='blue',
                                 facecolor='none', label='Single-frame crop')],
                loc='upper right', fontsize='xx-small', framealpha=0.8)

    # Col 5: DSM
    ax5 = prepare_ax(1, geo_active, 'DSM', "DSM")
    if ax5:
        if 'dsm' in cmap_ranges:
            v0, v1 = cmap_ranges['dsm']
        else:
            _dsm_valid = dsm_data[np.isfinite(dsm_data) & (dsm_data != 0)]
            if len(_dsm_valid) > 0:
                v0, v1 = float(np.percentile(_dsm_valid, 1)), float(np.percentile(_dsm_valid, 99))
            else:
                v0, v1 = np.nanmin(dsm_data), np.nanmax(dsm_data)
        im5 = ax5.imshow(dsm_data, cmap='terrain', vmin=v0, vmax=v1, aspect='equal')
        h_dsm, w_dsm = dsm_data.shape[:2]
        if _valid_xlim_geo is not None:
            ax5.set_xlim(*_valid_xlim_geo)
            ax5.set_ylim(*_valid_ylim_geo)
        else:
            ax5.set_xlim(0, w_dsm - 1)
            ax5.set_ylim(h_dsm - 1, 0)
        add_cb(ax5, im5, v0, v1, sc_z, off_z, "Elev (m)", norm_type)
        setup_spatial_axes(ax5, sample.get('crop_bounds'), dsm_data.shape)

        # Draw single-frame crop bounding box on DSM too (matches DOP rect)
        _sf_crop_dsm = sample.get('single_frame_crop')
        _crop_bounds_dsm = sample.get('crop_bounds')
        if _sf_crop_dsm is not None and _crop_bounds_dsm is not None:
            sf_min_x, sf_min_y, sf_max_x, sf_max_y = _sf_crop_dsm
            _ref_dsm = dsm_data.shape
            tl_d = world_to_dop_px(sf_min_x, sf_max_y, _crop_bounds_dsm, _ref_dsm)
            br_d = world_to_dop_px(sf_max_x, sf_min_y, _crop_bounds_dsm, _ref_dsm)
            from matplotlib.patches import Rectangle as MplRect
            rect_dsm = MplRect((tl_d[0], tl_d[1]), br_d[0] - tl_d[0], br_d[1] - tl_d[1],
                               linewidth=2.0, edgecolor='blue', facecolor='none', zorder=10)
            ax5.add_patch(rect_dsm)

    # DOP Mask - UAV Visibility + Camera FOV + Trajectory
    ax_dop_mask = prepare_ax(1, geo_active, 'DOPMask', "DOP + Trajectory + FoV")
    if ax_dop_mask:
        h_dop_base, w_dop_base = img_dop.shape[:2]
        
        # Project co-visible point_map pixels (horizontal + DSM-consistent) onto DOP image
        # Yellow = DOP pixels that receive co-visible projection (matchable)
        # Red = DOP pixels in FOV but NOT receiving projection (occluded/facades)
        w2c = sample.get('extrinsics_raw')
        K = sample.get('intrinsics')
        query = sample.get('query')
        crop_bounds = sample.get('crop_bounds')
        
        dop_mask_legend_handles = []
        has_overlay = False
        if pm_raw is not None and crop_bounds is not None and covisible_mask is not None:
            Hd, Wd = img_dop.shape[:2]
            
            # Get point_map as numpy
            pm_np = pm_raw.cpu().numpy() if torch.is_tensor(pm_raw) else pm_raw
            uav_h, uav_w = pm_np.shape[1:]
            
            # Use covisible_mask instead of just valid_pm
            # This filters facades via DSM consistency check
            valid_idx = np.where(covisible_mask)
            world_x = pm_np[0, valid_idx[0], valid_idx[1]]  # UTM X
            world_y = pm_np[1, valid_idx[0], valid_idx[1]]  # UTM Y
            
            # Project world coords to DOP pixel coords
            x_min, y_min, x_max, y_max = crop_bounds
            dop_u = ((world_x - x_min) / (x_max - x_min) * Wd).astype(int)
            dop_v = ((y_max - world_y) / (y_max - y_min) * Hd).astype(int)  # Y is flipped
            
            # Filter to valid DOP bounds
            in_dop = (dop_u >= 0) & (dop_u < Wd) & (dop_v >= 0) & (dop_v < Hd)
            dop_u_valid = dop_u[in_dop]
            dop_v_valid = dop_v[in_dop]
            
            # Mark projected pixels as visible (use scatter to accumulate)
            visible_2d = np.zeros((Hd, Wd), dtype=bool)
            visible_2d[dop_v_valid, dop_u_valid] = True
            
            # Fill sampling gaps in the projected visible region.
            # Individual projected pixels are sparse (UAV→DOP resolution mismatch),
            # creating tiny holes that would be falsely marked as "occluded".
            # Morphological closing (dilate→erode) fills these gaps while
            # preserving the true boundary shape, giving clean yellow/red borders.
            visible_2d = binary_closing(visible_2d, iterations=5)
            visible_2d = binary_dilation(visible_2d, iterations=1)  # slight expansion
            
            # FOV mask: rasterise the frustum footprint polygon directly.
            # This is O(polygon_area) vs O(DSM_pixels * matmul) for the old approach.
            in_fov_2d = np.zeros((Hd, Wd), dtype=bool)
            if frustum_corners is not None and crop_bounds is not None:
                corners_px = [world_to_dop_px(cx, cy, crop_bounds, (Hd, Wd))
                              for cx, cy in frustum_corners]
                # Clamp to avoid int32 overflow; cv2.fillPoly clips to image anyway
                _CLAMP = max(Hd, Wd) * 10
                corners_px = [(np.clip(px, -_CLAMP, _CLAMP),
                                np.clip(py, -_CLAMP, _CLAMP))
                               for px, py in corners_px]
                poly_pts = np.array(corners_px, dtype=np.int32).reshape(-1, 1, 2)
                fov_mask_img = np.zeros((Hd, Wd), dtype=np.uint8)
                cv2.fillPoly(fov_mask_img, [poly_pts], 1)
                in_fov_2d = fov_mask_img.astype(bool)
            
            # Occluded = in FOV but not covered by closed visible region.
            # This gives clean contiguous red regions (behind buildings, facades)
            # with clear boundaries against yellow visible areas.
            occluded_2d = in_fov_2d & ~visible_2d
            
            # Composite overlay directly onto the DOP image (float copy).
            # This avoids matplotlib's RGBA alpha-blending which makes red
            # invisible on dark satellite backgrounds.
            dop_float = img_dop.astype(np.float32) / 255.0 if img_dop.max() > 1 else img_dop.astype(np.float32)
            composited = dop_float.copy()
            # Yellow (visible) — moderate blend
            alpha_y = 0.45
            composited[visible_2d] = (1 - alpha_y) * composited[visible_2d] + alpha_y * np.array([1.0, 1.0, 0.0])
            # Red (occluded) — strong blend so it's always clearly visible
            alpha_r = 0.7
            composited[occluded_2d] = (1 - alpha_r) * composited[occluded_2d] + alpha_r * np.array([1.0, 0.0, 0.0])
            composited = np.clip(composited, 0, 1)

            num_vis = np.sum(visible_2d)
            num_occ = np.sum(occluded_2d)
            print(f"  DOP Mask: {num_vis} visible (projected), {num_occ} occluded")

            # Show the composited DOP+overlay image
            ax_dop_mask.imshow(composited, aspect='equal')
            if _valid_xlim_geo is not None:
                ax_dop_mask.set_xlim(*_valid_xlim_geo)
                ax_dop_mask.set_ylim(*_valid_ylim_geo)
            else:
                ax_dop_mask.set_xlim(0, w_dop_base - 1)
                ax_dop_mask.set_ylim(h_dop_base - 1, 0)
            has_overlay = True
        else:
            ax_dop_mask.imshow(img_dop, aspect='equal')
            if _valid_xlim_geo is not None:
                ax_dop_mask.set_xlim(*_valid_xlim_geo)
                ax_dop_mask.set_ylim(*_valid_ylim_geo)
            else:
                ax_dop_mask.set_xlim(0, w_dop_base - 1)
                ax_dop_mask.set_ylim(h_dop_base - 1, 0)
            ax_dop_mask.text(0.5, 0.5, "Mask Not Available", ha='center', va='center', transform=ax_dop_mask.transAxes)

        # Legend entries for the yellow/red regions
        if has_overlay:
            dop_mask_legend_handles.append(
                Patch(facecolor=(1, 1, 0, 0.6), edgecolor='none', label='Visible'))
            dop_mask_legend_handles.append(
                Patch(facecolor=(1, 0, 0, 0.9), edgecolor='none', label='Occluded'))

        # --- Draw camera FOV, trajectory and camera triangle on DOP Mask ---
        crop_bounds = sample.get('crop_bounds')
        if crop_bounds is not None:
            _ref_shape = img_dop.shape
            # Draw trajectory history
            if trajectory_history and len(trajectory_history) > 0:
                traj_px = [world_to_dop_px(cx, cy, crop_bounds, _ref_shape)
                           for cx, cy in trajectory_history]
                txs = [p[0] for p in traj_px]
                tys = [p[1] for p in traj_px]
                if len(txs) > 1:
                    ax_dop_mask.plot(txs, tys, '-', color='lime', lw=1.5, zorder=7)
                ax_dop_mask.scatter(txs[-1], tys[-1], c='lime', s=30, edgecolors='white',
                                    linewidth=0.8, zorder=8)
                dop_mask_legend_handles.append(
                    Line2D([0], [0], color='lime', lw=1.5, marker='o', markersize=4,
                           markerfacecolor='lime', markeredgecolor='white', label='Trajectory'))

            # Draw frustum polygon (FoV footprint on ground)
            # Only draw the edge (no fill) so it doesn't cover the yellow/red overlay
            if frustum_corners is not None:
                corners_px = [world_to_dop_px(cx, cy, crop_bounds, _ref_shape)
                              for cx, cy in frustum_corners]
                poly = MplPolygon(corners_px, closed=True, fill=False,
                                  edgecolor='cyan', linewidth=2.5, zorder=10)
                ax_dop_mask.add_patch(poly)
                dop_mask_legend_handles.append(
                    MplPolygon([(0, 0)], closed=True, facecolor='none',
                               edgecolor='cyan', linewidth=2.0, label='FoV'))

            # Draw camera frustum triangle (optical center → image plane edges)
            if camera_triangle is not None:
                tri_px = [world_to_dop_px(cx, cy, crop_bounds, _ref_shape)
                          for cx, cy in camera_triangle]
                tri = MplPolygon(tri_px, closed=True, fill=True,
                                 facecolor='orange', alpha=0.8,
                                 edgecolor='darkorange', linewidth=2.0, zorder=9)
                ax_dop_mask.add_patch(tri)
                dop_mask_legend_handles.append(
                    MplPolygon([(0, 0)], closed=True, facecolor='orange', alpha=0.8,
                               edgecolor='darkorange', linewidth=2.0, label='Camera'))

        if dop_mask_legend_handles:
            ax_dop_mask.legend(handles=dop_mask_legend_handles, loc='upper right',
                               fontsize='xx-small', framealpha=0.8)

    # 8. Draw match markers
    _ms = 8 if paper else 15
    _fs = 5 if paper else 7
    handles = [Line2D([0], [0], color='green', lw=2, label='LoD')]

    if len(final_matches) > 0:
        colors = ['cyan', 'magenta', 'yellow', 'lime', 'orange', 'white']
        for i, m in enumerate(final_matches):
            c = colors[i % len(colors)]
            label = str(i + 1)
            # Only draw on active axes (Don't show on LiDAR plot per user request)
            target_axes = []
            target_pts = []
            if ax0: target_axes.append(ax0); target_pts.append(m['uav'])
            if ax4: target_axes.append(ax4); target_pts.append(m['dop'])
            
            for ax, pt in zip(target_axes, target_pts):
                ax.scatter(pt[0], pt[1], c=c, s=_ms, edgecolors='black', linewidth=0.5, zorder=5)
                ax.text(pt[0] + 3, pt[1] - 3, label, fontsize=_fs, color='black', weight='bold',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5),
                        zorder=6)
            handles.append(Line2D([0], [0], marker='o', color='w', label=f'P{label}',
                                  markerfacecolor=c, markersize=4, markeredgecolor='black'))

    if ax0:
        _leg_fs = 3 if paper else 6
        ax0.legend(handles=handles, loc='upper right', fontsize=_leg_fs, framealpha=0.8)

    # 9. Title with sample info, normalization, and augmentation flags
    aug_hflip = sample.get('aug_hflip', False)
    aug_vflip = sample.get('aug_vflip', False)
    aug_color = sample.get('aug_color', False)
    aug_rotation = sample.get('aug_rotation', 0)
    aug_noise = sample.get('aug_noise', False)
    aug_erase = sample.get('aug_erase', False)
    aug_geodata_dropout = sample.get('aug_geodata_dropout', False)
    aug_partial_overlap = sample.get('aug_partial_overlap', False)
    
    aug_parts = []
    if aug_rotation: aug_parts.append(f'Rot{aug_rotation}°')
    if aug_hflip: aug_parts.append('HFlip')
    if aug_vflip: aug_parts.append('VFlip')
    if aug_color: aug_parts.append('Color')
    if aug_noise: aug_parts.append('Noise')
    if aug_erase: aug_parts.append('Erase')
    if aug_geodata_dropout: aug_parts.append('GeoDrop')
    if aug_partial_overlap: aug_parts.append('PartialOvlp')
    aug_str = '+'.join(aug_parts) if aug_parts else 'None'

    xyz_norm = sample.get('norm_type', 'N/A')
    lod_lvl = sample.get('lod_level', '2')
    gsd = sample.get('gsd', 0.0)

    frame_prefix = f"Frame {frame_info[0]+1}/{frame_info[1]} | " if frame_info is not None else ""
    if paper:
        # No suptitle in paper mode — cleaner output
        pass
    else:
        fig.suptitle(
            f"{frame_prefix}Sample: {sample_id} | GSD: {gsd:.3f}m | Norm: {xyz_norm} | LoD: {lod_lvl} | Aug: {aug_str}",
            fontsize=10, y=TITLE_Y
        )

    if fig_ext is not None:
        ext = fig_ext
    else:
        ext = 'pdf' if paper else 'png'
    outfile = os.path.join(output_dir, f"{sample_id}.{ext}")
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)
    print(f"Saved {outfile}")


def _generate_dop_slideshow_mp4(dop_files, output_dir, seq_name, 
                                 hold_frames=45, fade_frames=15, max_size=1024):
    """Generate an MP4 slideshow cycling through DOP years with crossfade transitions.
    
    Args:
        dop_files: List of (label, path, meta_dict) tuples from visualize_multiyear_dops
        output_dir: Output directory for the MP4
        seq_name: Sequence name for the output filename
        hold_frames: Frames to hold each DOP (at 30fps: 1.5s per year)
        fade_frames: Frames for crossfade transition between years
        max_size: Max dimension for output video (resize large DOPs)"""
    import subprocess, tempfile, shutil

    if len(dop_files) < 2:
        return

    mp4_path = os.path.join(output_dir, f"{seq_name}_dop_years.mp4")
    
    # Load and prepare all images
    images = []
    labels = []
    for label, path, _ in dop_files:
        img = cv2.imread(path)
        if img is None:
            continue
        # Crop to non-black content
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        valid_mask = gray > 0
        rows_ok = np.any(valid_mask, axis=1)
        cols_ok = np.any(valid_mask, axis=0)
        if rows_ok.any() and cols_ok.any():
            r0, r1 = np.where(rows_ok)[0][[0, -1]]
            c0, c1 = np.where(cols_ok)[0][[0, -1]]
            img = img[r0:r1+1, c0:c1+1]
        images.append(img)
        labels.append(label.split('\n')[0])  # Just "DOP 2010" etc.
    
    if len(images) < 2:
        return

    # Find common size (use first image as reference, resize all to match)
    h0, w0 = images[0].shape[:2]
    # Scale to max_size
    scale = min(max_size / max(h0, w0), 1.0)
    target_h = int(h0 * scale)
    target_w = int(w0 * scale)
    # Ensure even dimensions for H.264
    target_h = target_h if target_h % 2 == 0 else target_h + 1
    target_w = target_w if target_w % 2 == 0 else target_w + 1

    for i in range(len(images)):
        images[i] = cv2.resize(images[i], (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # Add text overlay with year label
    def add_label(frame, text):
        frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = target_h / 500
        thickness = max(2, int(font_scale * 2))
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        # Background rectangle
        pad = 10
        cv2.rectangle(frame, (pad, pad), (pad + tw + 2*pad, pad + th + 2*pad + baseline),
                      (0, 0, 0), -1)
        cv2.putText(frame, text, (2*pad, pad + th + pad), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
        return frame

    # Write frames to temp directory
    tmp_dir = tempfile.mkdtemp(prefix='dop_slideshow_')
    frame_idx = 0
    
    for i in range(len(images)):
        labeled = add_label(images[i], labels[i])
        # Hold frames
        for _ in range(hold_frames):
            cv2.imwrite(os.path.join(tmp_dir, f'frame_{frame_idx:05d}.png'), labeled)
            frame_idx += 1
        
        # Crossfade to next (or loop back to first)
        next_i = (i + 1) % len(images)
        next_labeled = add_label(images[next_i], labels[next_i])
        for f in range(fade_frames):
            alpha = f / fade_frames
            blended = cv2.addWeighted(labeled, 1 - alpha, next_labeled, alpha, 0)
            cv2.imwrite(os.path.join(tmp_dir, f'frame_{frame_idx:05d}.png'), blended)
            frame_idx += 1

    # Encode with ffmpeg
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', '30',
        '-i', os.path.join(tmp_dir, 'frame_%05d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-bf', '0', '-crf', '18',
        mp4_path,
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    
    if result.returncode == 0:
        size_mb = os.path.getsize(mp4_path) / 1024 / 1024
        print(f"  Saved DOP slideshow video: {mp4_path} ({size_mb:.1f} MB, {frame_idx} frames)")
    else:
        print(f"  ffmpeg failed for DOP slideshow: {result.stderr[:200]}")


def visualize_multiyear_dops(seq_dir, output_dir, dpi=150, paper=False, fig_ext=None):
    """Generate a comparison figure showing all available DOP years for a sequence.
    
    Reads dop/<year>.jpg files from the sequence directory and displays them
    in a grid alongside metadata from meta.json.  Supports both the new
    meta['dops'] dict format and the legacy meta['dop_years'] list.
    
    Args:
        seq_dir: Path to the sequence directory
        output_dir: Output directory for the figure
        dpi: Output DPI"""
    import json
    
    seq_dir = Path(seq_dir)
    meta_path = seq_dir / 'meta.json'
    
    if not meta_path.exists():
        print(f"  No meta.json found in {seq_dir}, skipping multi-year DOP visualization")
        return
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    # Collect DOP files from new 'dops' dict or legacy 'dop_years' list
    dop_files = []
    
    dops_dict = meta.get('dops', {})
    if dops_dict:
        # New format: {'dop_2021': {'file': 'dop/2021.jpg', 'year': 2021, ...}, ...}
        for key in sorted(dops_dict.keys()):
            dy = dops_dict[key]
            dop_path = seq_dir / dy['file']
            # Legacy fallback: try dop_<year>.jpg flat path
            if not dop_path.exists():
                dop_path = seq_dir / f"dop_{dy['year']}.jpg"
            if dop_path.exists():
                label = f"DOP {dy['year']}"
                if dy.get('capture_date') and dy['capture_date'] != str(dy['year']):
                    label += f"\n({dy['capture_date']})"
                label += f"\n{dy.get('source', '?')}, {dy.get('coverage', 0)*100:.0f}% cov"
                dop_files.append((label, str(dop_path), dy))
    else:
        # Legacy format: list of dicts in 'dop_years'
        dop_years = meta.get('dop_years', [])
        for dy in dop_years:
            dop_path = seq_dir / dy['file']
            if not dop_path.exists():
                dop_path = seq_dir / f"dop_{dy['year']}.jpg"
            if dop_path.exists():
                label = f"DOP {dy['year']}"
                if dy.get('capture_date') and dy['capture_date'] != str(dy['year']):
                    label += f"\n({dy['capture_date']})"
                label += f"\n{dy.get('source', '?')}, {dy.get('coverage', 0)*100:.0f}% cov"
                dop_files.append((label, str(dop_path), dy))
        # Also include legacy dop.jpg if present and no other years found
        primary_dop = seq_dir / 'dop.jpg'
        if primary_dop.exists() and not any(str(primary_dop) == p for _, p, _ in dop_files):
            dop_files.insert(0, ('DOP (primary)', str(primary_dop), None))

    # Fallback: scan dop/<year>.jpg files directly when meta.json has no dops/dop_years entries
    if not dop_files:
        dop_subdir = seq_dir / 'dop'
        if dop_subdir.is_dir():
            for jpg_path in sorted(dop_subdir.glob('*.jpg')):
                try:
                    year = int(jpg_path.stem)
                except ValueError:
                    continue
                dop_files.append((f"DOP {year}", str(jpg_path), {'year': year}))
        # Also check legacy flat dop_<year>.jpg files in sequence root
        if not dop_files:
            for jpg_path in sorted(seq_dir.glob('dop_*.jpg')):
                try:
                    year = int(jpg_path.stem.split('_', 1)[1])
                except (ValueError, IndexError):
                    continue
                dop_files.append((f"DOP {year}", str(jpg_path), {'year': year}))
    
    if len(dop_files) <= 1:
        print(f"  Only 1 or no DOPs available, skipping multi-year comparison")
        return
    
    n = len(dop_files)
    # Arrange in a grid: up to 4 per row
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows + 0.5))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]
    
    seq_name = meta.get('name', seq_dir.name)
    fig.suptitle(f"Multi-Year DOP Comparison \u2014 {seq_name}", fontsize=14, y=0.98)
    
    for i, (label, path, dy_meta) in enumerate(dop_files):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Crop to non-black content to hide padding
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            valid_mask = gray > 0
            rows_ok = np.any(valid_mask, axis=1)
            cols_ok = np.any(valid_mask, axis=0)
            if rows_ok.any() and cols_ok.any():
                r0, r1 = np.where(rows_ok)[0][[0, -1]]
                c0, c1 = np.where(cols_ok)[0][[0, -1]]
                img = img[r0:r1+1, c0:c1+1]
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'Failed to load', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title(label, fontsize=9)
        ax.axis('off')
    
    # Hide unused axes
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].axis('off')
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    if fig_ext is not None:
        _ext = fig_ext
    else:
        _ext = 'pdf' if paper else 'png'
    outfile = os.path.join(output_dir, f"{seq_dir.name}_dop_years.{_ext}")
    fig.savefig(outfile, dpi=dpi, bbox_inches='tight' if _ext == 'pdf' else None)
    plt.close(fig)
    print(f"  Saved multi-year DOP comparison: {outfile}")

    if not paper:
        # --- Generate MP4 slideshow cycling through DOP years ---
        _generate_dop_slideshow_mp4(dop_files, output_dir, seq_dir.name)




def _render_frame_worker_numpy(args_tuple):
    """Worker for ThreadPoolExecutor frame rendering.

    Receives a dict of numpy arrays (already converted by _torch_dict_to_numpy
    so no pickling of torch tensors is needed).  Converts back to torch tensors
    in-thread and calls visualize_sample, which uses the Agg matplotlib backend
    (safe for multi-threaded use when each thread manages its own figure)."""
    (frame_data_np, output_dir, frame_idx, frame_info, frustum_corners,
     trajectory_history, camera_triangle, cmap_ranges, vis_kwargs) = args_tuple

    # Re-wrap float numpy arrays as torch tensors for visualize_sample
    frame_data = _numpy_dict_to_torch(frame_data_np)

    # Force PNG for video frame workers (ffmpeg needs PNGs)
    vk = dict(vis_kwargs)
    vk['fig_ext'] = 'png'
    visualize_sample(
        frame_data, output_dir, frame_idx,
        frame_info=frame_info,
        frustum_corners=frustum_corners,
        trajectory_history=trajectory_history,
        camera_triangle=camera_triangle,
        cmap_ranges=cmap_ranges,
        **vk,
    )
    frame_sid = frame_data.get('sample_id', str(frame_idx))
    src = os.path.join(output_dir, f"{frame_sid}.png")
    dst = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
    if os.path.exists(src) and src != dst:
        os.rename(src, dst)






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', help='Path to dataset')
    parser.add_argument('--output_dir', default='outputs/vis_movingdrone', help='Output directory')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--stride', type=int, default=1, help='Frame stride (default 1)')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to visualize (Image Mode)')
    parser.add_argument('--normalization', type=str, default='01', choices=['01', 'imagenet'], help='Image normalization')
    parser.add_argument('--xyz_normalization', default='mean_std', help='XYZ Normalization (mean_std, minmax_01, minmax_11, none)')
    parser.add_argument('--augment', action='store_true', default=False, help='Enable augmentation (off by default for visualization)')
    parser.add_argument('--force_augment', action='store_true', help='Force all augmentations')
    parser.add_argument('--num_matches', type=int, default=5, help='Number of matches to visualize')
    parser.add_argument('--sample_id', type=str, default=None, help='Specific sample ID to visualize (e.g. brandenburger_tor_0010)')
    parser.add_argument('--seq_name', type=str, default=None, help='Specific sequence name to visualize (e.g. brandenburger_tor)')
    parser.add_argument('--lod_level', type=str, default='2', choices=['1', '2'], help='LoD level to visualize')
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame index for sequences (default 0)')
    parser.add_argument('--end_frame', type=int, default=None, help='End frame index for sequences (default None = end of sequence)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Parallel workers for rendering (default: cpu_count/2)')
    
    parser.add_argument('--no-vis-lod', action='store_false', dest='vis_lod', help='Disable LoD visualization')
    parser.add_argument('--no-vis-semantics', action='store_false', dest='vis_semantics', help='Disable semantic coloring')
    parser.add_argument('--no-vis-matches', action='store_false', dest='vis_matches', help='Disable match visualization')
    parser.add_argument('--no-vis-lidar', action='store_false', dest='vis_lidar', help='Disable LiDAR visualization')
    parser.add_argument('--no-vis-point-map', action='store_false', dest='vis_point_map', help='Disable point map visualization')
    parser.add_argument('--no-vis-depth-map', action='store_false', dest='vis_depth_map', help='Disable depth map visualization')
    parser.add_argument('--no-vis-normals', action='store_false', dest='vis_normals', help='Disable surface normals visualization')
    parser.add_argument('--no-vis-dop', action='store_false', dest='vis_dop', help='Disable DOP visualization')
    parser.add_argument('--no-vis-dsm', action='store_false', dest='vis_dsm', help='Disable DSM visualization')
    parser.add_argument('--no-vis-matching-masks', action='store_false', dest='vis_matching_masks', help='Disable visualization of correspondence masks')
    parser.set_defaults(
        vis_lod=True, vis_semantics=True, vis_matches=True, vis_dop=True,
        vis_lidar=True, vis_point_map=True, vis_depth_map=True, vis_normals=True,
        vis_dsm=True, vis_matching_masks=True
    )
    parser.add_argument('--crop_multiplier_range', type=float, nargs=2, default=[1.2, 1.5], help='Min/Max random multiplier for crop size')
    parser.add_argument('--no-crop-extension', action='store_const', const=[1.0, 1.0], dest='crop_multiplier_range', help='Disable random crop extension')
    parser.add_argument('--ensure_coverage', action='store_true', default=True, help='Ensure crop covers UAV footprint (default)')
    parser.add_argument('--no-ensure-coverage', action='store_false', dest='ensure_coverage', help='Disable footprint coverage check')
    parser.add_argument('--min_gsd', type=float, default=0.2, help='Minimum GSD (m/px) for DOP/DSM crops')
    parser.add_argument('--max_gsd', type=float, default=10.0, help='Maximum GSD (m/px) for DOP/DSM crops (default 10.0, no cap)')
    parser.add_argument('--sequence_length', type=int, default=None, help='Sequence length for Video Mode (default: Full Sequence)')
    parser.add_argument('--fast', action='store_true', help='Fast mode: disable LoD, LiDAR, matches, semantics')
    parser.add_argument('--paper', action='store_true', help='Paper mode: compact figure, no spatial axes, blue-pink normals')
    parser.add_argument('--fig_ext', type=str, default=None, help='Output figure extension (png, pdf, svg). Default: pdf if --paper else png')
    parser.add_argument('--dpi', type=int, default=150, help='Output DPI (lower=faster, default 150)')

    args = parser.parse_args()
    
    # --- Resolve Modes (Image vs Video) ---
    if args.num_samples is not None and args.sequence_length is not None:
        raise ValueError("Cannot specify both --num_samples and --sequence_length. Choose one.")

    # Determine operating mode:
    #   - Single image PNG only when --num_samples 1
    #   - Video output in all other cases (default, or --num_samples N>1 for subsampling)
    image_mode = (args.num_samples is not None and args.num_samples == 1)
    video_mode = not image_mode

    if image_mode:
        # Single-image mode: output one PNG
        args.sequence_length = 1
        print(f"Mode: SINGLE IMAGE")
    else:
        # Video Generation Mode: stream single frames one at a time for memory efficiency
        # We use sequence_length=1 dataset and group frames by sequence manually.
        args.sequence_length = 1  # Always load single frames — stream through full sequence
        subsample_n = args.num_samples  # None = all frames, or N = subsample N frames
        args.num_samples = float('inf')  # Collect all matching indices first
        if subsample_n is not None:
            print(f"Mode: VIDEO GENERATION (subsampling {subsample_n} frames, start={args.start_frame}, end={args.end_frame})")
        else:
            print(f"Mode: VIDEO GENERATION (all frames, start={args.start_frame}, end={args.end_frame})")

    # --seq_name is an alias for --sample_id
    if args.seq_name and not args.sample_id:
        args.sample_id = args.seq_name
    
    # Determine number of workers
    if args.num_workers is None:
        args.num_workers = max(1, (os.cpu_count() or 4) // 2)
    
    # Fast mode overrides: disable expensive visualizations
    if args.fast:
        args.vis_lod = False
        args.vis_lidar = False
        args.vis_matches = False
        args.vis_semantics = False
        
    print('aguments:', args.force_augment, args.augment)

    os.makedirs(args.output_dir, exist_ok=True)

    vis_matching_masks = args.vis_matching_masks
    
    # Matching needs DSM and Depth. LoD needs DSM for occlusion.
    load_dsm = args.vis_dsm or args.vis_matches or args.vis_lod or vis_matching_masks
    load_depth = args.vis_depth_map or args.vis_matches or vis_matching_masks or args.vis_point_map
    load_normals = args.vis_normals

    print(f"Loading MovingDrone dataset from {args.dataset_dir}...")
    print(f"Normalization: {args.normalization} | XYZ: {args.xyz_normalization} | Augment: {args.augment} (Force: {args.force_augment}) | SeqLen: {args.sequence_length}")
    try:
        ds = MovingDrone(
            args.dataset_dir, split=args.split,
            normalization=args.normalization,
            xyz_normalization=args.xyz_normalization,
            augment=args.augment or args.force_augment,
            force_augment=args.force_augment,
            stride=args.stride,
            lod_level=args.lod_level,
            load_lidar=args.vis_lidar,
            load_lod=args.vis_lod,
            load_depth=load_depth,
            load_normals=load_normals,
            load_dop=args.vis_dop,
            load_dsm=load_dsm,
            crop_multiplier_range=tuple(args.crop_multiplier_range),
            ensure_coverage=args.ensure_coverage,
            return_matching=vis_matching_masks,
            min_gsd=args.min_gsd,
            sequence_length=args.sequence_length,
            max_gsd=args.max_gsd,
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    if len(ds) == 0:
        print("Dataset is empty. Check paths.")
        return

    if args.sample_id or args.seq_name:
        query_id = args.seq_name if args.seq_name else args.sample_id
        print(f"Searching for samples matching {query_id}...")
        indices = []
        for i in range(len(ds)):
            s = ds.samples[i]
            fid = s.get('frame_id', s.get('start_frame', 0))
            sid_padded = f"{s['seq_path'].name}_{fid:04d}"
            sid_unpadded = f"{s['seq_path'].name}_{fid}"

            # Match by full ID, sequence name precisely, or substring match on seq_path
            if query_id in (sid_padded, sid_unpadded, s['seq_path'].name):
                indices.append(i)
                if image_mode and len(indices) >= 1:
                    break

        if not indices:
            # Try fuzzy matching if exact match fails
            for i in range(len(ds)):
                s = ds.samples[i]
                if query_id in s['seq_path'].name:
                    indices.append(i)
                    if image_mode and len(indices) >= 1:
                        break

        if not indices:
            print(f"Could not find any samples matching {query_id} in {args.split} split.")
            return
    else:
        n = len(ds)
        if image_mode:
            # Single image: pick the middle frame
            indices = [n // 2]
        else:
            # Video mode: all samples (will be grouped by sequence in the video loop)
            indices = list(range(n))

    print(f"Visualizing {len(indices)} samples...")

    vis_kwargs = dict(
        normalization=args.normalization,
        num_matches=args.num_matches,
        augment=args.augment,
        vis_lod=args.vis_lod,
        vis_semantics=args.vis_semantics,
        vis_matches=args.vis_matches,
        vis_lidar=args.vis_lidar,
        vis_point_map=args.vis_point_map,
        vis_depth_map=args.vis_depth_map,
        vis_normals=args.vis_normals,
        vis_dop=args.vis_dop,
        vis_dsm=args.vis_dsm,
        vis_matching_masks=vis_matching_masks,
        dpi=args.dpi,
        paper=args.paper,
        fig_ext=args.fig_ext,
    )

    if image_mode:
        # ── SINGLE IMAGE MODE ───────────────────────────────────────────────
        idx = indices[0]
        sample = ds[idx]
        w2c_sf = sample['extrinsics_raw'].cpu().numpy()
        K_sf = sample['intrinsics'].cpu().numpy()
        q_shape_sf = sample['query'].shape  # (C, H, W)
        ih_sf, iw_sf = q_shape_sf[1], q_shape_sf[2]

        dsm_sf = sample.get('dsm')
        norm_type_sf = sample.get('norm_type', 'none')
        if dsm_sf is not None and norm_type_sf not in (None, 'none', 'None'):
            dsm_raw_sf = torch.from_numpy(
                denorm(dsm_sf, sample.get('dsm_scale'), sample.get('dsm_offset'), norm_type_sf))
        else:
            dsm_raw_sf = dsm_sf

        geo_mask_sf = sample.get('geodata_mask')
        if dsm_raw_sf is not None:
            geo_np_sf = geo_mask_sf[0].cpu().numpy() > 0.5 if geo_mask_sf is not None else None
            if geo_np_sf is not None:
                valid_z_sf = dsm_raw_sf[2].cpu().numpy()[geo_np_sf]
                z_ground_sf = float(np.median(valid_z_sf)) if len(valid_z_sf) > 0 else 0.0
            else:
                z_ground_sf = float(torch.median(dsm_raw_sf[2]).item())
            corners_sf = compute_frustum_corners(w2c_sf, K_sf, ih_sf, iw_sf, z_ground_sf)
        else:
            corners_sf = None

        # Scale triangle to ~5% of crop width for visibility
        _cb_sf = sample.get('crop_bounds')
        if _cb_sf is not None:
            _tri_size_sf = abs(_cb_sf[2] - _cb_sf[0]) * 0.05
        else:
            _tri_size_sf = 20.0
        tri_sf = compute_camera_triangle(w2c_sf, size=_tri_size_sf)

        visualize_sample(sample, args.output_dir, idx,
                         frustum_corners=corners_sf,
                         camera_triangle=tri_sf,
                         **vis_kwargs)

        # Generate multi-year DOP comparison
        _seq_path = ds.samples[idx]['seq_path']
        visualize_multiyear_dops(_seq_path, args.output_dir, dpi=args.dpi, paper=args.paper, fig_ext=args.fig_ext)

    else:
        # ── VIDEO GENERATION MODE (streaming) ───────────────────────────────
        # Group single-frame indices by sequence name for ordered streaming.
        import subprocess
        import shutil
        from collections import defaultdict

        # Build a map: seq_name -> sorted list of (frame_id, ds_idx)
        seq_frame_map = defaultdict(list)
        for ds_idx in indices:
            s = ds.samples[ds_idx]
            seq_frame_map[str(s['seq_path'].name)].append((s['frame_id'], ds_idx))

        for seq_name, frame_entries in seq_frame_map.items():
            # Sort by frame_id to guarantee temporal order
            frame_entries.sort(key=lambda x: x[0])

            # Apply start/end frame filtering
            start_f = args.start_frame
            end_f = args.end_frame  # None means no upper bound

            if end_f is not None:
                frame_entries = [(fid, di) for fid, di in frame_entries
                                 if start_f <= fid < end_f]
            else:
                frame_entries = [(fid, di) for fid, di in frame_entries
                                 if fid >= start_f]

            if not frame_entries:
                print(f"  No frames found for sequence {seq_name} in range [{start_f}, {end_f})")
                continue

            # Subsample frames if --num_samples N was given (N > 1)
            if subsample_n is not None and len(frame_entries) > subsample_n:
                step = len(frame_entries) / subsample_n
                frame_entries = [frame_entries[int(i * step)] for i in range(subsample_n)]
                print(f"  Subsampled to {len(frame_entries)} frames")

            total_frames = len(frame_entries)
            print(f"\nSequence '{seq_name}': {total_frames} frames "
                  f"(frames {frame_entries[0][0]}-{frame_entries[-1][0]})")

            # Derive output video name from first/last frame
            first_fid = frame_entries[0][0]
            last_fid = frame_entries[-1][0]
            vid_name = f"{seq_name}_{first_fid:04d}-{last_fid:04d}"
            tmp_dir = os.path.join(args.output_dir, f"_tmp_{vid_name}")
            os.makedirs(tmp_dir, exist_ok=True)

            # ── Compute FIXED crop covering the entire sequence ───────────────
            # We compute the union footprint of all frame cameras and load one
            # DOP/DSM crop that stays constant across all frames.
            seq_path_main = ds.samples[frame_entries[0][1]]['seq_path']
            meta_main = ds.samples[frame_entries[0][1]]['meta']

            # Ensure pose/intrinsics caches are populated
            _ = ds[frame_entries[0][1]]  # warms _pose_cache, _intr_cache for this seq

            df_poses = ds._pose_cache.get(seq_path_main)
            K_raw_main = ds._intr_cache.get(seq_path_main)  # un-scaled (H_orig × W_orig)

            geodata_override = None  # will hold the fixed DOP/DSM override

            if df_poses is not None and K_raw_main is not None:
                # Compute union footprint from all frames in view
                fp_min_x, fp_min_y = np.inf, np.inf
                fp_max_x, fp_max_y = -np.inf, -np.inf
                for fid_i, ds_idx_i in frame_entries:
                    if fid_i >= len(df_poses):
                        continue
                    row = df_poses.iloc[fid_i]
                    w2c_i = quat_pos_to_w2c(
                        row['x'], row['y'], row['z'],
                        row['qw'], row['qx'], row['qy'], row['qz'],
                    )
                    fp = get_visible_footprint(
                        w2c_i, K_raw_main, meta_main['height'], meta_main['width'],
                        plane_z=35.0,  # Match dataset's default ground plane
                    )
                    if fp is not None and not any(np.isnan(fp)):
                        fmn_x, fmn_y, fmx_x, fmx_y = fp
                        fp_min_x = min(fp_min_x, fmn_x)
                        fp_min_y = min(fp_min_y, fmn_y)
                        fp_max_x = max(fp_max_x, fmx_x)
                        fp_max_y = max(fp_max_y, fmx_y)

                if np.isfinite(fp_min_x):
                    # Derive fixed crop from union footprint (+20 % padding each side)
                    cx_fp = (fp_min_x + fp_max_x) / 2
                    cy_fp = (fp_min_y + fp_max_y) / 2
                    w_fp = fp_max_x - fp_min_x
                    h_fp = fp_max_y - fp_min_y
                    H_px, W_px = ds.size  # output image size in pixels
                    scale_fp = max(w_fp * 1.2 / W_px, h_fp * 1.2 / H_px)
                    scale_fp = max(scale_fp, args.min_gsd)
                    half_w_fp = scale_fp * W_px / 2
                    half_h_fp = scale_fp * H_px / 2
                    fixed_crop = (
                        cx_fp - half_w_fp, cy_fp - half_h_fp,
                        cx_fp + half_w_fp, cy_fp + half_h_fp,
                    )

                    # Load full preprocessed geodata (uses internal cache)
                    preprocessed = ds._load_preprocessed_geodata(seq_path_main)
                    if preprocessed:
                        dop_img_full, dop_meta = preprocessed['dop']
                        dsm_h_full, dsm_meta = preprocessed['dsm']
                        native_gsd = dop_meta['gsd']

                        fc_min_x, fc_min_y, fc_max_x, fc_max_y = fixed_crop
                        metric_w_fc = fc_max_x - fc_min_x
                        metric_h_fc = fc_max_y - fc_min_y
                        px_w_fc = int(round(metric_w_fc / native_gsd))
                        px_h_fc = int(round(metric_h_fc / native_gsd))
                        center_x_fc = (fc_min_x + fc_max_x) / 2
                        center_y_fc = (fc_min_y + fc_max_y) / 2

                        g_min_x, g_min_y, g_max_x, g_max_y = dop_meta['bounds']
                        full_h_d, full_w_d = dop_img_full.shape[:2]

                        col_c_fc = (center_x_fc - g_min_x) / native_gsd
                        row_c_fc = (g_max_y - center_y_fc) / native_gsd
                        col_start_fc = int(round(col_c_fc - px_w_fc / 2))
                        row_start_fc = int(round(row_c_fc - px_h_fc / 2))
                        col_end_fc = col_start_fc + px_w_fc
                        row_end_fc = row_start_fc + px_h_fc

                        v_col_s = max(0, col_start_fc)
                        v_row_s = max(0, row_start_fc)
                        v_col_e = min(full_w_d, col_end_fc)
                        v_row_e = min(full_h_d, row_end_fc)
                        v_w_fc = v_col_e - v_col_s
                        v_h_fc = v_row_e - v_row_s

                        # DOP crop
                        dop_crop_np = np.zeros((px_h_fc, px_w_fc, 3), dtype=np.uint8)
                        if v_w_fc > 0 and v_h_fc > 0:
                            b_col_fc = v_col_s - col_start_fc
                            b_row_fc = v_row_s - row_start_fc
                            dop_crop_np[b_row_fc:b_row_fc+v_h_fc, b_col_fc:b_col_fc+v_w_fc] = \
                                dop_img_full[v_row_s:v_row_e, v_col_s:v_col_e]
                        image_dop_fc = TF.resize(TF.to_tensor(dop_crop_np), ds.size, interpolation=TF.InterpolationMode.BILINEAR)

                        # DSM crop
                        dsm_h_crop = dsm_h_full  # same grid as DOP
                        dsm_crop_np = np.full((px_h_fc, px_w_fc), -9999.0, dtype=np.float32)
                        if v_w_fc > 0 and v_h_fc > 0:
                            b_col_fc = v_col_s - col_start_fc
                            b_row_fc = v_row_s - row_start_fc
                            dsm_crop_np[b_row_fc:b_row_fc+v_h_fc, b_col_fc:b_col_fc+v_w_fc] = \
                                dsm_h_crop[v_row_s:v_row_e, v_col_s:v_col_e]
                        image_dsm_xyz_fc = torch.from_numpy(
                            dsm_to_xyz(dsm_crop_np, fixed_crop, ds.size)
                        )

                        # Geodata mask for fixed crop
                        geo_mask_fc = (
                            (image_dop_fc != 0).any(dim=0, keepdim=True).float()
                            * (image_dsm_xyz_fc != 0).any(dim=0, keepdim=True).float()
                        )

                        # Store the raw (un-normalized) fixed-crop DSM directly.
                        # We do NOT override dsm_scale/dsm_offset/norm_type because
                        # point_map and depth use per-frame normalization stats.
                        # Instead, mark _dsm_is_raw=True so visualize_sample skips
                        # DSM denormalization.
                        geodata_override = {
                            'dop': image_dop_fc,
                            'dsm': image_dsm_xyz_fc,        # raw XYZ — already in meters
                            '_dsm_raw': image_dsm_xyz_fc,   # same — for cmap/z_ground/matches
                            '_dsm_is_raw': True,             # signal to skip DSM denorm
                            'geodata_mask': geo_mask_fc,
                            'crop_bounds': fixed_crop,
                            'gsd': scale_fp,
                            'native_gsd': native_gsd,
                        }
                        print(f"  Fixed crop: {fixed_crop}, GSD={scale_fp:.3f} m/px")

            # Load the FIRST frame sample (for query image shape and trajectory matching).
            # Then apply the fixed-crop geodata override so it uses stable DOP/DSM.
            first_sample = ds[frame_entries[0][1]]
            if geodata_override is not None:
                first_sample.update(geodata_override)
                # Use the pre-computed raw fixed-crop DSM for z_ground/cmap/matching.
                # It is already in absolute UTM meters so no denorm is needed.
                dsm_raw_seq = geodata_override['_dsm_raw']
                geo_mask_seq = geodata_override['geodata_mask']
            else:
                norm_type_seq = first_sample.get('norm_type', 'none')
                dsm_scale = first_sample.get('dsm_scale')
                dsm_offset = first_sample.get('dsm_offset')
                dsm_raw_seq = first_sample['dsm']
                if norm_type_seq not in (None, 'none', 'None'):
                    dsm_raw_seq = torch.from_numpy(
                        denorm(dsm_raw_seq, dsm_scale, dsm_offset, norm_type_seq))
                geo_mask_seq = first_sample.get('geodata_mask')
            geo_mask_np_seq = (geo_mask_seq[0].cpu().numpy() > 0.5
                               if geo_mask_seq is not None else None)

            if geo_mask_np_seq is not None:
                valid_z = dsm_raw_seq[2].cpu().numpy()[geo_mask_np_seq]
                z_ground = float(np.median(valid_z)) if len(valid_z) > 0 else 0.0
            else:
                z_ground = float(torch.median(dsm_raw_seq[2]).item())

            # Global colormap ranges — derived from the already-loaded DSM.
            # DSM elevation range is used for DSM and point_map (both are world-Z).
            # Depth and LiDAR use camera-space Z (computed per-frame, see below).
            dsm_z = dsm_raw_seq[2].cpu().numpy()
            dsm_valid = dsm_z[geo_mask_np_seq] if geo_mask_np_seq is not None else dsm_z[dsm_z != 0]
            cmap_ranges = {}
            if len(dsm_valid) > 0:
                z_min_global = float(np.percentile(dsm_valid, 1))
                z_max_global = float(np.percentile(dsm_valid, 99))
                cmap_ranges['dsm']       = (z_min_global, z_max_global)
                cmap_ranges['point_map'] = (z_min_global, z_max_global)
                # 'depth' key is not set here: depth is camera-space Z
                # (not world-Z), with ranges pre-scanned separately below.

            # ── Pre-scan depth ranges across sampled frames ──
            # Load a subset of frames to determine a stable global depth/lidar range.
            # Using median of per-frame min/max avoids outlier frames dominating.
            _scan_step = max(1, total_frames // 20)  # sample ~20 frames
            _depth_mins = []
            _depth_maxs = []
            print(f"  Pre-scanning depth ranges ({len(range(0, total_frames, _scan_step))} frames)...")
            for _si in range(0, total_frames, _scan_step):
                _fid, _didx = frame_entries[_si]
                _s = ds[_didx]
                _d = _s.get('depth')
                if _d is None:
                    continue
                _nt = _s.get('norm_type', 'none')
                _dsc = _s.get('dsm_scale')
                _dof = _s.get('dsm_offset')
                if _d is not None and _nt not in (None, 'none', 'None') and _dsc is not None:
                    if isinstance(_dsc, (torch.Tensor, np.ndarray)) and (
                        (torch.is_tensor(_dsc) and _dsc.numel() >= 3) or
                        (isinstance(_dsc, np.ndarray) and _dsc.size >= 3)
                    ):
                        _dz_sc = _dsc[2]; _dz_of = _dof[2]
                    else:
                        _dz_sc = _dsc; _dz_of = _dof
                    _d_raw = denorm(_d, _dz_sc, _dz_of, _nt)
                else:
                    _d_raw = _d.cpu().numpy() if torch.is_tensor(_d) else _d
                if isinstance(_d_raw, torch.Tensor):
                    _d_raw = _d_raw.cpu().numpy()
                _d_flat = _d_raw.flatten()
                _d_valid = _d_flat[np.isfinite(_d_flat) & (_d_flat != 0)]
                if len(_d_valid) > 0:
                    _depth_mins.append(float(np.percentile(_d_valid, 2)))
                    _depth_maxs.append(float(np.percentile(_d_valid, 98)))
            if _depth_mins:
                # Use min/max of per-frame percentiles (not median) so that
                # frames with very different depth ranges (e.g. near-horizontal
                # camera views) are not clipped to all-dark or all-bright.
                _d_vmin = float(min(_depth_mins))
                _d_vmax = float(max(_depth_maxs))
                # Clip depth range to avoid overly wide colormaps
                _MAX_DEPTH_RANGE = 800.0  # metres
                if _d_vmax - _d_vmin > _MAX_DEPTH_RANGE:
                    _d_vmax = _d_vmin + _MAX_DEPTH_RANGE
                cmap_ranges['depth'] = (_d_vmin, _d_vmax)
                print(f"  Depth range (min/max of {len(_depth_mins)} frames): [{_d_vmin:.1f}, {_d_vmax:.1f}] m")

            # UAV image dimensions from first frame
            q_shape = first_sample['query'].shape  # (C, H, W)
            img_h_uav, img_w_uav = q_shape[1], q_shape[2]

            # Init trajectory and persistent matches
            persistent = init_persistent_matches(dsm_raw_seq, geo_mask_seq, args.num_matches)
            trajectory = []

            # ── Streaming render: load frame → submit to thread pool immediately ──
            # Using ThreadPoolExecutor (not mp.Pool) avoids fork+PyTorch deadlocks.
            # The Agg backend is safe for concurrent per-figure rendering.
            # We stream one frame at a time so memory stays bounded.
            n_workers = args.num_workers if args.num_workers is not None else max(1, mp.cpu_count() // 2)
            n_workers = min(n_workers, total_frames)  # never more workers than frames
            print(f"  Rendering {total_frames} frames with {n_workers} thread(s)...")

            futures = []
            pbar = tqdm(total=total_frames, desc="  Rendering", leave=False)
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                for t, (fid, ds_idx) in enumerate(frame_entries):
                    single = ds[ds_idx]
                    # Inject shared fixed-crop geodata so DOP/DSM/crop_bounds are stable
                    if geodata_override is not None:
                        single.update(geodata_override)
                        single.pop('_dsm_raw', None)

                    w2c = single['extrinsics_raw'].cpu().numpy()
                    K  = single['intrinsics'].cpu().numpy()

                    cam_center = get_camera_center_from_w2c(w2c)
                    trajectory.append((float(cam_center[0]), float(cam_center[1])))

                    # Compute DOP-space visible mask from this frame's point_map
                    _pm_frame = single.get('point_map')
                    _cb_frame = single.get('crop_bounds')
                    vis_dop = None
                    if _pm_frame is not None and dsm_raw_seq is not None and _cb_frame is not None:
                        # Denormalize point_map if needed
                        _nt_f = single.get('norm_type', 'none')
                        _sc_f = single.get('dsm_scale')
                        _of_f = single.get('dsm_offset')
                        if _nt_f not in (None, 'none', 'None') and _sc_f is not None:
                            _pm_raw_f = denorm(_pm_frame, _sc_f, _of_f, _nt_f)
                        else:
                            _pm_raw_f = _pm_frame
                        vis_dop = compute_visible_dop_mask(
                            _pm_raw_f, dsm_raw_seq, _cb_frame,
                            w2c, K, img_h_uav, img_w_uav)

                    persistent, frame_matches = update_matches_for_frame(
                        persistent, w2c, K, img_h_uav, img_w_uav,
                        dsm_raw_seq, geo_mask_seq, visible_dop=vis_dop)
                    single['matches'] = frame_matches

                    corners = compute_frustum_corners(w2c, K, img_h_uav, img_w_uav, z_ground)
                    _cb = single.get('crop_bounds')
                    _tri_size = abs(_cb[2] - _cb[0]) * 0.05 if _cb is not None else 20.0
                    tri = compute_camera_triangle(w2c, size=_tri_size)

                    # Compute single-frame crop (what dataset would choose for this frame alone)
                    # Reconstruct 4x4 W2C (extrinsics_raw is 3x4)
                    w2c_44 = np.eye(4)
                    w2c_44[:3, :] = w2c[:3, :]
                    sf_fp = get_visible_footprint(w2c_44, K_raw_main, meta_main['height'], meta_main['width'],
                                                    plane_z=z_ground)
                    if sf_fp is not None and not any(np.isnan(sf_fp)):
                        sf_w = sf_fp[2] - sf_fp[0]
                        sf_h = sf_fp[3] - sf_fp[1]
                        sf_cx = (sf_fp[0] + sf_fp[2]) / 2
                        sf_cy = (sf_fp[1] + sf_fp[3]) / 2
                        sf_scale = max(sf_w * 1.2 / ds.size[1], sf_h * 1.2 / ds.size[0])
                        sf_scale = max(sf_scale, 0.1)  # DEFAULT_MIN_GSD
                        sf_scale = min(sf_scale, 2.0)  # DEFAULT_MAX_GSD
                        sf_half_w = sf_scale * ds.size[1] / 2
                        sf_half_h = sf_scale * ds.size[0] / 2
                        single['single_frame_crop'] = (
                            sf_cx - sf_half_w, sf_cy - sf_half_h,
                            sf_cx + sf_half_w, sf_cy + sf_half_h,
                        )

                    args_tuple = (
                        _torch_dict_to_numpy(single), tmp_dir, t,
                        (t, total_frames), corners,
                        list(trajectory), tri, cmap_ranges, vis_kwargs,
                    )
                    fut = executor.submit(_render_frame_worker_numpy, args_tuple)
                    fut.add_done_callback(lambda _f: pbar.update(1))
                    futures.append(fut)

                    # Drain completed futures periodically to free memory
                    if len(futures) >= n_workers * 4:
                        done, futures_set = concurrent.futures.wait(
                            futures[:n_workers * 2],
                            return_when=concurrent.futures.ALL_COMPLETED,
                        )
                        for f in done:
                            exc = f.exception()
                            if exc:
                                print(f"  [WARNING] Frame render error: {exc}")
                        futures = [f for f in futures if not f.done()]

                # Wait for all remaining futures
                for f in concurrent.futures.as_completed(futures):
                    exc = f.exception()
                    if exc:
                        print(f"  [WARNING] Frame render error: {exc}")
            pbar.close()

            if not args.paper:
                # Stitch PNGs into H.264 MP4
                fps = ds.samples[frame_entries[0][1]]['meta'].get('frame_rate', 30)
                mp4_path = os.path.join(args.output_dir, f"{vid_name}.mp4")

                first_frame_path = os.path.join(tmp_dir, 'frame_0000.png')
                if os.path.exists(first_frame_path):
                    ref_img = cv2.imread(first_frame_path)
                    if ref_img is not None:
                        ref_h, ref_w = ref_img.shape[:2]
                        ref_w = ref_w if ref_w % 2 == 0 else ref_w + 1
                        ref_h = ref_h if ref_h % 2 == 0 else ref_h + 1
                        scale_filter = f'scale={ref_w}:{ref_h}:flags=lanczos,setsar=1'
                    else:
                        scale_filter = 'pad=ceil(iw/2)*2:ceil(ih/2)*2'
                else:
                    scale_filter = 'pad=ceil(iw/2)*2:ceil(ih/2)*2'

                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', os.path.join(tmp_dir, 'frame_%04d.png'),
                    '-vf', scale_filter,
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-bf', '0',
                    '-crf', '18',
                    mp4_path,
                ]
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  Saved video: {mp4_path}")
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                else:
                    print(f"  ffmpeg failed (returncode={result.returncode}):")
                    print(result.stderr)
                    print(f"  Temp frames kept at: {tmp_dir}")
            else:
                print("  Skipping MP4 generation in paper mode.")
                shutil.rmtree(tmp_dir, ignore_errors=True)

            # Generate multi-year DOP comparison if available
            if seq_path_main is not None:
                visualize_multiyear_dops(seq_path_main, args.output_dir, dpi=args.dpi, paper=args.paper, fig_ext=args.fig_ext)

    print(f"Done. Outputs saved to {args.output_dir}")


if __name__ == '__main__':
    main()
