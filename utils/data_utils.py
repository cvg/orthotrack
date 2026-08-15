import json
from pathlib import Path


import torch
import torchvision.transforms.functional as TF

import random
import numpy as np
from typing import Dict, Any, Optional

from utils.tensor_ops import normalize
from utils.pose import decompose_pose, compute_w2c_translation, get_camera_center_from_w2c

def sample_color_jitter_params(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
    """Pre-sample color jitter parameters for consistent augmentation across frames.
    
    Returns a dict with brightness_factor, contrast_factor, saturation_factor, hue_factor."""
    return {
        'brightness_factor': 1.0 + random.uniform(-brightness, brightness),
        'contrast_factor': 1.0 + random.uniform(-contrast, contrast),
        'saturation_factor': 1.0 + random.uniform(-saturation, saturation),
        'hue_factor': random.uniform(-hue, hue),
    }


def sample_noise_augmentation_decisions(prob=0.5, force_all=False):
    """Pre-sample noise augmentation decisions for consistent augmentation across frames.
    
    Returns a dict with do_noise, do_erase, do_geo_drop, do_partial booleans."""
    return {
        'do_noise': force_all or random.random() > prob,
        'do_erase': force_all or random.random() > (1 - 0.3),  # 30% prob
        'do_geo_drop': force_all or random.random() > (1 - 0.2),  # 20% prob
        'do_partial': force_all or random.random() > (1 - 0.3),  # 30% prob
    }


def sample_noise_augmentation_params(img_H, img_W, geo_H, geo_W):
    """Pre-sample noise augmentation parameters for consistent augmentation across frames.
    
    Args:
        img_H, img_W: Query image dimensions
        geo_H, geo_W: Geodata (DOP/DSM) dimensions
    
    Returns a dict with noise_sigma, erase_patches, geo_drop_patches, partial_params."""
    # Gaussian noise sigma
    noise_sigma = random.uniform(0.01, 0.05)
    
    # Random erasing patches on query
    C = 3
    n_erase = random.randint(1, 3)
    erase_patches = []
    for _ in range(n_erase):
        eh = random.randint(int(img_H * 0.05), int(img_H * 0.2))
        ew = random.randint(int(img_W * 0.05), int(img_W * 0.2))
        ey = random.randint(0, img_H - eh)
        ex = random.randint(0, img_W - ew)
        fill_val = torch.rand(C, 1, 1) if random.random() > 0.5 else torch.zeros(C, 1, 1)
        erase_patches.append((ey, ex, eh, ew, fill_val))
    
    # Geodata dropout patches
    n_geo = random.randint(1, 2)
    geo_drop_patches = []
    for _ in range(n_geo):
        eh = random.randint(int(geo_H * 0.1), int(geo_H * 0.3))
        ew = random.randint(int(geo_W * 0.1), int(geo_W * 0.3))
        ey = random.randint(0, geo_H - eh)
        ex = random.randint(0, geo_W - ew)
        geo_drop_patches.append((ey, ex, eh, ew))
    
    # Partial overlap params
    crop_frac = random.uniform(0.2, 0.5)
    edge = random.randint(0, 3)
    
    return {
        'noise_sigma': noise_sigma,
        'erase_patches': erase_patches,
        'geo_drop_patches': geo_drop_patches,
        'partial_params': (edge, crop_frac),
    }


def augment_data(data: Dict[str, Any], prob: float = 0.5, force_all: bool = False,
                 force_hflip: Optional[bool] = None, force_vflip: Optional[bool] = None,
                 force_color: Optional[bool] = None,
                 color_jitter_params: Optional[Dict[str, float]] = None,
                 force_rotation: Optional[int] = None,
                 force_noise: Optional[bool] = None,
                 force_erase: Optional[bool] = None,
                 force_geo_drop: Optional[bool] = None,
                 force_partial: Optional[bool] = None,
                 noise_params: Optional[Dict[str, Any]] = None,
                 enable_noise: bool = True, enable_flip: bool = False,
                 enable_rotation: bool = True) -> Dict[str, Any]:
    """
    Apply augmentations to a data sample.
    
    Augmentations:
      - Image rotation by 0°/90°/180°/270° (default ON) — safe for PnP since
        all rotation matrices M_cam have det=+1 and stay in SO(3).
      - Horizontal / Vertical flips (default OFF) — flips change camera
        chirality (det(R)=-1), incompatible with PnP.
      - Color jitter on query image
      - Noise augmentations (Gaussian noise, random erasing, geodata dropout,
        partial geodata overlap)
    
    Args:
        enable_rotation: If True, randomly rotate images by 0/90/180/270°.
            Default True.  All rotations produce det(R')=+1 (proper rotations),
            fully compatible with PnP-based absolute pose estimation.
        enable_flip: If True, allow random H/V flips.  Default False because
            a single image flip creates a mirror camera (det(R)=-1) that cannot
            be recovered by PnP, causing ~90° rotation errors in pose metrics."""
    # Track augmentations
    data['aug_hflip'] = False
    data['aug_vflip'] = False
    data['aug_color'] = False
    H, W = data['query'].shape[-2:]

    # Horizontal Flip  (gated by enable_flip)
    do_hflip = False
    if enable_flip:
        do_hflip = force_hflip if force_hflip is not None else (force_all or random.random() > prob)
    if do_hflip:
        data['aug_hflip'] = True
        data['query'] = TF.hflip(data['query'])
        data['dop'] = TF.hflip(data['dop'])
        if 'dsm' in data: 
            data['dsm'] = TF.hflip(data['dsm'])
        if 'point_map' in data: 
            data['point_map'] = TF.hflip(data['point_map'])
            # Absolute UTM coordinates — do NOT negate on flip.
        if 'query_mask' in data: data['query_mask'] = TF.hflip(data['query_mask'])
        if 'geodata_mask' in data: data['geodata_mask'] = TF.hflip(data['geodata_mask'])
        if 'depth' in data: data['depth'] = TF.hflip(data['depth'])
        if 'normals' in data: data['normals'] = TF.hflip(data['normals'])

        # Flip Matching Maps (values and spatial)
        if 'query_in_dop' in data:
            # Spatial flip
            data['query_in_dop'] = TF.hflip(data['query_in_dop'])
            # Coordinate flip: these are relative displacements, U becomes -U
            data['query_in_dop'][0] *= -1 
        if 'mask_query_in_dop' in data: data['mask_query_in_dop'] = TF.hflip(data['mask_query_in_dop'])
        if 'mask_query_in_dop_oneway' in data: data['mask_query_in_dop_oneway'] = TF.hflip(data['mask_query_in_dop_oneway'])

        if 'dop_in_query' in data:
            data['dop_in_query'] = TF.hflip(data['dop_in_query'])
            # Relative displacement: U becomes -U
            data['dop_in_query'][0] *= -1
        if 'mask_dop_in_query' in data: data['mask_dop_in_query'] = TF.hflip(data['mask_dop_in_query'])
        if 'mask_dop_in_query_oneway' in data: data['mask_dop_in_query_oneway'] = TF.hflip(data['mask_dop_in_query_oneway'])
        
        # Flip Projected Elements
        if 'lod_uav' in data and data['lod_uav'] is not None and len(data['lod_uav']) > 0:
            data['lod_uav'][:, 0] = W - data['lod_uav'][:, 0] - 1
        if 'lod_dop' in data and data['lod_dop'] is not None and len(data['lod_dop']) > 0:
            data['lod_dop'][:, 0] = W - data['lod_dop'][:, 0] - 1
        if 'lidar_uav' in data and data['lidar_uav'] is not None and len(data['lidar_uav']) > 0:
            data['lidar_uav'][:, 0] = W - data['lidar_uav'][:, 0] - 1
        if 'matches' in data and isinstance(data['matches'], list):
            for m in data['matches']:
                m['uav'] = (W - m['uav'][0] - 1, m['uav'][1])
                m['dop'] = (W - m['dop'][0] - 1, m['dop'][1])

        # Keypoints are absolute UTM 3D coordinates — do NOT negate values.
        # (Spatial reordering is handled by the 2D pixel flip of projected keypoints above.)

        if 'intrinsics' in data:
            if data['intrinsics'].ndim == 3: # Sequence (L, 3, 3)
                W_img = data['query'].shape[3] # (L, C, H, W) -> H, W are last 2
                data['intrinsics'][:, 0, 2] = W_img - data['intrinsics'][:, 0, 2] - 1
            else:
                W_img = data['query'].shape[2]
                data['intrinsics'][0, 2] = W_img - data['intrinsics'][0, 2] - 1
        
        # Flip Pose (Camera X axis only, NOT world X axis)
        # Since point_map/dsm use absolute UTM coords (NOT negated),
        # we only apply the camera-side mirror: R' = M @ R, t' = M @ t
        # where M = diag(-1,1,1).  No world-axis S needed.
        if 'extrinsics' in data:
            if data['extrinsics'].ndim == 2: # (3, 4)
                data['extrinsics'][0, :] *= -1 # Row 0: M @ [R|t]
            elif data['extrinsics'].ndim == 3: 
                data['extrinsics'][:, 0, :] *= -1

        if 'R' in data:
             if data['R'].ndim == 2:
                 data['R'][0, :] *= -1 # Row 0: M @ R
             elif data['R'].ndim == 3: 
                 data['R'][:, 0, :] *= -1
        if 't' in data:
             # t' = M @ t
             if data['t'].ndim == 2: data['t'][0, :] *= -1 
             elif data['t'].ndim == 3: data['t'][:, 0, :] *= -1

    # Vertical Flip  (gated by enable_flip)
    do_vflip = False
    if enable_flip:
        do_vflip = force_vflip if force_vflip is not None else (force_all or random.random() > prob)
    if do_vflip:
        data['aug_vflip'] = True
        data['query'] = TF.vflip(data['query'])
        data['dop'] = TF.vflip(data['dop'])
        if 'dsm' in data: 
            data['dsm'] = TF.vflip(data['dsm'])
            # No negation for Y in absolute UTM.
        if 'point_map' in data: 
            data['point_map'] = TF.vflip(data['point_map'])
        if 'query_mask' in data: data['query_mask'] = TF.vflip(data['query_mask'])
        if 'geodata_mask' in data: data['geodata_mask'] = TF.vflip(data['geodata_mask'])
        if 'depth' in data: data['depth'] = TF.vflip(data['depth'])
        if 'normals' in data: data['normals'] = TF.vflip(data['normals'])

        # Flip Matching Maps
        if 'query_in_dop' in data:
            data['query_in_dop'] = TF.vflip(data['query_in_dop'])
            data['query_in_dop'][1] *= -1 # V becomes -V
        if 'mask_query_in_dop' in data: data['mask_query_in_dop'] = TF.vflip(data['mask_query_in_dop'])
        if 'mask_query_in_dop_oneway' in data: data['mask_query_in_dop_oneway'] = TF.vflip(data['mask_query_in_dop_oneway'])

        if 'dop_in_query' in data:
            data['dop_in_query'] = TF.vflip(data['dop_in_query'])
            data['dop_in_query'][1] *= -1 # V becomes -V
        if 'mask_dop_in_query' in data: data['mask_dop_in_query'] = TF.vflip(data['mask_dop_in_query'])
        if 'mask_dop_in_query_oneway' in data: data['mask_dop_in_query_oneway'] = TF.vflip(data['mask_dop_in_query_oneway'])
        
        # Flip Projected Elements
        if 'lod_uav' in data and data['lod_uav'] is not None and len(data['lod_uav']) > 0:
            data['lod_uav'][:, 1] = H - data['lod_uav'][:, 1] - 1
        if 'lod_dop' in data and data['lod_dop'] is not None and len(data['lod_dop']) > 0:
            data['lod_dop'][:, 1] = H - data['lod_dop'][:, 1] - 1
        if 'lidar_uav' in data and data['lidar_uav'] is not None and len(data['lidar_uav']) > 0:
            data['lidar_uav'][:, 1] = H - data['lidar_uav'][:, 1] - 1
        if 'matches' in data and isinstance(data['matches'], list):
            for m in data['matches']:
                m['uav'] = (m['uav'][0], H - m['uav'][1] - 1)
                m['dop'] = (m['dop'][0], H - m['dop'][1] - 1)

        # Keypoints are absolute UTM 3D coordinates — do NOT negate values.

        if 'intrinsics' in data:
            if data['intrinsics'].ndim == 3: # Sequence
                H_img = data['query'].shape[2]
                data['intrinsics'][:, 1, 2] = H_img - data['intrinsics'][:, 1, 2] - 1
            else:
                H_img = data['query'].shape[1]
                data['intrinsics'][1, 2] = H_img - data['intrinsics'][1, 2] - 1
                
        # Flip Pose (Camera Y axis only, NOT world Y axis)
        # R' = M_v @ R, t' = M_v @ t  where M_v = diag(1,-1,1)
        if 'extrinsics' in data:
            if data['extrinsics'].ndim == 2:
                data['extrinsics'][1, :] *= -1 # Row 1: M_v @ [R|t]
            elif data['extrinsics'].ndim == 3: 
                data['extrinsics'][:, 1, :] *= -1

        if 'R' in data:
             if data['R'].ndim == 2: 
                 data['R'][1, :] *= -1 # Row 1: M_v @ R
             elif data['R'].ndim == 3: 
                 data['R'][:, 1, :] *= -1
        if 't' in data:
             # t' = M_v @ t
             if data['t'].ndim == 2: data['t'][1, :] *= -1
             elif data['t'].ndim == 3: data['t'][:, 1, :] *= -1

    # ================================================================
    # Image Rotation Augmentation (0°, 90°, 180°, 270°)
    # Unlike flips, rotations preserve det(R)=+1 (proper rotations in SO(3)),
    # making them fully compatible with PnP-based pose estimation.
    #
    # For rotation by k * 90° CCW (matching torch.rot90 convention):
    #   - Pixel (u,v) → new location in rotated image
    #   - R' = M_cam @ R  where M_cam is a proper rotation (det=+1)
    #   - t' = M_cam @ t
    #   - K' = intrinsics with updated cx, cy (and swapped fx/fy for 90°/270°)
    # ================================================================
    data['aug_rotation'] = 0
    if enable_rotation:
        if force_rotation is not None:
            rot_k = force_rotation  # Use pre-determined rotation for sequence consistency
        elif force_all:
            rot_k = random.choice([1, 2, 3])  # always rotate when forced
        else:
            rot_k = random.choice([0, 1, 2, 3])  # 25% chance each

        if rot_k > 0:
            data['aug_rotation'] = rot_k * 90
            H, W = data['query'].shape[-2:]

            # 1. Rotate all spatial tensors
            for key in ['query', 'dop', 'dsm', 'point_map',
                        'query_mask', 'geodata_mask', 'depth', 'normals']:
                if key in data:
                    data[key] = torch.rot90(data[key], k=rot_k, dims=[-2, -1])

            # 2. Rotate matching maps (spatial rotation + displacement vectors)
            #    k=1 (90°CCW): (du,dv)→(dv,-du)
            #    k=2 (180°):   (du,dv)→(-du,-dv)
            #    k=3 (270°CCW): (du,dv)→(-dv,du)
            for mkey in ['query_in_dop', 'dop_in_query']:
                if mkey in data:
                    m = torch.rot90(data[mkey], k=rot_k, dims=[-2, -1])
                    du, dv = m[0].clone(), m[1].clone()
                    if rot_k == 1:   m[0], m[1] = dv, -du
                    elif rot_k == 2: m[0], m[1] = -du, -dv
                    elif rot_k == 3: m[0], m[1] = -dv, du
                    data[mkey] = m
            for mkey in ['mask_query_in_dop', 'mask_dop_in_query',
                         'mask_query_in_dop_oneway', 'mask_dop_in_query_oneway']:
                if mkey in data:
                    data[mkey] = torch.rot90(data[mkey], k=rot_k, dims=[-2, -1])

            # 3. Rotate projected pixel coordinates (N, 2) with columns [x, y]
            for pkey in ['lod_uav', 'lod_dop', 'lidar_uav']:
                if pkey in data and data[pkey] is not None and len(data[pkey]) > 0:
                    pts = data[pkey]
                    x, y = pts[:, 0].clone(), pts[:, 1].clone()
                    if rot_k == 1:   pts[:, 0], pts[:, 1] = y, W - 1 - x
                    elif rot_k == 2: pts[:, 0], pts[:, 1] = W - 1 - x, H - 1 - y
                    elif rot_k == 3: pts[:, 0], pts[:, 1] = H - 1 - y, x
                    data[pkey] = pts
            if 'matches' in data and isinstance(data['matches'], list):
                for m_entry in data['matches']:
                    for field in ['uav', 'dop']:
                        x, y = m_entry[field]
                        if rot_k == 1:   m_entry[field] = (y, W - 1 - x)
                        elif rot_k == 2: m_entry[field] = (W - 1 - x, H - 1 - y)
                        elif rot_k == 3: m_entry[field] = (H - 1 - y, x)

            # Keypoints are absolute UTM 3D coords — values unchanged.

            # 4. Update intrinsics
            #    k=1: fx'=fy, fy'=fx, cx'=cy,      cy'=W-1-cx
            #    k=2: fx'=fx, fy'=fy, cx'=W-1-cx,   cy'=H-1-cy
            #    k=3: fx'=fy, fy'=fx, cx'=H-1-cy,   cy'=cx
            if 'intrinsics' in data:
                K = data['intrinsics'].clone()
                if K.ndim == 2:  # (3, 3)
                    fx, fy = K[0, 0].clone(), K[1, 1].clone()
                    cx, cy = K[0, 2].clone(), K[1, 2].clone()
                    if rot_k == 1:
                        K[0, 0], K[1, 1] = fy, fx
                        K[0, 2], K[1, 2] = cy, W - 1 - cx
                    elif rot_k == 2:
                        K[0, 2], K[1, 2] = W - 1 - cx, H - 1 - cy
                    elif rot_k == 3:
                        K[0, 0], K[1, 1] = fy, fx
                        K[0, 2], K[1, 2] = H - 1 - cy, cx
                elif K.ndim == 3:  # (L, 3, 3) sequence
                    fx, fy = K[:, 0, 0].clone(), K[:, 1, 1].clone()
                    cx, cy = K[:, 0, 2].clone(), K[:, 1, 2].clone()
                    if rot_k == 1:
                        K[:, 0, 0], K[:, 1, 1] = fy, fx
                        K[:, 0, 2], K[:, 1, 2] = cy, W - 1 - cx
                    elif rot_k == 2:
                        K[:, 0, 2], K[:, 1, 2] = W - 1 - cx, H - 1 - cy
                    elif rot_k == 3:
                        K[:, 0, 0], K[:, 1, 1] = fy, fx
                        K[:, 0, 2], K[:, 1, 2] = H - 1 - cy, cx
                data['intrinsics'] = K

            # 5. Rotate camera pose: R' = M_cam @ R,  t' = M_cam @ t
            #    All M_cam have det=+1 (proper rotations), preserving SO(3).
            M_cam_dict = {
                1: torch.tensor([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]),
                2: torch.tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]),
                3: torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]),
            }
            M_cam = M_cam_dict[rot_k]

            if 'extrinsics' in data:
                ext = data['extrinsics']
                if ext.ndim == 2:
                    if ext.shape[0] == 3:  # (3, 4)
                        data['extrinsics'] = M_cam @ ext
                    elif ext.shape[0] == 4:  # (4, 4) homogeneous
                        R = ext[:3, :3]
                        t = ext[:3, 3:4]
                        new_R = M_cam @ R
                        new_t = M_cam @ t
                        new_ext = ext.clone()
                        new_ext[:3, :3] = new_R
                        new_ext[:3, 3:4] = new_t
                        data['extrinsics'] = new_ext
                elif ext.ndim == 3:
                    if ext.shape[1] == 3:  # (L, 3, 4)
                        data['extrinsics'] = M_cam.unsqueeze(0) @ ext
                    elif ext.shape[1] == 4:  # (L, 4, 4) homogeneous
                        R = ext[:, :3, :3]
                        t = ext[:, :3, 3:4]
                        new_R = M_cam.unsqueeze(0) @ R
                        new_t = M_cam.unsqueeze(0) @ t
                        new_ext = ext.clone()
                        new_ext[:, :3, :3] = new_R
                        new_ext[:, :3, 3:4] = new_t
                        data['extrinsics'] = new_ext

            if 'R' in data:
                R = data['R']
                if R.ndim == 2:
                    data['R'] = M_cam @ R
                elif R.ndim == 3:
                    data['R'] = M_cam.unsqueeze(0) @ R

            if 't' in data:
                t_val = data['t']
                if t_val.ndim == 2:   # (3, 1)
                    data['t'] = M_cam @ t_val
                elif t_val.ndim == 3: # (L, 3, 1)
                    data['t'] = M_cam.unsqueeze(0) @ t_val

    # Color Jitter
    do_color = force_color if force_color is not None else (force_all or random.random() > prob)
    if do_color:
        data['aug_color'] = True
        # Use pre-sampled params if provided (for sequence consistency), else sample fresh
        if color_jitter_params is not None:
            params = color_jitter_params
        else:
            params = sample_color_jitter_params()
        # Apply using functional API for deterministic transforms
        img = data['query']
        img = TF.adjust_brightness(img, params['brightness_factor'])
        img = TF.adjust_contrast(img, params['contrast_factor'])
        img = TF.adjust_saturation(img, params['saturation_factor'])
        img = TF.adjust_hue(img, params['hue_factor'])
        data['query'] = img
    
    # ================================================================
    # Noise-based augmentations (query image + geodata)
    # These help the refinement head learn robustness to:
    #   - Gaussian noise: sensor noise, compression artifacts
    #   - Random erasing: occlusions, missing regions
    #   - Geodata dropout: missing/corrupt DSM/DOP patches
    # ================================================================
    if enable_noise:
        data['aug_noise'] = False
        data['aug_erase'] = False
        data['aug_geodata_dropout'] = False
        
        # 1. Gaussian noise on query image (simulates sensor noise)
        do_noise = force_noise if force_noise is not None else (force_all or random.random() > prob)
        if do_noise and 'query' in data:
            data['aug_noise'] = True
            # Use pre-sampled sigma if provided, else sample fresh
            if noise_params and 'noise_sigma' in noise_params:
                sigma = noise_params['noise_sigma']
            else:
                sigma = random.uniform(0.01, 0.05)
            noise = torch.randn_like(data['query']) * sigma
            data['query'] = (data['query'] + noise).clamp(0, 1)
        
        # 2. Random erasing on query (simulates occlusions)
        do_erase = force_erase if force_erase is not None else (force_all or random.random() > (1 - 0.3))  # 30% probability
        if do_erase and 'query' in data:
            data['aug_erase'] = True
            C, H, W = data['query'].shape[-3:]
            # Use pre-sampled params if provided, else sample fresh
            if noise_params and 'erase_patches' in noise_params:
                patches = noise_params['erase_patches']
            else:
                n_patches = random.randint(1, 3)
                patches = []
                for _ in range(n_patches):
                    eh = random.randint(int(H * 0.05), int(H * 0.2))
                    ew = random.randint(int(W * 0.05), int(W * 0.2))
                    ey = random.randint(0, H - eh)
                    ex = random.randint(0, W - ew)
                    fill_val = torch.rand(C, 1, 1) if random.random() > 0.5 else torch.zeros(C, 1, 1)
                    patches.append((ey, ex, eh, ew, fill_val))
            for (ey, ex, eh, ew, fill_val) in patches:
                if data['query'].ndim == 4:  # (L, C, H, W)
                    data['query'][..., ey:ey+eh, ex:ex+ew] = fill_val
                else:
                    data['query'][:, ey:ey+eh, ex:ex+ew] = fill_val
        
        # 3. Geodata (DOP/DSM) patch dropout (simulates missing/corrupt geodata)
        #    Only zero REFERENCE data (DOP, DSM).  Keep GT point_map and
        #    geodata_mask intact so the loss still supervises those regions —
        #    the network should learn to hallucinate correct 3D coords where
        #    reference data is missing.  The U-Net validity_mask channel
        #    (derived from DSM != 0 in the model forward pass) tells the
        #    network which input regions are unreliable.
        do_geo_drop = force_geo_drop if force_geo_drop is not None else (force_all or random.random() > (1 - 0.2))  # 20% probability
        if do_geo_drop:
            data['aug_geodata_dropout'] = True
            H_g, W_g = data.get('dop', data.get('dsm', data['query'])).shape[-2:]
            # Use pre-sampled params if provided, else sample fresh
            if noise_params and 'geo_drop_patches' in noise_params:
                patches = noise_params['geo_drop_patches']
            else:
                n_patches = random.randint(1, 2)
                patches = []
                for _ in range(n_patches):
                    eh = random.randint(int(H_g * 0.1), int(H_g * 0.3))
                    ew = random.randint(int(W_g * 0.1), int(W_g * 0.3))
                    ey = random.randint(0, H_g - eh)
                    ex = random.randint(0, W_g - ew)
                    patches.append((ey, ex, eh, ew))
            for (ey, ex, eh, ew) in patches:
                sl = (Ellipsis, slice(ey, ey + eh), slice(ex, ex + ew))
                # Only zero reference data — NOT GT point_map or geodata_mask
                for key in ['dop', 'dsm']:
                    if key in data:
                        data[key][sl] = 0
        
        # 4. Partial geodata overlap (simulates UAV at edge of geodata coverage)
        # In real scenarios the UAV may fly partially outside the area covered
        # by the available DOP/DSM, so only part of the query image has valid
        # geodata.  We zero-out ONLY reference data (DOP, DSM) from one or
        # more edges.  GT point_map and geodata_mask stay intact so the loss
        # still supervises — the network learns to hallucinate 3D coords in
        # uncovered regions using context + the validity_mask input channel.
        data['aug_partial_overlap'] = False
        do_partial = force_partial if force_partial is not None else (force_all or random.random() > (1 - 0.3))  # 30% probability
        if do_partial:
            data['aug_partial_overlap'] = True
            H_img, W_img = data['query'].shape[-2:]
            # Use pre-sampled params if provided, else sample fresh
            if noise_params and 'partial_params' in noise_params:
                edge, crop_frac = noise_params['partial_params']
            else:
                crop_frac = random.uniform(0.2, 0.5)
                edge = random.randint(0, 3)
            if edge == 0:  # left
                cut = int(W_img * crop_frac)
                sl = (Ellipsis, slice(None), slice(0, cut))
            elif edge == 1:  # right
                cut = int(W_img * crop_frac)
                sl = (Ellipsis, slice(None), slice(W_img - cut, W_img))
            elif edge == 2:  # top
                cut = int(H_img * crop_frac)
                sl = (Ellipsis, slice(0, cut), slice(None))
            else:  # bottom
                cut = int(H_img * crop_frac)
                sl = (Ellipsis, slice(H_img - cut, H_img), slice(None))

            # Only zero reference data — NOT GT point_map or geodata_mask
            for key in ['dop', 'dsm']:
                if key in data:
                    data[key][sl] = 0
        
    return data

def normalize_data(data: Dict[str, Any], normalization: str = '01', xyz_normalization: str = 'mean_std') -> Dict[str, Any]:
    """
    Apply normalization to images and XYZ data."""
    # 4. Image Normalization
    if normalization == 'imagenet':
        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Determine if sequence (L, C, H, W) or single image (C, H, W)
        if data['query'].ndim == 4:
             MEAN = IMAGENET_MEAN.unsqueeze(0)
             STD = IMAGENET_STD.unsqueeze(0)
        else:
             MEAN = IMAGENET_MEAN
             STD = IMAGENET_STD
             
        data['query'] = (data['query'] - MEAN) / STD
        # DOP is usually always single image (C, H, W) even in sequence mode (map)
        # But check just in case
        if data['dop'].ndim == 4:
            data['dop'] = (data['dop'] - MEAN) / STD
        else:
            data['dop'] = (data['dop'] - IMAGENET_MEAN) / IMAGENET_STD

    # 5. XYZ Normalization
    if 'dsm' in data and xyz_normalization is not None:
        # Per-scene statistics for normalization
        dsm = data['dsm'] # (3, H, W) XYZ map

        # Calculate offset/scale if not provided
        dsm_offset = data.get('dsm_offset')
        dsm_scale = data.get('dsm_scale')

        if dsm_offset is None or (isinstance(dsm_offset, torch.Tensor) and dsm_offset.sum() == 0):
             valid_stats_mask = (dsm != 0).all(dim=0)
             if valid_stats_mask.sum() > 100:
                 if dsm.ndim == 4: valid_points = dsm.view(dsm.size(0), -1)
                 valid_points = dsm[:, valid_stats_mask]
             else:
                 valid_points = dsm.view(dsm.size(0), -1)

             if xyz_normalization in ('minmax_01', 'minmax_11'):
                 dsm_offset = valid_points.min(dim=1).values
                 per_axis_scale = (valid_points.max(dim=1).values - dsm_offset) + 1e-8
             else:
                 # mean_std (default)
                 dsm_offset = valid_points.mean(dim=1)
                 per_axis_scale = valid_points.std(dim=1) + 1e-8

             # Use isotropic scale (same value for all 3 axes) to preserve
             # perspective geometry.  With per-axis scale, the intrinsics become
             # invalid for reprojection/PnP because the anisotropic scaling
             # distorts the pinhole camera model.
             dsm_scale = per_axis_scale.max().expand(3)

        data['dsm'] = normalize(dsm, dsm_scale, dsm_offset, xyz_normalization)
        if 'point_map' in data:
            data['point_map'] = normalize(data['point_map'], dsm_scale, dsm_offset, xyz_normalization)
        if 'depth' in data:
            # Depth is single-channel (1, H, W), use only Z component of offset/scale
            depth_offset = dsm_offset[2] if dsm_offset.ndim == 1 and dsm_offset.shape[0] == 3 else dsm_offset
            depth_scale = dsm_scale[2] if dsm_scale.ndim == 1 and dsm_scale.shape[0] == 3 else dsm_scale
            data['depth'] = normalize(data['depth'], depth_scale, depth_offset, xyz_normalization)
            
        # Normalize Keypoints
        if 'keypoints' in data and data['keypoints'] is not None:
             # keypoints: List or Tensor (N, 3)
             kpts = data['keypoints']
             if isinstance(kpts, list):
                 kpts_list = [k for k in kpts]
                 if len(kpts_list) > 0:
                     if isinstance(kpts_list[0], torch.Tensor):
                         kpts_tensor = torch.cat(kpts_list, dim=0) if kpts_list[0].ndim > 1 else torch.stack(kpts_list)
                     else:
                         kpts_tensor = torch.tensor(np.array(kpts_list))
                 else:
                     kpts_tensor = None
             else:
                 kpts_tensor = kpts
            
             if kpts_tensor is not None and kpts_tensor.ndim == 2:
                 data['keypoints'] = normalize(kpts_tensor, dsm_scale, dsm_offset, xyz_normalization)
        
        # Normalize Pose if present
        if 'extrinsics' in data: # pose_world2query
            pose = data['extrinsics']
            
            # Decompose and normalize translation
            # If sequence (L, 4, 4)
            if pose.ndim == 3:
                # Iterate or vectorize
                # Not fully vectorized in utils yet, doing loop
                R_list, t_list = [], []
                norm_poses = []
                for i in range(len(pose)):
                    p = pose[i]
                    R_w2c, _ = decompose_pose(p)
                    # For absolute pose normalization we need Camera Center in World
                    cam_pos_world = get_camera_center_from_w2c(p)
                    cam_pos_norm = normalize(cam_pos_world, dsm_scale, dsm_offset, xyz_normalization)
                    # Recompute t_w2c from new C_norm
                    t_w2c_norm = compute_w2c_translation(R_w2c, cam_pos_norm)
                    
                    p_norm = torch.zeros((4, 4), device=p.device)
                    p_norm[:3, :3] = R_w2c
                    p_norm[:3, 3] = t_w2c_norm.squeeze()
                    p_norm[3, 3] = 1
                    norm_poses.append(p_norm)
                    R_list.append(R_w2c)
                    t_list.append(t_w2c_norm)
                
                data['extrinsics'] = torch.stack(norm_poses)
                data['R'] = torch.stack(R_list)
                data['t'] = torch.stack(t_list)
            else:
                R_w2c, _ = decompose_pose(pose)
                cam_pos_world = get_camera_center_from_w2c(pose)
                cam_pos_norm = normalize(cam_pos_world, dsm_scale, dsm_offset, xyz_normalization)
                t_w2c_norm = compute_w2c_translation(R_w2c, cam_pos_norm)
                
                extrinsics_norm = torch.zeros((3, 4), device=pose.device)
                extrinsics_norm[:3, :3] = R_w2c
                extrinsics_norm[:3, 3] = t_w2c_norm.squeeze()
                
                data['R'] = R_w2c.view(3, 3)
                data['t'] = t_w2c_norm.view(3, 1)
                data['extrinsics'] = extrinsics_norm
                
        # Metadata
        data['dsm_offset'] = dsm_offset
        data['dsm_scale'] = dsm_scale
        data['norm_type'] = xyz_normalization

    elif 'dsm' in data:
        # No XYZ normalization — store identity metadata so viz knows data is raw
        data['norm_type'] = 'none'

    return data
def load_ground_truth(data_dir: Path, sequence: str) -> dict:
    """Load the ground truth JSON for a sequence, generating it on the fly if needed."""
    gt_path = data_dir / sequence / 'ground_truth.json'
    if gt_path.exists():
        import json
        with open(gt_path, 'r') as f:
            return json.load(f)
            
    # Fallback to generate from MovingDrone published layout (video.mp4, poses.csv)
    import json
    import pandas as pd
    import cv2
    
    # 1. Read intrinsics
    with open(data_dir / sequence / 'intrinsics.json', 'r') as f:
        intr_dict = json.load(f)
        
    # 2. Extract video to images/ cache if missing
    images_dir = data_dir / sequence / 'images'
    if not images_dir.exists():
        print(f"  [load_ground_truth] Extracting {sequence} video to cache...")
        images_dir.mkdir(parents=True)
        cap = cv2.VideoCapture(str(data_dir / sequence / 'video.mp4'))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.imwrite(str(images_dir / f"{idx:05d}.jpg"), frame)
            idx += 1
        cap.release()
        
    # 3. Read poses and assemble the expected dict
    poses_df = pd.read_csv(data_dir / sequence / 'poses.csv')
    gt_data = {
        'intrinsics': intr_dict,
        'frames': []
    }
    for idx, row in poses_df.iterrows():
        gt_data['frames'].append({
            'image_path': f"images/{idx:05d}.jpg",
            # We don't have depth maps in this minimal dataset layout, just leave out depth_path
            # Provide pose: [tx, ty, tz, qw, qx, qy, qz]
            'pose': [row['x'], row['y'], row['z'], row['qw'], row['qx'], row['qy'], row['qz']]
        })
        
    return gt_data

