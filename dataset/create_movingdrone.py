"""
Create MovingDrone-format scenes from a trajectory + frames/mesh + geodata/priors.

Practical offline path (no live Google Earth Studio required):
  trajectory JSON + frames (or --mesh-dir render) + geodata tiles OR --priors-dir

Optional upstream:
  GES can still produce the trajectory JSON and/or footage frames.

Outputs a scene directory matching dataset/MovingDrone.py / scripts/run_tracking.py
(local layout uses scenes/, not sequences/).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Force EGL backend for headless rendering BEFORE importing Open3D
os.environ.setdefault('OPEN3D_RENDERING_BACKEND', 'egl')
# Suppress Mesa warnings on headless nodes
os.environ.setdefault('MESA_GL_VERSION_OVERRIDE', '4.5')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import cv2
import numpy as np
import pyproj
import rasterio
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import random
import xml.etree.ElementTree as ET
import shutil
import tempfile
import pathlib
import zipfile
from loguru import logger

from scripts.resample_trajectories import resample_trajectory, compute_ecef_speeds


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars/arrays (converts to Python native types)."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Open3D for rendering
try:
    import open3d as o3d
    import open3d.visualization.rendering as rendering
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # allow large DOP/DSM GeoTIFFs

from utils.geo import (
    GeoTIFFHandler,
    compute_z_offset_from_rendered_depth,
    get_tile_bounds_from_name,
    get_tile_bounds_from_geotiff,
    boxes_intersect,
    get_tile_bounds_from_gml_content,
    find_intersecting_tiles,
    compute_sequence_bbox,
)
from utils.pose import compute_intrinsics
from utils.lod import convert_lod_to_obj, convert_lod_tiles_parallel
from utils.image import is_frame_black, is_frame_sky_colored, is_frame_invalid
from utils.augmentation import (
    perturb_trajectory,
    apply_motion_blur,
    apply_sudden_jitter,
    apply_wind_gust_episodes,
    randomize_sun_direction,
    _generate_smooth_noise,
    _generate_multiband_noise,
)

# Repo root (script lives under dataset/)
_REPO_ROOT = Path(__file__).resolve().parent.parent


# ============================================================================
# Utilities
# ============================================================================

class MissingModalityError(RuntimeError):
    """Raised when a required modality is missing during dataset creation."""
    pass


def _citygml2obj_script() -> Path:
    return _REPO_ROOT / 'thirdparty' / 'CityGML2OBJv2' / 'CityGML2OBJs.py'


def _check_modality(modality_name: str, condition: bool, detail: str = ""):
    """
    Validate that a modality was successfully created.
    Always raises MissingModalityError if the condition is not met.

    Args:
        modality_name: Human-readable name (e.g. 'DOP', 'DSM', 'LoD2')
        condition: True if the modality is present/valid
        detail: Extra context for the message"""
    if condition:
        return
    msg = f"Missing modality '{modality_name}'"
    if detail:
        msg += f": {detail}"
    raise MissingModalityError(msg)


def _list_image_frames(directory: Path) -> List[Path]:
    """Sorted image files in a directory (non-recursive)."""
    if not directory or not Path(directory).exists():
        return []
    directory = Path(directory)
    exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    files: List[Path] = []
    for ext in exts:
        files.extend(directory.glob(ext))
    # Unique by resolved path, stable sort by name
    uniq = {f.resolve(): f for f in files}
    return sorted(uniq.values(), key=lambda p: p.name)


def resolve_frames_dir(
    frames_dir: Optional[str] = None,
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Locate an existing frame folder for offline video authoring.

    Search order:
      1. Explicit --frames-dir
      2. output/rendering (mesh render output)
      3. <trajectory_parent>/frames
      4. <trajectory_parent>/footage
      5. <trajectory_parent> itself (GES export: loose JPEGs next to JSON)
    """
    candidates: List[Path] = []
    if frames_dir:
        candidates.append(Path(frames_dir))
    if output_path is not None:
        candidates.append(Path(output_path) / 'rendering')
    if input_path is not None:
        parent = Path(input_path).parent
        candidates.extend([parent / 'frames', parent / 'footage', parent])

    for cand in candidates:
        frames = _list_image_frames(cand)
        if frames:
            logger.info(f"Using frames from {cand} ({len(frames)} images)")
            return cand
    return None


def stage_frames_for_encoding(
    src_dir: Path,
    dst_dir: Path,
    max_frames: Optional[int] = None,
) -> List[Path]:
    """
    Copy/rename frames into dst_dir as frame_0000.jpg, frame_0001.jpg, ...
    Returns the list of staged paths.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if dst_dir.exists() and dst_dir.resolve() != src_dir.resolve():
        shutil.rmtree(dst_dir, ignore_errors=True)
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_frames = _list_image_frames(src_dir)
    if max_frames is not None:
        src_frames = src_frames[:max_frames]
    if not src_frames:
        raise FileNotFoundError(f"No frames found in {src_dir}")

    staged: List[Path] = []
    same_dir = src_dir.resolve() == dst_dir.resolve()
    for i, src in enumerate(src_frames):
        dst = dst_dir / f"frame_{i:04d}.jpg"
        if same_dir and src.resolve() == dst.resolve():
            staged.append(dst)
            continue
        img = cv2.imread(str(src))
        if img is None:
            raise ValueError(f"Failed to read frame: {src}")
        cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        staged.append(dst)
    return staged


def copy_scene_priors(priors_dir: Path, output_path: Path) -> Dict[str, Any]:
    """
    Copy pre-extracted MovingDrone priors into the output scene.

    Expects (any subset of):
      dop/  (or dop_*.jpg / dop.jpg)
      dsm.npz, lod1.npz, lod2.npz, mesh.npz, lidar.npz
      optional meta.json (utm_offset / dops / dsm metadata reused when present)
    """
    priors_dir = Path(priors_dir)
    output_path = Path(output_path)
    if not priors_dir.exists():
        raise FileNotFoundError(f"Priors directory not found: {priors_dir}")

    copied: Dict[str, Any] = {'files': [], 'utm_offset': None, 'dsm_meta': None, 'dops': None}

    dop_src = priors_dir / 'dop'
    if dop_src.is_dir() and list(dop_src.glob('*.jpg')):
        dop_dst = output_path / 'dop'
        if dop_dst.exists():
            shutil.rmtree(dop_dst)
        shutil.copytree(dop_src, dop_dst)
        copied['files'].append('dop/')
        logger.info(f"Copied dop/ ({len(list(dop_dst.glob('*.jpg')))} crops)")
    else:
        # Legacy flat DOP files
        for pattern in ('dop_*.jpg', 'dop.jpg'):
            for f in priors_dir.glob(pattern):
                shutil.copy2(f, output_path / f.name)
                copied['files'].append(f.name)
                logger.info(f"Copied {f.name}")

    for fname in ('dsm.npz', 'lod1.npz', 'lod2.npz', 'mesh.npz', 'lidar.npz'):
        src = priors_dir / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)
            copied['files'].append(fname)
            logger.info(f"Copied {fname}")

    meta_src = priors_dir / 'meta.json'
    if meta_src.exists():
        try:
            with open(meta_src, 'r') as f:
                prior_meta = json.load(f)
            copied['utm_offset'] = prior_meta.get('utm_offset')
            copied['dsm_meta'] = prior_meta.get('dsm')
            copied['dops'] = prior_meta.get('dops')
            if copied['utm_offset'] is not None:
                logger.info(f"Reusing utm_offset from priors meta.json: {copied['utm_offset']}")
        except Exception as e:
            logger.warning(f"Could not read priors meta.json: {e}")

    if 'dsm.npz' not in copied['files']:
        raise MissingModalityError(
            f"Priors dir {priors_dir} has no dsm.npz (required for OrthoTrack scenes)"
        )
    has_dop = 'dop/' in copied['files'] or any(n.startswith('dop') for n in copied['files'])
    if not has_dop:
        raise MissingModalityError(
            f"Priors dir {priors_dir} has no dop/ crops (required for OrthoTrack scenes)"
        )
    return copied


def encode_video_from_frames(
    footage_dir: Path,
    video_path: Path,
    frame_rate: float,
    width: int,
    height: int,
    frame_files: Optional[List[Path]] = None,
    keep_rendering: bool = False,
) -> bool:
    """Encode frame_XXXX.jpg sequence to video.mp4. Returns True on success."""
    footage_dir = Path(footage_dir)
    video_path = Path(video_path)
    if frame_files is None:
        frame_files = sorted(footage_dir.glob('frame_*.jpg'))
    if not frame_files:
        logger.warning(f"No frame_*.jpg files in {footage_dir}")
        return False

    start_number = int(frame_files[0].stem.split('_')[-1])
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', str(frame_rate),
        '-start_number', str(start_number),
        '-i', str(footage_dir / 'frame_%04d.jpg'),
        '-frames:v', str(len(frame_files)),
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        '-bf', '0',
        '-pix_fmt', 'yuv420p',
        '-vf', f'scale={width}:{height}',
        str(video_path),
    ]

    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            logger.info(f"Saved: {video_path} (H.264)")
            if not keep_rendering:
                shutil.rmtree(footage_dir, ignore_errors=True)
                logger.info("Deleted rendering/ (use --keep-rendering to preserve frames)")
            return True
        logger.warning(f"ffmpeg encoding failed: {result.stderr[:200]}")
    except FileNotFoundError:
        logger.warning("ffmpeg not found, using OpenCV fallback")
    except Exception as e:
        logger.warning(f"ffmpeg encoding error: {e}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, frame_rate, (width, height))
    for frame_file in tqdm(frame_files, desc="Writing video (fallback)"):
        frame = cv2.imread(str(frame_file))
        if frame is not None:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            out.write(frame)
    out.release()
    if not keep_rendering:
        shutil.rmtree(footage_dir, ignore_errors=True)
    ok = video_path.exists() and video_path.stat().st_size > 0
    if ok:
        logger.info(f"Saved: {video_path} (OpenCV)")
    return ok


def verify_sequence(seq_dir: Path, expect_video: bool = True,
                    expect_depth: bool = True,
                    expect_normals: bool = True,
                    expect_geodata: bool = True,
                    expect_lod: bool = True,
                    destroy_on_fail: bool = True) -> Tuple[bool, List[str]]:
    """
    Verify that a generated sequence directory contains all required files.
    If verification fails and destroy_on_fail, the sequence directory is removed.

    Core files always required:
      - poses.csv, meta.json, intrinsics.json
    Optional by flags:
      - video.mp4, depth/, normals/, dop/, dsm.npz, lod1/2.npz
    """
    seq_dir = Path(seq_dir)
    issues = []

    if not seq_dir.exists():
        return False, [f"Sequence directory does not exist: {seq_dir}"]

    # --- Required files with minimum size checks ---
    required_files = {
        'poses.csv': 10,         # non-empty
        'meta.json': 10,         # non-empty
        'intrinsics.json': 10,   # non-empty
    }
    if expect_geodata:
        required_files['dsm.npz'] = 100
    if expect_lod:
        required_files['lod1.npz'] = 100
        required_files['lod2.npz'] = 100
    if expect_video:
        required_files['video.mp4'] = 1024  # at least 1KB

    for fname, min_size in required_files.items():
        fpath = seq_dir / fname
        if not fpath.exists():
            issues.append(f"Missing file: {fname}")
        elif fpath.stat().st_size < min_size:
            issues.append(f"File too small ({fpath.stat().st_size}B < {min_size}B): {fname}")

    # --- Warning-only files (non-fatal) ---
    lidar_path = seq_dir / 'lidar.npz'
    if not lidar_path.exists():
        logger.warning(f"{seq_dir.name} has no lidar.npz (optional)")
    elif lidar_path.stat().st_size < 100:
        logger.warning(f"{seq_dir.name} lidar.npz is suspiciously small ({lidar_path.stat().st_size}B)")

    if expect_lod:
        for lod_name in ('lod1.npz', 'lod2.npz'):
            if not (seq_dir / lod_name).exists():
                logger.warning(f"{seq_dir.name} missing optional-looking {lod_name} after expect_lod=True")
    else:
        for lod_name in ('lod1.npz', 'lod2.npz'):
            if not (seq_dir / lod_name).exists():
                logger.warning(f"{seq_dir.name} has no {lod_name} (LoD optional for tracking)")

    # --- DOP files ---
    if expect_geodata:
        dop_dir = seq_dir / 'dop'
        dop_files = list(dop_dir.glob('*.jpg')) if dop_dir.exists() else []
        if not dop_files:
            dop_files = list(seq_dir.glob('dop_*.jpg')) + list(seq_dir.glob('dop.jpg'))
        if not dop_files:
            issues.append("Missing DOP: no dop/<year>.jpg, dop_<year>.jpg, or dop.jpg found")
        else:
            for df in dop_files:
                if df.stat().st_size < 100:
                    issues.append(f"DOP file too small ({df.stat().st_size}B < 100B): {df.name}")

    # --- Depth maps ---
    if expect_depth:
        depth_dir = seq_dir / 'depth'
        if not depth_dir.exists():
            issues.append("Missing directory: depth/")
        else:
            depth_files = list(depth_dir.glob('depth_*.npz'))
            if len(depth_files) == 0:
                issues.append("No depth_*.npz files in depth/")

    # --- Normal maps ---
    if expect_normals:
        normals_dir = seq_dir / 'normals'
        if not normals_dir.exists():
            issues.append("Missing directory: normals/")
        else:
            normal_files = list(normals_dir.glob('normal_*.npz'))
            if len(normal_files) == 0:
                issues.append("No normal_*.npz files in normals/")

    # --- Validate poses.csv has data rows ---
    poses_path = seq_dir / 'poses.csv'
    num_poses = 0
    if poses_path.exists() and poses_path.stat().st_size >= 10:
        try:
            with open(poses_path, 'r') as f:
                line_count = sum(1 for _ in f)
            num_poses = line_count - 1  # subtract header
            if num_poses < 1:
                issues.append(f"poses.csv has no data rows (only {line_count} lines)")
        except Exception as e:
            issues.append(f"Cannot read poses.csv: {e}")

    # --- Cross-check: pose count must match video frame count ---
    video_path = seq_dir / 'video.mp4'
    if video_path.exists() and num_poses > 0:
        try:
            cap = cv2.VideoCapture(str(video_path))
            video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if video_frames > 0 and num_poses != video_frames:
                issues.append(
                    f"Frame count mismatch: poses.csv has {num_poses} rows "
                    f"but video.mp4 has {video_frames} frames"
                )
        except Exception:
            pass  # video read failure already caught by file size check

    # --- Cross-check: depth count must match pose count ---
    if expect_depth and num_poses > 0:
        depth_dir = seq_dir / 'depth'
        if depth_dir.exists():
            depth_count = len(list(depth_dir.glob('depth_*.npz')))
            if depth_count > 0 and depth_count != num_poses:
                issues.append(
                    f"Frame count mismatch: poses.csv has {num_poses} rows "
                    f"but depth/ has {depth_count} files"
                )

    # --- Verdict ---
    if issues:
        logger.error(f"VERIFICATION FAILED for: {seq_dir.name}")
        for issue in issues:
            logger.error(f"  ✗ {issue}")
        if destroy_on_fail:
            logger.error(f"Removing incomplete sequence: {seq_dir}")
            shutil.rmtree(seq_dir, ignore_errors=True)
        return False, issues

    logger.success(f"Sequence verified: {seq_dir.name} (required files present)")
    return True, []




# Sky color in BGR (for OpenCV) - light sky blue (135, 206, 235) 
SKY_COLOR_BGR = np.array([235, 206, 135], dtype=np.uint8)






class DOPCoverageError(RuntimeError):
    """Raised when DOP has insufficient coverage (too many black/missing pixels)."""
    pass


def check_dop_coverage(dop_img: np.ndarray, max_black_ratio: float = 0.15,
                       black_threshold: int = 10,
                       white_threshold: int = 250,
                       max_white_ratio: float = 0.95) -> Tuple[float, bool]:
    """
    Check if DOP image has sufficient coverage (not too many black or white pixels).
    
    WMS services outside their coverage area often return all-white images.
    This check rejects both all-black (missing data) and all-white (out-of-coverage)
    images.
    
    Args:
        dop_img: DOP image as numpy array (H, W, 3)
        max_black_ratio: Maximum allowed ratio of black pixels (default: 15%)
        black_threshold: Pixel intensity below which is considered black
        white_threshold: Pixel intensity above which is considered white
        max_white_ratio: Maximum allowed ratio of white pixels (default: 95%)
        
    Returns:
        Tuple of (invalid_ratio, is_valid) where invalid_ratio is the larger of
        black_ratio and white_ratio"""
    if dop_img is None:
        return 1.0, False
    
    # Calculate ratio of black (or very dark) pixels
    mean_intensity = dop_img.mean(axis=2)
    black_ratio = (mean_intensity < black_threshold).mean()
    
    # Calculate ratio of white (or near-white) pixels — detects WMS out-of-coverage
    white_ratio = (dop_img.min(axis=2) > white_threshold).mean()
    
    invalid_ratio = max(black_ratio, white_ratio)
    is_valid = (black_ratio <= max_black_ratio) and (white_ratio <= max_white_ratio)
    
    return invalid_ratio, is_valid


# ============================================================================
# Rendering Logic
# ============================================================================

def select_mesh_dirs_for_trajectory(
    mesh_root: Path,
    utm_positions,
    margin_m: float = 800.0,
    tile_size_m: float = 400.0,
) -> List[Path]:
    """
    Select VirtualCity-style mesh tile folders intersecting a trajectory.

    Tile folders are named like ``3995_58031_-002`` where the first two integers
    are UTM easting/northing in hectometers (metres / 100). Tiles are typically
    ~400 m on a side.
    """
    import re
    mesh_root = Path(mesh_root)
    utm_positions = np.asarray(utm_positions, dtype=float)
    if utm_positions.ndim != 2 or utm_positions.shape[0] == 0:
        return []

    min_e = float(utm_positions[:, 0].min()) - margin_m
    max_e = float(utm_positions[:, 0].max()) + margin_m
    min_n = float(utm_positions[:, 1].min()) - margin_m
    max_n = float(utm_positions[:, 1].max()) + margin_m

    pat = re.compile(r'^(\d+)_(\d+)_')
    selected: List[Path] = []
    subdirs = [p for p in mesh_root.iterdir() if p.is_dir()]
    # Flat directory of OBJs (no tile subfolders)
    if not subdirs:
        if list(mesh_root.glob('*.obj')) or list(mesh_root.glob('**/*.obj')):
            return [mesh_root]
        return []

    for p in subdirs:
        m = pat.match(p.name)
        if not m:
            # Non-conforming folder: keep if it contains OBJ (small custom meshes)
            if list(p.glob('*.obj')) or list(p.glob('**/*.obj')):
                selected.append(p)
            continue
        e0 = int(m.group(1)) * 100.0
        n0 = int(m.group(2)) * 100.0
        e1, n1 = e0 + tile_size_m, n0 + tile_size_m
        if boxes_intersect((min_e, min_n, max_e, max_n), (e0, n0, e1, n1)):
            selected.append(p)

    return sorted(selected, key=lambda x: x.name)


def load_meshes(mesh_dir, sr_enhancer=None, sr_batch_size=1, sr_textures=False):
    if not OPEN3D_AVAILABLE:
        logger.error("Open3D not available. Cannot load meshes.")
        return []
        
    logger.info("Loading meshes...")
    
    if isinstance(mesh_dir, (list, tuple)):
        # If it's a list, source_dir is just the parent of the first item
        source_dir = pathlib.Path(mesh_dir[0]).parent if mesh_dir else pathlib.Path(".")
    else:
        source_dir = pathlib.Path(mesh_dir)

    
    if sr_enhancer and sr_textures:
        # Create a temporary directory for enhanced meshes
        temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="enhanced_meshes_"))
        logger.info(f"Enhancing textures in temporary directory: {temp_dir}")
        
        # 1. Collect all textures
        texture_paths = list(source_dir.glob("**/*.jpg")) + list(source_dir.glob("**/*.jpeg")) + list(source_dir.glob("**/*.png"))
        texture_paths = sorted(list(set(texture_paths))) # Unique paths
        
        if texture_paths:
            logger.info(f"Enhancing {len(texture_paths)} textures with Real-ESRGAN (batch_size={sr_batch_size})...")
            enhanced_imgs = sr_enhancer.process_images_batched([str(p) for p in texture_paths], batch_size=sr_batch_size)
            
            for original_p, enhanced_pil in zip(texture_paths, enhanced_imgs):
                if enhanced_pil is None:
                    continue
                # Save as PNG to avoid re-compression artifacts
                out_p = temp_dir / (original_p.stem + ".png")
                enhanced_pil.save(out_p)
        
        # 2. Copy and update meshes
        for obj_p in source_dir.glob("**/*.obj"):
            shutil.copy(obj_p, temp_dir / obj_p.name)
        
        for mtl_p in source_dir.glob("**/*.mtl"):
            out_mtl = temp_dir / mtl_p.name
            with open(mtl_p, 'r') as f_in, open(out_mtl, 'w') as f_out:
                for line in f_in:
                    if line.strip().startswith("map_Kd"):
                        new_line = line.replace(".jpg", ".png").replace(".JPG", ".png").replace(".jpeg", ".png")
                        f_out.write(new_line)
                    else:
                        f_out.write(line)

        
        load_dir = temp_dir
    else:
        load_dir = source_dir

    models = []
    
    # Support list of files/directories or single directory
    if isinstance(mesh_dir, (list, tuple)):
        obj_paths = []
        for item in mesh_dir:
            p = pathlib.Path(item)
            if p.is_dir():
                obj_paths.extend(p.glob("**/*.obj"))  # Recursive search
            else:
                obj_paths.append(p)
    else:
        obj_paths = list(load_dir.glob("**/*.obj"))  # Recursive search

    logger.info(f"Found {len(obj_paths)} OBJ files to load")
    for p in obj_paths:
        try:
            m = o3d.io.read_triangle_model(str(p))
            models.append((p.name, m))
        except Exception as e:
            logger.warning(f"Failed to load {p.name}: {e}")
            
    return models


# ============================================================================
# Trajectory Perturbation & Realism
# ============================================================================















# Light sky blue (135, 206, 235) normalized to [0, 1]
DEFAULT_SKY_COLOR = [135/255, 206/255, 235/255, 1.0]

def render_sequence(models, json_path, output_dir, utm_zone=33,
                    high_quality=False, realism=False, lighting=False,
                    temperature=6500.0, sun_intensity=None, ibl_intensity=None,
                    sr_enhancer=None, sr_frames=False, max_frames=None,
                    background_color=None, save_depth=True,
                    save_normals=True,
                    random_sun=False, sun_seed=None,
                    low_light=False, high_bright=False,
                    exposure=1.0):
    """
    Render trajectory to a sequence of frames."""
    if background_color is None:
        background_color = DEFAULT_SKY_COLOR
    if not OPEN3D_AVAILABLE:
        print("Error: Open3D not available. Rendering skipped.")
        return

    from utils.pose import PoseLoader
    loader = PoseLoader(json_path, utm_zone=utm_zone)

    width = loader.width
    height = loader.height
    
    print(f"Rendering sequence to {output_dir} ({width}x{height})...")
    if sr_frames:
        print(f"Frame SR enabled: Final output will be {width*4}x{height*4}")

    # 1. Setup Renderer
    render = None
    render_failed = False
    try:
        render = rendering.OffscreenRenderer(width, height)
        render.scene.set_background(background_color)
    except Exception as e:
        print(f"Warning: Open3D OffscreenRenderer failed: {e}")
        print(f"         This usually means no GPU/EGL is available (headless node).")
        if save_depth:
            print(f"         Falling back to DEPTH-ONLY mode via raycasting (no RGB rendering).")
            render_failed = True
        else:
            print(f"         Skipping rendering entirely (no depth requested either).")
            return
    
    enable_realism = high_quality or realism
    enable_lighting = high_quality or lighting or realism
    
    if render and not render_failed:
        render.scene.view.set_post_processing(True)

        if enable_realism:
            render.scene.view.set_antialiasing(True)
            render.scene.view.set_sample_count(4) 
            render.scene.view.set_ambient_occlusion(True, True) 
            render.scene.view.set_shadowing(True, rendering.View.ShadowType.PCF)
            # Use FILMIC tone mapping for better color saturation and dynamic range.
            # ACES tends to wash out colors with high light intensity.
            cg_options = rendering.ColorGrading(
                rendering.ColorGrading.Quality.HIGH,
                rendering.ColorGrading.ToneMapping.FILMIC
            )
        else:
            render.scene.view.set_antialiasing(False)
            render.scene.view.set_ambient_occlusion(False, False)
            render.scene.view.set_shadowing(False, rendering.View.ShadowType.PCF)

    # 2. Origin Shift for Jitter Fix
    total_bbox = o3d.geometry.AxisAlignedBoundingBox()
    for name, model in models:
        total_bbox += model.meshes[0].mesh.get_axis_aligned_bounding_box()
        for mesh_part in model.meshes[1:]:
             total_bbox += mesh_part.mesh.get_axis_aligned_bounding_box()
    
    center = total_bbox.get_center()
    print(f"   Scene center: {center}. Shifting origin for stability.")

    T_origin_shift = np.eye(4)
    T_origin_shift[:3, 3] = -center

    for name, model in models:
        # Transform the model to local origin
        for mesh_part in model.meshes:
            mesh_part.mesh.transform(T_origin_shift)
            
        if render and not render_failed:
            if not enable_lighting:
                for mat in model.materials:
                    mat.shader = "defaultUnlit"
                    mat.base_color = [1.0, 1.0, 1.0, 1.0]
            
            render.scene.add_model(name, model)

    # Setup Raycasting Scene for high-accuracy depth
    ray_scene = None
    if save_depth:
        print("   Initializing Raycasting Scene for high-accuracy depth...")
        ray_scene = o3d.t.geometry.RaycastingScene()
        for name, model in models:
            for mesh_part in model.meshes:
                # Mesh parts have already been shifted by T_origin_shift above
                v_t = o3d.core.Tensor(np.asarray(mesh_part.mesh.vertices).astype(np.float32))
                f_t = o3d.core.Tensor(np.asarray(mesh_part.mesh.triangles).astype(np.uint32))
                ray_scene.add_triangles(v_t, f_t)

    # 3. Lighting
    if render and not render_failed and enable_lighting:
        if low_light:
            # Low-light / dusk / overcast scenario
            # Much lower intensities, cool-blue tint
            sun_dir = [0.2, 0.1, -0.3]  # low grazing angle
            intensity = 15000 if sun_intensity is None else sun_intensity
            sun_color = [0.7, 0.75, 0.9]  # blue-ish dusk tint
            ibl_val = 8000 if ibl_intensity is None else ibl_intensity
            print(f"   Low-light mode: sun={intensity}, ibl={ibl_val}")
        elif high_bright:
            # Very bright / overexposed bright-day scenario.
            # High sun intensity blows out highlights — matches the old 120k look.
            sun_dir, sun_az, sun_el = randomize_sun_direction(seed=sun_seed)
            print(f"   High-bright mode: azimuth={sun_az:.1f}°, elevation={sun_el:.1f}°")
            intensity = 140000 if sun_intensity is None else sun_intensity
            sun_color = [1.0, 0.98, 0.95]   # pure sunlight, slightly warm
            ibl_val = 50000 if ibl_intensity is None else ibl_intensity
            print(f"   High-bright mode: sun={intensity}, ibl={ibl_val}")
        elif random_sun:
            sun_dir, sun_az, sun_el = randomize_sun_direction(seed=sun_seed)
            print(f"   Random sun: azimuth={sun_az:.1f}°, elevation={sun_el:.1f}°")
            # Balanced intensity: 75k avoids blown-out highlights with FILMIC
            intensity = 75000 if sun_intensity is None else sun_intensity
            # Warm/cool color based on elevation (low sun = warmer)
            if sun_el < 30:
                sun_color = [1.0, 0.85, 0.65]  # golden hour
            elif sun_el > 55:
                sun_color = [0.9, 0.95, 1.0]   # overhead, slightly cool
            else:
                sun_color = [1.0, 1.0, 1.0]
            ibl_val = 25000 if ibl_intensity is None else ibl_intensity
        elif enable_realism:
            sun_dir = [0.5, 0.3, -1.0] 
            intensity = 75000 if sun_intensity is None else sun_intensity
            if temperature <= 4000: sun_color = [1.0, 0.75, 0.55]
            elif temperature >= 8000: sun_color = [0.65, 0.75, 1.0]
            else: sun_color = [1.0, 1.0, 1.0]
            ibl_val = 25000 if ibl_intensity is None else ibl_intensity
        else:
            sun_dir = [0.0, 0.0, -1.0] 
            intensity = 90000 if sun_intensity is None else sun_intensity
            sun_color = [1.0, 1.0, 1.0]
            ibl_val = 30000 if ibl_intensity is None else ibl_intensity

        render.scene.scene.enable_sun_light(True)
        render.scene.scene.set_sun_light(sun_dir, sun_color, intensity)
        render.scene.scene.enable_indirect_light(True)
        render.scene.scene.set_indirect_light_intensity(ibl_val)
    elif render and not render_failed:
        render.scene.scene.enable_sun_light(False)
        render.scene.scene.enable_indirect_light(False)

    # 4. Render Loop
    do_rgb = render and not render_failed
    
    frames_dir = Path(output_dir) / "rendering"
    if do_rgb:
        if frames_dir.exists():
            print(f"   Clearing existing frames in {frames_dir}")
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
    
    depth_dir = Path(output_dir) / "depth"
    if save_depth:
        if depth_dir.exists():
            print(f"   Clearing existing depth maps in {depth_dir}")
            shutil.rmtree(depth_dir)
        depth_dir.mkdir(parents=True, exist_ok=True)
    
    normals_dir = Path(output_dir) / "normals"
    if save_normals:
        if normals_dir.exists():
            print(f"   Clearing existing normals in {normals_dir}")
            shutil.rmtree(normals_dir)
        normals_dir.mkdir(parents=True, exist_ok=True)
    
    aspect_ratio = width / float(height)
    num_to_render = loader.num_frames
    if max_frames:
        num_to_render = min(num_to_render, max_frames)

    # Pre-compute intrinsics once if FOV is constant across frames
    first_fov = loader.get_fov_vertical(0)
    fov_is_constant = all(loader.get_fov_vertical(i) == first_fov for i in range(num_to_render))
    K_cached = compute_intrinsics(width, height, first_fov)  # Always compute at least for first FOV
    
    # Pre-allocate ray buffer to avoid per-frame allocation (~50MB instead of ~450MB/frame)
    if save_depth and ray_scene:
        ray_buffer = np.empty((height, width, 6), dtype=np.float32)
        # Pre-compute pixel grid (constant across frames)
        grid_y, grid_x = np.mgrid[0:height, 0:width]
        grid_y_f32 = grid_y.astype(np.float32)
        grid_x_f32 = grid_x.astype(np.float32)

    desc = "Rendering frames" if do_rgb else "Computing depth maps"
    for i in tqdm(range(num_to_render), desc=desc):
        pose = loader.get_pose(i)
        pos = pose.position
        pos_local = pos - center
        K = K_cached if fov_is_constant else compute_intrinsics(width, height, loader.get_fov_vertical(i))
        
        actual_R_c2w = pose.rotation_matrix
        forward = actual_R_c2w @ np.array([0, 0, 1])
        up = actual_R_c2w @ np.array([0, -1, 0])
        
        if do_rgb:
            # Set Camera
            render.scene.camera.set_projection(
                loader.get_fov_vertical(i), aspect_ratio, 1.0, 100000.0, 
                rendering.Camera.FovType.Vertical
            )
            
            # Standard look_at based on trajectory orientation
            target_local = pos_local + forward * 10000.0
            up_ortho = up - np.dot(up, forward) * forward
            up_ortho = up_ortho / np.linalg.norm(up_ortho)
            render.scene.camera.look_at(target_local, pos_local, up_ortho)

            # Capture Image
            img = render.render_to_image()
            img_np = np.asarray(img)
            
            if sr_frames and sr_enhancer:
                pil_img = Image.fromarray(img_np)
                enhanced_pil = sr_enhancer.process_batch([pil_img], auto_downsample=False)[0]
                img_np = np.array(enhanced_pil)

            frame_path = frames_dir / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(frame_path), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Capture Depth (works even without GPU renderer)
        if save_depth and ray_scene:
            C_local = pos_local.astype(np.float32)
            
            # Inline optimized ray generation - reuse pre-allocated buffer
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            x_norm = (grid_x_f32 - cx) / fx
            y_norm = (grid_y_f32 - cy) / fy
            
            # Directions in camera frame -> world frame (fused into buffer)
            R_T = actual_R_c2w.T.astype(np.float32)
            ray_buffer[:, :, 3] = x_norm * R_T[0, 0] + y_norm * R_T[1, 0] + R_T[2, 0]
            ray_buffer[:, :, 4] = x_norm * R_T[0, 1] + y_norm * R_T[1, 1] + R_T[2, 1]
            ray_buffer[:, :, 5] = x_norm * R_T[0, 2] + y_norm * R_T[1, 2] + R_T[2, 2]
            
            # Normalize directions in-place
            norms = np.sqrt(ray_buffer[:, :, 3]**2 + ray_buffer[:, :, 4]**2 + ray_buffer[:, :, 5]**2)
            ray_buffer[:, :, 3] /= norms
            ray_buffer[:, :, 4] /= norms
            ray_buffer[:, :, 5] /= norms
            
            # Origins (broadcast fill)
            ray_buffer[:, :, 0] = C_local[0]
            ray_buffer[:, :, 1] = C_local[1]
            ray_buffer[:, :, 2] = C_local[2]
            
            ray_tensor_t = o3d.core.Tensor(ray_buffer)
            
            ans = ray_scene.cast_rays(ray_tensor_t)
            t_hit = ans['t_hit'].numpy()
            
            # Mask out infinite distances (no hit)
            hit_mask = ~np.isinf(t_hit)
            t_hit[~hit_mask] = 0
            
            depth_path = depth_dir / f"depth_{i:04d}.npz"
            np.savez(depth_path, depth=t_hit.astype(np.float32))
            
            # Save surface normals to separate folder
            if save_normals:
                normals = ans['primitive_normals'].numpy()  # (H, W, 3)
                normals[~hit_mask] = 0
                normal_path = normals_dir / f"normal_{i:04d}.npz"
                np.savez(normal_path, normals=normals.astype(np.float32))

    print(f"Done. Results saved to {output_dir}")
    return center


# ============================================================================
# Geodata Tile Detection Functions
# ============================================================================





















# ============================================================================
# Geodata Preprocessing (merged from preprocess_sequence_geodata.py)
# ============================================================================



def preprocess_dop(seq_dir, geodata_base, dop_tiles, bbox):
    """Single-year DOP preprocessing. For new sequences, prefer preprocess_dop_multiyear()."""
    print("\n  Preprocessing DOP (legacy single-file mode)...")

    min_x, min_y, max_x, max_y = bbox
    dop_paths = [Path(geodata_base) / t for t in dop_tiles]

    if not dop_paths:
        print("    No DOP tiles found")
        return None

    h = GeoTIFFHandler(dop_path=str(dop_paths[0]))
    gsd = h.dop_resolution

    width_m = max_x - min_x
    height_m = max_y - min_y
    width_px = int(width_m / gsd)
    height_px = int(height_m / gsd)

    print(f"    Coverage: {width_m:.0f}m x {height_m:.0f}m")
    print(f"    GSD: {gsd:.2f}m/px")
    print(f"    Output size: {width_px}x{height_px}px")

    dop_buffer = np.zeros((height_px, width_px, 3), dtype=np.uint8)

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    for dop_p in tqdm(dop_paths, desc="    Cropping DOP tiles"):
        h = GeoTIFFHandler(dop_path=str(dop_p))
        tile = h.crop_fixed_pixels(center_x, center_y, width_px, height_px, is_dsm=False)
        if tile is not None and tile.data is not None:
            mask = (tile.data != 0).any(axis=-1)
            dop_buffer[mask] = tile.data[mask]

    # Validate DOP coverage before saving
    invalid_ratio, is_valid = check_dop_coverage(dop_buffer, max_black_ratio=0.15)
    print(f"    DOP invalid pixel ratio: {invalid_ratio*100:.1f}%")
    
    if not is_valid:
        raise DOPCoverageError(
            f"DOP has insufficient coverage: {invalid_ratio*100:.1f}% invalid pixels.\n"
            f"This usually means the trajectory is outside Berlin FIS Broker coverage area.\n"
            f"Sequences in Schönefeld, Brandenburg or other areas outside Berlin cannot "
            f"have valid DOP data."
        )

    dop_path = seq_dir / "dop.jpg"
    cv2.imwrite(str(dop_path), cv2.cvtColor(dop_buffer, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"    Saved: {dop_path} ({dop_path.stat().st_size / 1024 / 1024:.1f} MB)")

    return {
        'file': 'dop.jpg',
        'bounds': list(bbox),
        'gsd': float(gsd),
        'width': width_px,
        'height': height_px
    }


def preprocess_dop_multiyear(seq_dir, geodata_base, tiles, bbox, gm=None,
                            min_coverage=0.90, wms_cache_dir=None):
    """
    Crop and save DOPs from all available years as dop/<year>.jpg.
    Uses locally downloaded GeoTIFF tiles where available, falling back to
    Berlin GDI WMS for years without local tiles.
    Skips any DOP year below ``min_coverage`` (default 90%).
    
    Args:
        seq_dir: Output sequence directory
        geodata_base: Path to geodata base (map) directory
        tiles: Full tiles dict with keys like 'dop', 'dop_2010', 'dop_2023', etc.
        bbox: (min_x, min_y, max_x, max_y) bounding box
        gm: GeodataManager instance for metadata extraction
        min_coverage: Minimum fraction of valid (non-black, non-white) pixels
            required to accept a DOP crop (default 0.90 = 90%).
        wms_cache_dir: Optional path to persistent WMS tile cache (e.g. the map/
            directory).  When set, WMS responses are cached on a 500 m UTM grid
            so overlapping requests across sequences share the same tiles.
        
    Returns:
        List of dicts with per-year DOP metadata"""
    # Default cache dir to geodata_base (the shared map/ directory)
    if wms_cache_dir is None:
        wms_cache_dir = geodata_base
    from utils.geodata_berlin import DOP_GSD, DOP_REGION_BBOX
    
    print("\n  Preprocessing multi-year DOPs...")
    min_x, min_y, max_x, max_y = bbox
    
    # Check whether the bbox overlaps Berlin GDI DOP coverage (union of all
    # region bounding boxes).  If it doesn't, skip Berlin GDI WMS entirely to
    # avoid futile HTTP requests that always return blank images.
    berlin_min_x = min(b[0] for b in DOP_REGION_BBOX.values())
    berlin_min_y = min(b[1] for b in DOP_REGION_BBOX.values())
    berlin_max_x = max(b[2] for b in DOP_REGION_BBOX.values())
    berlin_max_y = max(b[3] for b in DOP_REGION_BBOX.values())
    bbox_in_berlin = not (max_x < berlin_min_x or min_x > berlin_max_x or
                          max_y < berlin_min_y or min_y > berlin_max_y)
    if not bbox_in_berlin:
        print("    Bbox is outside Berlin GDI coverage — skipping Berlin WMS for all years")
    
    # Collect all DOP keys: those with tiles AND GDI years (empty tiles → WMS)
    dop_keys = sorted([
        k for k in tiles.keys()
        if k.startswith('dop')
    ])
    
    if not dop_keys:
        print("    No DOP sources found")
        return []
    
    # Detect BB DOP year from HTML metadata
    bb_dop_year = None
    bb_tiles = tiles.get('dop', []) or tiles.get('dop_bb_fallback', [])
    if bb_tiles and gm is not None:
        # Extract tile IDs from paths like 'dop/33391-5820/dop_33391-5820.tif'
        bb_tile_ids = []
        for tp in bb_tiles:
            parts = Path(tp).parts
            if len(parts) >= 2 and parts[0] == 'dop':
                bb_tile_ids.append(parts[1])
        if bb_tile_ids:
            bb_dop_year = gm.detect_bb_dop_year(bb_tile_ids)
            if bb_dop_year:
                print(f"    Detected BB DOP year: {bb_dop_year}")
    
    dop_year_results = []
    saved_years = {}  # year -> (coverage, index in dop_year_results)
    
    # Process GDI DOPs first (dop_<year>), then BB DOP (dop)
    # so BB can replace GDI if it has better coverage for the same year
    gdi_keys = sorted([k for k in dop_keys if k not in ('dop', 'dop_bb_fallback')])
    bb_keys = [k for k in dop_keys if k in ('dop', 'dop_bb_fallback')]
    ordered_dop_keys = gdi_keys + bb_keys
    
    for dop_key in ordered_dop_keys:
        dop_tile_paths = tiles.get(dop_key, [])
        
        # Determine year from key
        year = None
        is_bb_dop = dop_key in ('dop', 'dop_bb_fallback')
        if is_bb_dop:
            if not dop_tile_paths:
                continue  # BB DOPs need tile files (no WMS fallback)
            year = bb_dop_year  # May be None
            if year is None:
                print(f"    BB DOP ({dop_key}): could not detect year, skipping")
                continue
        else:
            try:
                year = int(dop_key.split('_')[1])
            except (ValueError, IndexError):
                continue
        
        # Resolve tile paths
        dop_paths = [Path(geodata_base) / t for t in dop_tile_paths]
        valid_paths = [p for p in dop_paths if p.exists()]
        
        # Helper: attempt to read the DOP from locally-downloaded ZIP tiles.
        # Downloads the region ZIP for this year/bbox if not yet present, then
        # reads any JP2/TIF tiles (ECW will silently fail with GeoTIFFHandler).
        def _try_zip_tiles(yr, gm_inst, geodata_base_path):
            """Download + read ZIP tiles. Returns (dop_buffer, tiles_ok, gsd) or None."""
            if gm_inst is None:
                return None
            from utils.geodata_berlin import DOP_YEARS as _DY
            if yr not in _DY:
                return None
            print(f"    DOP {yr}: trying ZIP tile download as fallback...")
            tile_dir = gm_inst.download_dop_year(yr, bbox=bbox)
            if tile_dir is None:
                return None
            tile_files = gm_inst.get_filtered_tile_files(
                f"dop_{yr}", tile_dir, bounds=bbox)
            if not tile_files:
                print(f"    DOP {yr}: ZIP download OK but no tiles overlap bbox")
                return None
            _gsd = DOP_GSD
            _w_px = int((max_x - min_x) / _gsd)
            _h_px = int((max_y - min_y) / _gsd)
            _buf = np.zeros((_h_px, _w_px, 3), dtype=np.uint8)
            _cx, _cy = (min_x + max_x) / 2, (min_y + max_y) / 2
            _ok = 0
            for tf in tile_files:
                try:
                    _h = GeoTIFFHandler(dop_path=tf)
                    _tile = _h.crop_fixed_pixels(_cx, _cy, _w_px, _h_px, is_dsm=False)
                    if _tile is not None and _tile.data is not None:
                        _m = (_tile.data != 0).any(axis=-1)
                        _buf[_m] = _tile.data[_m]
                        _ok += 1
                except Exception:
                    pass  # ECW tiles will fail here — that's expected
            if _ok == 0:
                print(f"    DOP {yr}: ZIP tiles unreadable (ECW?), no data")
                return None
            print(f"    DOP {yr}: ZIP tiles OK ({_ok} readable)")
            return _buf, _ok, _gsd

        if not valid_paths:
            # No local tile files — try WMS for GDI DOPs
            if not is_bb_dop:
                if not bbox_in_berlin:
                    # Already logged once above; skip silently per year.
                    continue
                from utils.geodata_berlin import fetch_dop_wms_crop
                gsd = DOP_GSD
                width_px = int((max_x - min_x) / gsd)
                height_px = int((max_y - min_y) / gsd)
                print(f"    DOP {year}: no local tiles, fetching via WMS...")
                wms_img = fetch_dop_wms_crop(year, bbox, width_px, height_px,
                                             cache_dir=wms_cache_dir)
                if wms_img is not None:
                    dop_buffer = wms_img
                    tiles_ok = 1
                    print(f"    DOP {year}: WMS OK ({width_px}x{height_px})")
                else:
                    # WMS failed → try downloading ZIP tiles as last resort
                    zip_result = _try_zip_tiles(year, gm, geodata_base)
                    if zip_result is not None:
                        dop_buffer, tiles_ok, gsd = zip_result
                        width_px = int((max_x - min_x) / gsd)
                        height_px = int((max_y - min_y) / gsd)
                    else:
                        print(f"    DOP {year}: WMS + ZIP both failed, skipping")
                        continue
            else:
                print(f"    DOP {year}: no valid tile files found, skipping")
                continue
        else:
            # Tile-based processing
            gsd = DOP_GSD
            try:
                h = GeoTIFFHandler(dop_path=str(valid_paths[0]))
                gsd = h.dop_resolution
            except Exception:
                pass
            
            width_m = max_x - min_x
            height_m = max_y - min_y
            width_px = int(width_m / gsd)
            height_px = int(height_m / gsd)
            
            dop_buffer = np.zeros((height_px, width_px, 3), dtype=np.uint8)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            
            tiles_ok = 0
            for dop_p in valid_paths:
                try:
                    h = GeoTIFFHandler(dop_path=str(dop_p))
                    tile = h.crop_fixed_pixels(center_x, center_y, width_px, height_px, is_dsm=False)
                    if tile is not None and tile.data is not None:
                        mask = (tile.data != 0).any(axis=-1)
                        dop_buffer[mask] = tile.data[mask]
                        tiles_ok += 1
                except Exception as e:
                    pass  # Skip unreadable tiles (e.g. ECW)
            
            # WMS fallback for GDI DOPs with unreadable tiles (ECW format)
            if tiles_ok == 0 and not is_bb_dop and bbox_in_berlin:
                from utils.geodata_berlin import fetch_dop_wms_crop
                print(f"    DOP {year}: tiles unreadable (ECW?), fetching via WMS...")
                wms_img = fetch_dop_wms_crop(year, bbox, width_px, height_px,
                                             cache_dir=wms_cache_dir)
                if wms_img is not None:
                    dop_buffer = wms_img
                    tiles_ok = 1
                    print(f"    DOP {year}: WMS OK ({width_px}x{height_px})")
            
            # ZIP fallback if both tiles and WMS failed
            if tiles_ok == 0 and not is_bb_dop:
                zip_result = _try_zip_tiles(year, gm, geodata_base)
                if zip_result is not None:
                    dop_buffer, tiles_ok, gsd = zip_result
                    width_px = int((max_x - min_x) / gsd)
                    height_px = int((max_y - min_y) / gsd)
            
            if tiles_ok == 0:
                print(f"    DOP {year}: no tiles produced valid data, skipping")
                continue
        
        # Check coverage — skip DOPs that don't sufficiently cover the bbox
        invalid_ratio, is_valid = check_dop_coverage(dop_buffer, max_black_ratio=0.01,
                                                      max_white_ratio=0.01)
        coverage = float(1.0 - invalid_ratio)
        if coverage < min_coverage:
            if coverage < 0.05:
                print(f"    DOP {year}: essentially empty ({coverage*100:.0f}% coverage, likely outside data extent), skipping")
            else:
                print(f"    DOP {year}: only {coverage*100:.0f}% coverage (need {min_coverage*100:.0f}%), skipping")
            continue
        
        # Check if this year was already saved with better coverage
        if year in saved_years:
            old_coverage, old_idx = saved_years[year]
            if coverage <= old_coverage:
                print(f"    DOP {year} ({dop_key}): coverage {coverage*100:.0f}% <= existing {old_coverage*100:.0f}%, skipping")
                continue
            else:
                # Replace the old entry
                print(f"    DOP {year} ({dop_key}): coverage {coverage*100:.0f}% > existing {old_coverage*100:.0f}%, replacing")
                dop_year_results[old_idx] = None  # Will be filtered out later
        
        # Save into dop/ subdirectory
        dop_subdir = seq_dir / "dop"
        dop_subdir.mkdir(exist_ok=True)
        filename = f"dop/{year}.jpg"
        dop_path = seq_dir / filename
        cv2.imwrite(str(dop_path), cv2.cvtColor(dop_buffer, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Get capture date from metadata
        capture_date = str(year)
        if gm is not None:
            # Try to get detailed capture date from first tile
            if is_bb_dop:
                if bb_tile_ids:
                    meta = gm.get_tile_metadata('dop', bb_tile_ids[0])
                    if meta.get('capture_date', 'Unknown') != 'Unknown':
                        capture_date = meta['capture_date']
            else:
                meta = gm.get_tile_metadata(dop_key, 'berlin_gdi_region')
                if meta.get('capture_date', 'Unknown') != 'Unknown':
                    capture_date = meta['capture_date']
        
        # Determine TrueDOP status from the GDI dataset name.
        # TrueDOP = geometrically corrected (no building lean / shadow artefacts).
        # BB DOPs from geobasis-bb are standard orthos, not TrueDOPs.
        if is_bb_dop:
            _is_truedop = False
        else:
            from utils.geodata_berlin import DOP_YEARS as _DOP_YEARS
            _dataset_name = _DOP_YEARS.get(year, dop_key)
            _is_truedop = _dataset_name.startswith('truedop')

        result = {
            'year': year,
            'file': filename,
            'capture_date': capture_date,
            'is_truedop': _is_truedop,
            'gsd': float(gsd),
            'bounds': list(bbox),
            'width': width_px,
            'height': height_px,
            'coverage': coverage,
            'source': 'geobasis-bb' if is_bb_dop else 'berlin-gdi',
        }
        saved_years[year] = (coverage, len(dop_year_results))
        dop_year_results.append(result)
        
        size_mb = dop_path.stat().st_size / 1024 / 1024
        print(f"    DOP {year}: saved {filename} ({size_mb:.1f} MB, "
              f"coverage={coverage*100:.0f}%, GSD={gsd:.2f}m, src={dop_key})")
    
    # Filter out replaced entries and sort by year
    dop_year_results = [r for r in dop_year_results if r is not None]
    dop_year_results.sort(key=lambda r: r['year'])
    print(f"    Total: {len(dop_year_results)} DOP year(s) saved")
    
    if not dop_year_results:
        raise DOPCoverageError(
            f"No DOP year achieved 100% coverage for bbox "
            f"({min_x:.0f}, {min_y:.0f}) to ({max_x:.0f}, {max_y:.0f}).\n"
            f"Ensure all required DOP tiles are downloaded (check for timeouts)."
        )
    
    return dop_year_results


def preprocess_dsm(seq_dir, geodata_base, dsm_tiles, bbox):
    """Crop and save global DSM as NPZ. Returns (dsm_meta_dict, dsm_buffer)."""
    print("\n  Preprocessing DSM...")

    min_x, min_y, max_x, max_y = bbox
    dsm_paths = [Path(geodata_base) / t for t in dsm_tiles]

    if not dsm_paths:
        print("    No DSM tiles found")
        return None, None

    h = GeoTIFFHandler(dsm_path=str(dsm_paths[0]))
    gsd = h.dsm_resolution

    width_m = max_x - min_x
    height_m = max_y - min_y
    width_px = int(width_m / gsd)
    height_px = int(height_m / gsd)

    print(f"    Coverage: {width_m:.0f}m x {height_m:.0f}m")
    print(f"    GSD: {gsd:.2f}m/px")
    print(f"    Output size: {width_px}x{height_px}px")

    dsm_buffer = np.full((height_px, width_px), -9999.0, dtype=np.float32)

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    for dsm_p in tqdm(dsm_paths, desc="    Cropping DSM tiles"):
        h = GeoTIFFHandler(dsm_path=str(dsm_p))
        tile = h.crop_fixed_pixels(center_x, center_y, width_px, height_px, is_dsm=True)
        if tile is not None and tile.data is not None:
            mask = (tile.data > -9000)
            dsm_buffer[mask] = tile.data[mask]

    # Report DSM coverage diagnostics
    total_px = dsm_buffer.size
    valid_px = int(np.sum(dsm_buffer > -9000))
    coverage_pct = 100.0 * valid_px / max(total_px, 1)
    print(f"    DSM coverage: {valid_px}/{total_px} pixels ({coverage_pct:.1f}%)")
    if coverage_pct < 90.0:
        print(f"    ⚠ WARNING: DSM coverage is below 90% — some tiles may be missing or have holes!")

    dsm_path = seq_dir / "dsm.npz"
    np.savez_compressed(dsm_path,
                       height=dsm_buffer,
                       bounds=np.array(bbox, dtype=np.float64),
                       gsd=np.array(gsd, dtype=np.float32))

    print(f"    Saved: {dsm_path} ({dsm_path.stat().st_size / 1024 / 1024:.1f} MB)")

    dsm_meta = {
        'file': 'dsm.npz',
        'bounds': list(bbox),
        'gsd': float(gsd),
    }
    return dsm_meta, dsm_buffer


def generate_lidar_from_mesh(seq_dir, points_per_m2=50.0):
    """
    Generate a synthetic lidar.npz from mesh.npz (or lod2.npz fallback) by
    uniformly sampling points on triangle surfaces.

    This is used as a fallback when real ALS LiDAR data is unavailable for the
    region (e.g. Berlin/Brandenburg coverage gaps).

    Args:
        seq_dir: Path to the sequence directory (must contain mesh.npz or lod2.npz).
        points_per_m2: Approximate sampling density (points per square metre).

    Returns:
        True if lidar.npz was created, False otherwise."""
    seq_dir = Path(seq_dir)
    lidar_path = seq_dir / "lidar.npz"
    if lidar_path.exists():
        return True  # already exists

    # Choose best available mesh source
    mesh_path = seq_dir / "mesh.npz"
    lod2_path = seq_dir / "lod2.npz"
    source_path = None
    if mesh_path.exists() and mesh_path.stat().st_size > 100:
        source_path = mesh_path
    elif lod2_path.exists() and lod2_path.stat().st_size > 100:
        source_path = lod2_path

    if source_path is None:
        print("    No mesh.npz or lod2.npz available for LiDAR fallback")
        return False

    print(f"  Generating synthetic LiDAR from {source_path.name}...")

    data = np.load(source_path, allow_pickle=True)
    vertices = data['vertices'].astype(np.float64)
    faces = data['faces'].astype(np.int64)

    if len(vertices) == 0 or len(faces) == 0:
        print("    Mesh has no geometry")
        return False

    # Get triangle vertices: (F, 3, 3)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute per-triangle area
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = areas.sum()

    if total_area < 1e-6:
        print("    Mesh has near-zero total area")
        return False

    total_points = int(total_area * points_per_m2)
    # Cap to match typical real ALS density (real sequences have 10-50M points)
    max_points = 50_000_000
    if total_points > max_points:
        print(f"    Capping from {total_points:,} to {max_points:,} points")
        total_points = max_points

    print(f"    Mesh area: {total_area:,.0f} m² → sampling {total_points:,} points")

    # Sample faces proportional to area
    probs = areas / total_area
    rng = np.random.default_rng(42)
    # Process in chunks to avoid memory spikes with huge meshes
    chunk_size = 10_000_000
    all_points = []
    remaining = total_points
    while remaining > 0:
        n = min(chunk_size, remaining)
        sampled_faces = rng.choice(len(faces), size=n, p=probs)

        # Random barycentric coordinates
        r1 = rng.random(n).astype(np.float32)
        r2 = rng.random(n).astype(np.float32)
        sqrt_r1 = np.sqrt(r1)

        # Barycentric → Cartesian
        sv0 = v0[sampled_faces].astype(np.float32)
        sv1 = v1[sampled_faces].astype(np.float32)
        sv2 = v2[sampled_faces].astype(np.float32)

        u = 1.0 - sqrt_r1
        v = sqrt_r1 * (1.0 - r2)
        w = sqrt_r1 * r2

        chunk_pts = (u[:, None] * sv0 + v[:, None] * sv1 + w[:, None] * sv2)
        all_points.append(chunk_pts)
        remaining -= n
        del sampled_faces, r1, r2, sqrt_r1, sv0, sv1, sv2, chunk_pts

    points = np.concatenate(all_points, axis=0)

    # Save in same format as real LiDAR
    np.savez_compressed(lidar_path, points=points)
    size_mb = lidar_path.stat().st_size / 1024 / 1024
    print(f"    Saved synthetic LiDAR: {lidar_path} ({size_mb:.1f} MB, {len(points):,} points)")
    return True


def preprocess_lidar(seq_dir, geodata_base, lidar_tiles, bbox, utm_offset=None):
    """Load and save LiDAR points as NPZ. Applies utm_offset if provided."""
    print("\n  Preprocessing LiDAR...")

    if not lidar_tiles:
        print("    No LiDAR tiles found")
        return

    try:
        import laspy
    except ImportError:
        print("    Warning: laspy not installed, skipping LiDAR")
        return

    min_x, min_y, max_x, max_y = bbox
    lidar_paths = [Path(geodata_base) / t for t in lidar_tiles]

    all_points = []
    all_intensity = []
    all_classification = []

    for las_p in tqdm(lidar_paths, desc="    Loading LiDAR tiles"):
        try:
            las = laspy.read(str(las_p))
            x, y, z = las.x, las.y, las.z

            mask = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
            if not mask.any():
                continue

            pts = np.stack([x[mask], y[mask], z[mask]], axis=1)
            all_points.append(pts)

            if hasattr(las, 'intensity'):
                all_intensity.append(las.intensity[mask])
            if hasattr(las, 'classification'):
                all_classification.append(las.classification[mask])

        except Exception as e:
            print(f"    Warning: Failed to read {las_p.name}: {e}")

    if not all_points:
        print("    No LiDAR points in coverage area")
        return

    points = np.vstack(all_points).astype(np.float32)
    intensity = np.hstack(all_intensity).astype(np.uint16) if all_intensity else None
    classification = np.hstack(all_classification).astype(np.uint8) if all_classification else None

    # Apply UTM offset to shift to local coordinates
    if utm_offset is not None:
        offset = np.array(utm_offset, dtype=np.float32)
        points -= offset

    print(f"    Total points: {len(points):,}")

    lidar_path = seq_dir / "lidar.npz"
    save_dict = {'points': points}
    if intensity is not None:
        save_dict['intensity'] = intensity
    if classification is not None:
        save_dict['classification'] = classification

    np.savez_compressed(lidar_path, **save_dict)
    print(f"    Saved: {lidar_path} ({lidar_path.stat().st_size / 1024 / 1024:.1f} MB)")


def preprocess_lod(seq_dir, geodata_base, lod_obj_dirs, lod_level=2, utm_offset=None, bbox=None):
    """Load and merge LoD OBJ files into single NPZ. Applies utm_offset if provided."""
    print(f"\n  Preprocessing LoD{lod_level}...")

    if not lod_obj_dirs:
        print(f"    No LoD{lod_level} tiles found")
        return

    all_vertices = []
    all_faces = []
    all_normals = []
    all_labels = [] # Store semantic labels per face
    vertex_offset = 0

    # Margin for cropping (meters).  Must be generous enough that buildings
    # at the edge of the DOP/DSM bounds are fully included (some buildings
    # extend hundreds of metres and their vertices must all survive the
    # per-vertex crop).  A value of 500 m ensures the LoD always covers
    # at least the same spatial extent as the DOP and DSM.
    crop_margin = 500.0
    if bbox:
        min_x, min_y, max_x, max_y = bbox
        min_x -= crop_margin
        min_y -= crop_margin
        max_x += crop_margin
        max_y += crop_margin

    for obj_dir_rel in tqdm(lod_obj_dirs, desc=f"    Loading LoD{lod_level} tiles"):
        obj_path_full = Path(geodata_base) / obj_dir_rel

        if obj_path_full.is_dir():
             obj_files = list(obj_path_full.glob("*.obj"))
        else:
             # Handle flat directory structure: obj_dir_rel is a stem prefix
             parent = obj_path_full.parent
             stem = obj_path_full.name
             # Find files starting with stem and ending in .obj
             obj_files = list(parent.glob(f"{stem}*.obj"))
        
        if not obj_files:
            continue

        # Filter out "base" mesh if semantic parts exist to avoid duplication
        # e.g. if we have "LoD2_..._BE.obj" AND "LoD2_..._BE-WallSurface.obj",
        # the base file is a duplicate aggregate.
        # Semantic parts usually have a hyphenated suffix like "-WallSurface"
        semantic_files = [p for p in obj_files if '-' in p.name]
        if len(semantic_files) > 0:
            # We have semantic parts. Exclude files that are likely the base aggregate.
            # Base files usually match the stem exactly (plus .obj) or don't have the semantic suffix.
            filtered_files = []
            for p in obj_files:
                fname = p.name
                # Keep if it is a semantic part
                if p in semantic_files:
                    filtered_files.append(p)
                    continue
                
                # If it's the base file (stem + .obj), skip it
                # Logic: if stem matches exactly, or if it doesn't have a semantic suffix
                # (Assuming semantic files have a hyphen - check for "Surface")
                # Note: p.stem might contain suffixes if p is a semantic file.
                # We need to check against the prefix stem (the loop var).
                is_base = (p.name == f"{obj_dir_rel if obj_path_full.is_dir() else stem}.obj") or \
                          (p.name == f"{stem}.obj") or \
                          ("Surface" not in p.name and "Body" not in p.name) # Heuristic
                
                if is_base:
                    # Satisfy linter if needed, or just skip
                    pass
                    # print(f"    Skipping aggregate file: {p.name}")
                    continue
                filtered_files.append(p)
            obj_files = filtered_files

        for obj_path in obj_files:
            try:
                # Extract semantic label from filename (e.g. ...WallSurface.obj -> wall)
                fname = obj_path.name.lower()
                label = 'other'
                
                # Priority order to handle substrings (e.g. InteriorWallSurface vs WallSurface)
                if 'interiorwallsurface' in fname: label = 'wall_interior'
                elif 'outerceilingsurface' in fname: label = 'ceiling_outer'
                elif 'outerfloorsurface' in fname: label = 'floor_outer'
                elif 'wallsurface' in fname: label = 'wall'
                elif 'roofsurface' in fname: label = 'roof'
                elif 'groundsurface' in fname: label = 'ground'
                elif 'closuresurface' in fname: label = 'closure'
                elif 'ceilingsurface' in fname: label = 'ceiling'
                elif 'floorsurface' in fname: label = 'floor'
                elif 'door' in fname: label = 'door'
                elif 'window' in fname: label = 'window'
                elif 'wall' in fname: label = 'wall'
                elif 'roof' in fname: label = 'roof'
                elif 'ground' in fname: label = 'ground'
                elif 'closure' in fname: label = 'closure'
                
                vertices = []
                faces = []
                normals = []

                with open(obj_path, 'r') as f:
                    for line in f:
                        if line.startswith('v '):
                            vertices.append([float(x) for x in line.split()[1:4]])
                        elif line.startswith('vn '):
                            normals.append([float(x) for x in line.split()[1:4]])
                        elif line.startswith('f '):
                            face_verts = []
                            for vert_str in line.split()[1:]:
                                v_idx = int(vert_str.split('/')[0]) - 1
                                face_verts.append(v_idx) # Local index 0-based
                            if len(face_verts) >= 3:
                                faces.append(face_verts[:3])

                if not vertices:
                    continue

                vertices_np = np.array(vertices, dtype=np.float32)
                faces_np = np.array(faces, dtype=np.int32)
                
                
                # Apply UTM offset if provided
                if utm_offset is not None:
                     vertices_np -= np.array(utm_offset, dtype=np.float32)

                # Apply cropping if bbox provided
                if bbox:
                    # Check which vertices are inside the bbox
                    mask = (vertices_np[:, 0] >= min_x) & (vertices_np[:, 0] <= max_x) & \
                           (vertices_np[:, 1] >= min_y) & (vertices_np[:, 1] <= max_y)
                    
                    if not np.any(mask):
                        continue # Skip this OBJ entirely if outside
                        
                    # If partially inside, we need to reindex
                    # Filter vertices -> new indices
                    # Create mapping from old index to new index (-1 for removed)
                    old_to_new = np.full(len(vertices_np), -1, dtype=np.int32)
                    new_indices = np.where(mask)[0]
                    old_to_new[new_indices] = np.arange(len(new_indices))
                    
                    # Filter faces: keep only if all 3 vertices are inside
                    valid_faces = []
                    for face in faces_np:
                        if np.all(mask[face]):
                            valid_faces.append(old_to_new[face])
                    
                    if not valid_faces:
                        continue
                        
                    vertices_np = vertices_np[mask]
                    faces_np = np.array(valid_faces, dtype=np.int32)
                    
                    # Filter normals? (Normals are per-vertex usually in OBJ but loaded as flat list here)
                    # The simple loader assumes v and vn align or handles indices. 
                    # The current simple loader logic for `f` ignores normal indices (v/vt/vn) and assumes v only.
                    # So normals list collected from `vn` lines is just a pool, not strictly 1:1 with v unless specified.
                    # Since we don't use normal indices in `f` parsing above, let's skip rigorous normal cropping for now
                    # or just keep all normals? No, that breaks alignment if we assumed 1:1.
                    normals_np = np.zeros((0, 3), dtype=np.float32)

                # Offset faces by global vertex count
                faces_np += vertex_offset
                
                all_vertices.append(vertices_np)
                all_faces.append(faces_np)
                # Store one label per face
                all_labels.extend([label] * len(faces_np))
                
                vertex_offset += len(vertices_np)

            except Exception as e:
                print(f"    Warning: Failed to read {obj_path}: {e}")

    if not all_vertices:
        print(f"    No LoD{lod_level} geometry found in bbox")
        return

    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces)
    labels = np.array(all_labels)

    # Apply UTM offset to shift to local coordinates
    # Already applied per-mesh in loop!
    # if utm_offset is not None:
    #     offset = np.array(utm_offset, dtype=np.float32)
    #     vertices -= offset

    save_dict = {'vertices': vertices, 'faces': faces, 'labels': labels}
    
    print(f"    Total vertices: {len(vertices):,}")
    print(f"    Total faces: {len(faces):,}")

    lod_path = seq_dir / f"lod{lod_level}.npz"
    np.savez_compressed(lod_path, **save_dict)
    print(f"    Saved: {lod_path} ({lod_path.stat().st_size / 1024 / 1024:.1f} MB)")


def preprocess_mesh(seq_dir, geodata_base, mesh_tile_dirs, utm_offset=None, bbox=None):
    """Load and merge mesh OBJ tiles into single NPZ.

    Similar to preprocess_lod but for high-quality textured mesh tiles.
    Only geometry (vertices + faces) is stored; textures are discarded.

    Args:
        seq_dir: Sequence output directory
        geodata_base: Root geodata (map) directory
        mesh_tile_dirs: List of relative paths to mesh tile directories
        utm_offset: [x, y, z] offset to subtract from vertices
        bbox: [min_x, min_y, max_x, max_y] in local coords for cropping"""
    print(f"\n  Preprocessing mesh tiles...")

    if not mesh_tile_dirs:
        print(f"    No mesh tiles found")
        return

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    crop_margin = 500.0
    if bbox:
        min_x, min_y, max_x, max_y = bbox
        min_x -= crop_margin
        min_y -= crop_margin
        max_x += crop_margin
        max_y += crop_margin

    for tile_rel in tqdm(mesh_tile_dirs, desc="    Loading mesh tiles"):
        tile_path = Path(geodata_base) / tile_rel
        if not tile_path.is_dir():
            continue

        obj_files = list(tile_path.glob("*.obj"))
        if not obj_files:
            continue

        for obj_path in obj_files:
            try:
                vertices = []
                faces = []

                with open(obj_path, 'r') as f:
                    for line in f:
                        if line.startswith('v '):
                            vertices.append([float(x) for x in line.split()[1:4]])
                        elif line.startswith('f '):
                            face_verts = []
                            for vert_str in line.split()[1:]:
                                v_idx = int(vert_str.split('/')[0]) - 1
                                face_verts.append(v_idx)
                            if len(face_verts) >= 3:
                                # Triangulate quads/polygons
                                for i in range(1, len(face_verts) - 1):
                                    faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])

                if not vertices:
                    continue

                vertices_np = np.array(vertices, dtype=np.float32)
                faces_np = np.array(faces, dtype=np.int32)

                # Apply UTM offset
                if utm_offset is not None:
                    vertices_np -= np.array(utm_offset, dtype=np.float32)

                # Crop by bbox
                if bbox:
                    mask = (vertices_np[:, 0] >= min_x) & (vertices_np[:, 0] <= max_x) & \
                           (vertices_np[:, 1] >= min_y) & (vertices_np[:, 1] <= max_y)

                    if not np.any(mask):
                        continue

                    old_to_new = np.full(len(vertices_np), -1, dtype=np.int32)
                    new_indices = np.where(mask)[0]
                    old_to_new[new_indices] = np.arange(len(new_indices))

                    valid_faces = []
                    for face in faces_np:
                        if np.all(mask[face]):
                            valid_faces.append(old_to_new[face])

                    if not valid_faces:
                        continue

                    vertices_np = vertices_np[mask]
                    faces_np = np.array(valid_faces, dtype=np.int32)

                faces_np += vertex_offset
                all_vertices.append(vertices_np)
                all_faces.append(faces_np)
                vertex_offset += len(vertices_np)

            except Exception as e:
                print(f"    Warning: Failed to read {obj_path}: {e}")

    if not all_vertices:
        print(f"    No mesh geometry found in bbox")
        return

    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces)

    print(f"    Total vertices: {len(vertices):,}")
    print(f"    Total faces: {len(faces):,}")

    mesh_path = seq_dir / "mesh.npz"
    np.savez_compressed(mesh_path, vertices=vertices, faces=faces)
    print(f"    Saved: {mesh_path} ({mesh_path.stat().st_size / 1024 / 1024:.1f} MB)")


def save_intrinsics_json(seq_dir, width, height, fx, fy, cx, cy):
    """Save camera intrinsics as JSON."""
    json_path = seq_dir / "intrinsics.json"
    intrinsics = {
        'width': width,
        'height': height,
        'fx': round(fx, 6),
        'fy': round(fy, 6),
        'cx': round(cx, 6),
        'cy': round(cy, 6),
    }
    with open(json_path, 'w') as f:
        json.dump(intrinsics, f, indent=2)
    print(f"    Saved: {json_path}")


def preprocess_geodata(seq_dir, geodata_base, tiles, lod1_obj_paths, lod2_obj_paths,
                       poses_csv, width, height, fx, fy, cx, cy, margin=50, gm=None):
    """
    Run all geodata preprocessing for a sequence.
    Crops and saves DOP, DSM, LiDAR, and LoD data in fast-loading formats.
    Computes UTM offset (median of DSM XYZ) and applies to all stored spatial data.
    Also saves multi-year DOPs as dop/<year>.jpg.

    Returns:
        dict with 'dop', 'dsm' metadata, 'utm_offset', and 'dop_years' list"""
    print(f"\n{'='*60}")
    print(f"Preprocessing geodata for: {seq_dir.name}")
    print(f"{'='*60}")

    bbox = compute_sequence_bbox(poses_csv, margin_m=margin,
                                 depth_dir=seq_dir / 'depth',
                                 fx=fx, fy=fy, cx=cx, cy=cy,
                                 width=width, height=height,
                                 subsample_frames=4)
    print(f"Sequence bbox: ({bbox[0]:.0f}, {bbox[1]:.0f}) to ({bbox[2]:.0f}, {bbox[3]:.0f})")

    geodata_base = Path(geodata_base)

    # --- Re-discover tiles using the full frustum bbox ---
    # The initial tile discovery used camera positions ± 200m which can be
    # much smaller than the frustum-based bbox.  Re-discover to ensure all
    # modalities cover the full computed extent.
    expanded = find_intersecting_tiles(bbox, geodata_base, gm=gm)
    for key in expanded:
        old_set = set(tiles.get(key, []))
        merged = list(old_set | set(expanded[key]))
        if len(merged) > len(old_set):
            print(f"  Tile re-discovery: {key} {len(old_set)} -> {len(merged)} (+{len(merged)-len(old_set)})")
        tiles[key] = merged

    # -- Handle LoD OBJ conversion for newly discovered GML tiles ---
    citygml2obj_script = _citygml2obj_script()
    for lod_level_i, gml_key, obj_list in [(1, 'lod1', lod1_obj_paths), (2, 'lod2', lod2_obj_paths)]:
        new_gml_tiles = tiles.get(gml_key, [])
        existing_obj_set = set(obj_list)
        if new_gml_tiles and citygml2obj_script.exists():
            obj_root = geodata_base / f'LoDv{lod_level_i}_obj'
            for gml_rel in new_gml_tiles:
                gml_path = geodata_base / gml_rel
                if not gml_path.exists():
                    continue
                expected_obj_dir = f"LoDv{lod_level_i}_obj/{gml_path.stem}"
                if expected_obj_dir not in existing_obj_set:
                    ok = convert_lod_to_obj(gml_path, obj_root, citygml2obj_script, lod_level_i)
                    if ok:
                        obj_list.append(expected_obj_dir)
                        print(f"    Converted new LoD{lod_level_i} tile: {gml_path.stem}")

    dop_tiles = tiles.get('dop', [])
    dsm_tiles = tiles.get('dsm', [])
    lidar_tiles = tiles.get('als', [])
    mesh_tiles = tiles.get('mesh', [])

    dop_meta = None
    dsm_meta = None
    dsm_buffer = None

    # --- Validate tile availability for critical modalities ---
    _check_modality('DOP', bool(dop_tiles), 'no DOP tiles found for trajectory bbox')
    _check_modality('DSM', bool(dsm_tiles), 'no DSM tiles found for trajectory bbox')
    # Non-critical modalities: warn but do not abort
    if not lidar_tiles:
        print("  ⚠ WARNING: No ALS tiles found — LiDAR will be missing")
    if not lod1_obj_paths:
        print("  ⚠ WARNING: No LoD1 OBJ meshes — LoD1 will be missing")
    if not lod2_obj_paths:
        print("  ⚠ WARNING: No LoD2 OBJ meshes — LoD2 will be missing")
    if not mesh_tiles:
        print("  ⚠ WARNING: No mesh tiles found — mesh will be missing")

    # Note: preprocess_dop() (single dop.jpg) removed — all DOPs handled by
    # preprocess_dop_multiyear() as dop/<year>.jpg files.

    if dsm_tiles:
        dsm_meta, dsm_buffer = preprocess_dsm(seq_dir, geodata_base, dsm_tiles, bbox)

    # --- Compute UTM offset from DSM ---
    utm_offset = [0.0, 0.0, 0.0]
    local_bounds = None
    if dsm_buffer is not None:
        valid_z = dsm_buffer[dsm_buffer > -9000]
        median_z = float(np.median(valid_z)) if len(valid_z) > 0 else 0.0
        median_x = (bbox[0] + bbox[2]) / 2
        median_y = (bbox[1] + bbox[3]) / 2
        utm_offset = [median_x, median_y, median_z]
        print(f"\n  UTM offset: [{utm_offset[0]:.2f}, {utm_offset[1]:.2f}, {utm_offset[2]:.2f}]")

        # Re-save DSM with offset applied to bounds
        local_bounds = np.array([
            bbox[0] - utm_offset[0], bbox[1] - utm_offset[1],
            bbox[2] - utm_offset[0], bbox[3] - utm_offset[1]
        ], dtype=np.float64)
        # Shift height values by Z offset
        shifted_height = dsm_buffer.copy()
        shifted_height[shifted_height > -9000] -= utm_offset[2]
        dsm_path = seq_dir / "dsm.npz"
        np.savez_compressed(dsm_path,
                           height=shifted_height,
                           bounds=local_bounds,
                           gsd=np.array(dsm_meta['gsd'], dtype=np.float32))
        print(f"    Re-saved DSM with local coords: {dsm_path}")

        # Update DSM meta bounds to local
        dsm_meta['bounds'] = local_bounds.tolist()

    if lidar_tiles:
        try:
            preprocess_lidar(seq_dir, geodata_base, lidar_tiles, bbox, utm_offset=utm_offset)
        except Exception as e:
            print(f"  ⚠ WARNING: LiDAR preprocessing failed: {e}")

    if lod1_obj_paths:
        try:
            preprocess_lod(seq_dir, geodata_base, lod1_obj_paths, lod_level=1, utm_offset=utm_offset, bbox=local_bounds.tolist() if local_bounds is not None else None)
        except Exception as e:
            print(f"  ⚠ WARNING: LoD1 preprocessing failed: {e}")

    if lod2_obj_paths:
        try:
            preprocess_lod(seq_dir, geodata_base, lod2_obj_paths, lod_level=2, utm_offset=utm_offset, bbox=local_bounds.tolist() if local_bounds is not None else None)
        except Exception as e:
            print(f"  ⚠ WARNING: LoD2 preprocessing failed: {e}")

    if mesh_tiles:
        try:
            preprocess_mesh(seq_dir, geodata_base, mesh_tiles, utm_offset=utm_offset, bbox=local_bounds.tolist() if local_bounds is not None else None)
        except Exception as e:
            print(f"  ⚠ WARNING: Mesh preprocessing failed: {e}")

    save_intrinsics_json(seq_dir, width, height, fx, fy, cx, cy)

    # --- Multi-year DOPs ---
    dop_year_results = preprocess_dop_multiyear(seq_dir, geodata_base, tiles, bbox, gm=gm)
    
    # Update DOP year bounds to local coordinates
    if dsm_buffer is not None:
        for dyr in dop_year_results:
            dyr['bounds'] = [
                dyr['bounds'][0] - utm_offset[0],
                dyr['bounds'][1] - utm_offset[1],
                dyr['bounds'][2] - utm_offset[0],
                dyr['bounds'][3] - utm_offset[1],
            ]

    # --- Validate output files were actually created ---
    has_any_dop = any((seq_dir / d['file']).exists() for d in dop_year_results) if dop_year_results else False
    _check_modality('DOP output', has_any_dop,
                    f'no dop/<year>.jpg files created in {seq_dir}')
    _check_modality('DSM output', (seq_dir / 'dsm.npz').exists(),
                    f'dsm.npz not created in {seq_dir}')

    # --- LiDAR fallback: generate from mesh if ALS was unavailable ---
    if not (seq_dir / 'lidar.npz').exists():
        print(f"  ⚠ WARNING: No ALS-based lidar.npz — attempting mesh-based fallback...")
        try:
            if not generate_lidar_from_mesh(seq_dir):
                print(f"  ⚠ WARNING: LiDAR fallback also failed — lidar.npz will be missing")
        except Exception as e:
            print(f"  ⚠ WARNING: LiDAR fallback failed: {e}")

    # Non-critical: warn only
    if not (seq_dir / 'lod1.npz').exists():
        print(f"  ⚠ WARNING: lod1.npz not created in {seq_dir}")
    if not (seq_dir / 'lod2.npz').exists():
        print(f"  ⚠ WARNING: lod2.npz not created in {seq_dir}")
    if not (seq_dir / 'mesh.npz').exists():
        print(f"  ⚠ WARNING: mesh.npz not created in {seq_dir}")

    print(f"\nPreprocessing complete!")

    return {
        'dsm': dsm_meta,
        'utm_offset': utm_offset,
        'dop_years': dop_year_results,
        'tiles': tiles,
    }


# ============================================================================
# Main Conversion Function
# ============================================================================

def _write_poses_csv(json_data, transformer, output_path):
    """Write poses.csv from trajectory JSON data.
    
    Extracted into a helper so it can be called early (for depth-based bbox
    computation) as well as at the normal poses.csv creation step.
    
    Args:
        json_data: Parsed trajectory JSON
        transformer: WGS84 → UTM transformer
        output_path: Path to write the CSV file"""
    ecef_transformer_csv = pyproj.Transformer.from_crs('EPSG:4978', 'EPSG:4326', always_xy=True)
    
    all_lons = np.array([f['coordinate']['longitude'] for f in json_data['cameraFrames']])
    all_lats = np.array([f['coordinate']['latitude'] for f in json_data['cameraFrames']])
    all_xs, all_ys = transformer.transform(all_lons, all_lats)
    
    with open(output_path, 'w') as f:
        f.write("frame_id,x,y,z,qw,qx,qy,qz,latitude,longitude,altitude,fov_vertical,roll,pitch,yaw\n")
        
        for i, frame_data in enumerate(json_data['cameraFrames']):
            coord = frame_data['coordinate']
            rotation = frame_data['rotation']
            
            lon, lat = coord['longitude'], coord['latitude']
            x, y = all_xs[i], all_ys[i]
            z = coord['altitude']
            
            rx, ry, rz = rotation['x'], rotation['y'], rotation['z']
            R_json = Rotation.from_euler('XYZ', [rx, ry, rz], degrees=True).as_matrix()
            
            ecef_pos = frame_data['position']
            lon_ecef, lat_ecef, _ = ecef_transformer_csv.transform(ecef_pos['x'], ecef_pos['y'], ecef_pos['z'])
            
            lat_rad, lon_rad = np.deg2rad(lat_ecef), np.deg2rad(lon_ecef)
            sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
            sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
            
            R_ecef_to_enu = np.array([
                [-sin_lon,           cos_lon,            0],
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [cos_lat * cos_lon,  cos_lat * sin_lon,  sin_lat]
            ])
            
            R_cam_to_enu = R_ecef_to_enu @ R_json
            r_final = Rotation.from_matrix(R_cam_to_enu)
            
            quat = r_final.as_quat()  # (x, y, z, w)
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            euler = r_final.as_euler('xyz', degrees=True)
            roll, pitch, yaw = euler
            fov_vertical = frame_data['fovVertical']
            
            f.write(f"{i},{x:.6f},{y:.6f},{z:.6f},{qw:.8f},{qx:.8f},{qy:.8f},{qz:.8f},"
                    f"{lat:.10f},{lon:.10f},{z:.6f},{fov_vertical},{roll:.6f},{pitch:.6f},{yaw:.6f}\n")


def convert_dataset(input_file, output_root, geodata_dir=None, convert_lod=True,
                    dop_dir=None, dsm_dir=None, lod1_dir=None, lod2_dir=None, als_dir=None,
                    render_mesh=False, mesh_dir=None,
                    high_quality=False, realism=False,
                    lighting=False, temperature=6500.0,
                    sun_intensity=None, ibl_intensity=None,
                    background_color=None,
                    save_depth=True, save_normals=True, max_frames=None,
                    keep_rendering=False,
                    sr_mode='none', sr_batch_size=1,
                    auto_geodata=False, map_dir=None,
                    geodata_margin=50.0,
                    exposure=1.0,
                    trajectory_noise=False, position_noise_m=0.15,
                    rotation_noise_deg=0.03, roll_noise_deg=0.05,
                    bank_on_turns=True, bank_factor=0.15,
                    noise_smoothness=15, noise_seed=None,
                    motion_blur=False, blur_strength=1.0,
                    random_sun=False, sun_seed=None,
                    low_light=False, high_bright=False,
                    sudden_jitter=False, jitter_frame=None,
                    jitter_probability=0.2, jitter_magnitude_deg=2.0,
                    jitter_seed=None,
                    wind_gusts=False, gust_probability=0.4,
                    n_episodes_max=2, episode_duration_s=4.0,
                    gust_magnitude_deg=0.35, gust_seed=None,
                    target_fps=None,
                    velocity_bound=True, max_speed_kmh=100.0,
                    frames_dir=None, priors_dir=None, video_src=None):


    """
    Create a MovingDrone scene directory from a trajectory JSON.

    Offline-friendly inputs:
      - frames_dir / GES frames next to the JSON → video.mp4
      - video_src → copy existing video.mp4
      - mesh_dir + render_mesh → Open3D render
      - priors_dir → copy pre-extracted dop/dsm/lod npz
      - geodata_dir / auto_geodata → crop from Berlin-style tile trees

    Width, height, fps and FOV are read from the trajectory JSON.
    """
    input_path = Path(input_file)
    output_path = Path(output_root)
    if map_dir: map_dir = Path(map_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    priors_copied = None
    used_existing_video = False
    
    # Find the JSON file
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No JSON file found in {input_path}")
    logger.info(f"Using JSON file: {input_path}")
    
    # Load JSON data
    with open(input_path, 'r') as f:
        json_data = json.load(f)

    # ===== Velocity Bounding (before any other processing) =====
    # Resample trajectory so max speed is bounded. Save preprocessed copy.
    if velocity_bound and max_speed_kmh is not None:
        fps = json_data['frameRate']
        speeds = compute_ecef_speeds(json_data['cameraFrames'], fps)
        current_max_kmh = speeds.max() * 3.6
        if current_max_kmh > max_speed_kmh:
            print(f"\n--- Velocity bounding: {current_max_kmh:.0f} km/h → ≤{max_speed_kmh:.0f} km/h ---")
            json_data = resample_trajectory(
                json_data, target_max_speed_kmh=max_speed_kmh,
                noise_level=0.0, verbose=True,
            )
        else:
            print(f"\n--- Velocity OK: {current_max_kmh:.0f} km/h ≤ {max_speed_kmh:.0f} km/h (no resampling) ---")

        # Save preprocessed trajectory
        preproc_dir = input_path.parent.parent / 'trajectories_preprocessed'
        preproc_dir.mkdir(parents=True, exist_ok=True)
        preproc_path = preproc_dir / input_path.name
        with open(preproc_path, 'w') as f:
            json.dump(json_data, f, cls=NumpyJSONEncoder)
        print(f"    Saved preprocessed trajectory → {preproc_path}")

    # ===== Trajectory Perturbation (before any processing) =====
    if trajectory_noise:
        print("\n--- Applying trajectory perturbation ---")
        json_data = perturb_trajectory(
            json_data,
            position_noise_m=position_noise_m,
            rotation_noise_deg=rotation_noise_deg,
            roll_noise_deg=roll_noise_deg,
            bank_on_turns=bank_on_turns,
            bank_factor=bank_factor,
            seed=noise_seed,
        )

    # ===== Sudden Jitter (single sharp disturbance) =====
    jitter_applied_frame = None
    gust_episodes = None
    if sudden_jitter:
        print("\n--- Applying sudden jitter ---")
        json_data, jitter_applied_frame = apply_sudden_jitter(
            json_data,
            jitter_frame=jitter_frame,
            jitter_probability=jitter_probability,
            jitter_magnitude_deg=jitter_magnitude_deg,
            seed=jitter_seed,
        )

    # ===== Wind Gust Episodes (sustained episodic shake) =====
    if wind_gusts:
        print("\n--- Applying wind gust episodes ---")
        json_data, gust_episodes = apply_wind_gust_episodes(
            json_data,
            probability=gust_probability,
            n_episodes_max=n_episodes_max,
            episode_duration_s=episode_duration_s,
            gust_magnitude_deg=gust_magnitude_deg,
            seed=gust_seed,
        )
    
    # Extract name from JSON, but use output directory as fallback for "Unbenannt" or empty names
    json_name = json_data.get('name', '').strip()
    if not json_name or json_name.lower() in ['unbenannt', 'unnamed', 'untitled']:
        # Use output directory name as a better fallback
        name = output_path.name
        print(f"JSON name '{json_name}' is generic, using output directory name: {name}")
    else:
        name = json_name
    
    width = json_data['width']
    height = json_data['height']
    # NOTE: GES 'numFrames' is actually the last frame INDEX (0-based),
    # so the real frame count is len(cameraFrames) = numFrames + 1.
    num_frames = len(json_data['cameraFrames'])
    frame_rate = json_data['frameRate']

    # ===== FPS Downsampling =====
    if target_fps and target_fps < frame_rate:
        step = max(1, round(frame_rate / target_fps))
        original_count = num_frames
        json_data['cameraFrames'] = json_data['cameraFrames'][::step]
        num_frames = len(json_data['cameraFrames'])
        actual_fps = frame_rate / step
        frame_rate = actual_fps
        json_data['frameRate'] = frame_rate
        print(f"FPS downsampling: {json_data.get('_original_fps', json_data['frameRate'])} -> {actual_fps:.1f} fps "
              f"(step={step}, {original_count} -> {num_frames} frames)")

    # ===== Max Frames Truncation =====
    # Apply max_frames to cameraFrames BEFORE rendering so that the rendered
    # frames, poses.csv, depth maps, and video are all consistent.
    if max_frames and num_frames > max_frames:
        print(f"Max frames truncation: {num_frames} -> {max_frames} frames")
        json_data['cameraFrames'] = json_data['cameraFrames'][:max_frames]
        num_frames = max_frames

    print(f"Dataset: {name}")
    print(f"Size: {width}x{height}")
    print(f"Frames: {num_frames}")
    print(f"Frame rate: {frame_rate}")
    
    # Create transformer for lat/lon to UTM
    # Auto-detect UTM zone from first frame longitude
    lon_init = json_data['cameraFrames'][0]['coordinate']['longitude']
    utm_zone = int((lon_init + 180) / 6) + 1

    target_crs = f'EPSG:326{utm_zone:02d}'
    transformer = pyproj.Transformer.from_crs('EPSG:4326', target_crs, always_xy=True)
    ecef_to_utm = pyproj.Transformer.from_crs('EPSG:4978', target_crs, always_xy=True)
    print(f"Using UTM Zone {utm_zone} (CRS: {target_crs})")

    # ===== Save trajectory visualization (XY + Z) before rendering =====
    try:
        _frames = json_data['cameraFrames']
        _lats = [f['coordinate']['latitude'] for f in _frames]
        _lons = [f['coordinate']['longitude'] for f in _frames]
        _alts = [f['coordinate']['altitude'] for f in _frames]
        _utm_xy = [transformer.transform(lon, lat) for lon, lat in zip(_lons, _lats)]
        _xs = [p[0] for p in _utm_xy]
        _ys = [p[1] for p in _utm_xy]
        _ts = np.arange(len(_frames)) / frame_rate  # time in seconds

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{name} — trajectory ({num_frames} frames, {frame_rate:.0f} fps)', fontsize=13)

        # Left: XY plan view
        ax = axes[0]
        sc = ax.scatter(_xs, _ys, c=_ts, cmap='viridis', s=3, linewidths=0)
        ax.plot(_xs[0], _ys[0], 'go', markersize=8, label='Start')
        ax.plot(_xs[-1], _ys[-1], 'r^', markersize=8, label='End')
        ax.set_xlabel('UTM Easting (m)')
        ax.set_ylabel('UTM Northing (m)')
        ax.set_title('XY Trajectory')
        ax.set_aspect('equal')
        ax.legend(fontsize=9)
        cb = fig.colorbar(sc, ax=ax, label='Time (s)', shrink=0.8)

        # Right: Altitude over time
        ax = axes[1]
        ax.plot(_ts, _alts, color='steelblue', linewidth=1.2)
        ax.fill_between(_ts, _alts, alpha=0.15, color='steelblue')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Altitude (m)')
        ax.set_title('Altitude Profile')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        poses_pdf = output_path / 'poses.pdf'
        fig.savefig(str(poses_pdf), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Trajectory plot saved: {poses_pdf}")
    except Exception as e:
        print(f"Warning: could not save trajectory plot: {e}")

    # ===== 0. Check trajectory coordinate/position consistency =====
    # The 'coordinate' field (WGS84) is used for actual pose generation.
    # The 'position' field (ECEF) is only used for rotation reference.
    # Some trajectory generators have bugs producing mismatched ECEF values.
    # We warn but continue since coordinate is correct.
    frame0 = json_data['cameraFrames'][0]
    coord = frame0.get('coordinate', {})
    pos = frame0.get('position', {})
    if coord and pos:
        coord_utm_x, coord_utm_y = transformer.transform(coord['longitude'], coord['latitude'])
        coord_alt = coord.get('altitude', 0)
        pos_utm = ecef_to_utm.transform(pos['x'], pos['y'], pos['z'])
        
        dist_2d = np.sqrt((coord_utm_x - pos_utm[0])**2 + (coord_utm_y - pos_utm[1])**2)
        alt_diff = abs(coord_alt - pos_utm[2])
        
        if dist_2d > 500 or alt_diff > 500:
            print(f"  WARNING: Trajectory has inconsistent coordinate/position fields!")
            print(f"    coordinate (WGS84->UTM): ({coord_utm_x:.0f}, {coord_utm_y:.0f}, {coord_alt:.0f}m)")
            print(f"    position (ECEF->UTM):    ({pos_utm[0]:.0f}, {pos_utm[1]:.0f}, {pos_utm[2]:.0f}m)")
            print(f"    2D mismatch: {dist_2d:.0f}m, altitude mismatch: {alt_diff:.0f}m")
            print(f"    Continuing with coordinate field (correct position)...")

    # ===== Early intrinsics computation (needed for depth-based bbox) =====
    fov_vertical = json_data['cameraFrames'][0]['fovVertical']
    fov_rad = np.deg2rad(fov_vertical)
    fy = (height / 2) / np.tan(fov_rad / 2)
    fx = fy  # Assume square pixels
    cx = width / 2
    cy = height / 2
    print(f"Intrinsics: fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")

    # ===== 1. Geodata fetching (Manual or Auto) =====
    geodata_refs = None
    lod1_obj_paths = []
    lod2_obj_paths = []
    tiles = {}  # Populated in Phase 1B (after rendering)
    gm = None
    
    # Priority 1: Automated Fetching — Phase 1A: Mesh download only
    # (DOP/DSM/ALS/LoD are downloaded in Phase 1B, after rendering produces
    #  depth maps for accurate visible-region bbox.)
    if auto_geodata:
        if not map_dir:
            map_dir = Path("data/MovingDrone/map")
        print("\n1A. Downloading mesh tiles for rendering...")
        from utils.geodata_berlin import GeodataManager
        gm = GeodataManager(map_dir=map_dir)
        
        # Get trajectory UTM and EPSG:4326 positions
        utm_positions = []
        gps_positions = []
        for frame in json_data['cameraFrames']:
            coord = frame['coordinate']
            x, y = transformer.transform(coord['longitude'], coord['latitude'])
            utm_positions.append([x, y])
            gps_positions.append([coord['longitude'], coord['latitude']])
        utm_positions = np.array(utm_positions)
        gps_positions = np.array(gps_positions)
        
        # Berlin Check
        if not gm.is_in_berlin(gps_positions):
            raise ValueError(f"Error: Automated geodata is currently only supported for Berlin. Trajectory {name} is outside Berlin bounds.")
        
        # Phase 1A: Discover and download ONLY mesh tiles (needed for rendering)
        needed_mesh = gm.discover_tiles(utm_positions, data_types=["mesh"])
        pad = 1000  # 1km padding
        f_bbox_cam = (utm_positions[:, 0].min()-pad, utm_positions[:, 1].min()-pad,
                      utm_positions[:, 0].max()+pad, utm_positions[:, 1].max()+pad)
        fetched_mesh = gm.download_all(needed_mesh, max_workers=8, bounds=f_bbox_cam, per_tile_timeout=600)
        
        if not mesh_dir and fetched_mesh.get('mesh'):
            mesh_dir = fetched_mesh['mesh']
        
        geodata_dir = map_dir
        print(f"   Phase 1A: {len(fetched_mesh.get('mesh', []))} mesh tiles downloaded")


    # Priority 2: Manual local geodata
    elif geodata_dir:
        geodata_path = Path(geodata_dir)
        if geodata_path.exists():
            print("\n1. Detecting required geodata tiles from local directory...")
            utm_positions = []
            for frame in json_data['cameraFrames']:
                coord = frame['coordinate']
                x, y = transformer.transform(coord['longitude'], coord['latitude'])
                utm_positions.append([x, y])
            utm_positions = np.array(utm_positions)
            bbox = [utm_positions[:, 0].min() - 200, utm_positions[:, 1].min() - 200, 
                    utm_positions[:, 0].max() + 200, utm_positions[:, 1].max() + 200]

            from utils.geodata_berlin import GeodataManager
            gm = GeodataManager(map_dir=geodata_path)
            tiles = find_intersecting_tiles(bbox, geodata_path, gm=gm,
                                          dop_dir=dop_dir, dsm_dir=dsm_dir,
                                          lod1_dir=lod1_dir, lod2_dir=lod2_dir,
                                          als_dir=als_dir)
            print(f"   Found: {len(tiles['dop'])} DOP, {len(tiles['dsm'])} DSM, {len(tiles['lod1'])} LoD1, {len(tiles['lod2'])} LoD2, {len(tiles.get('als', []))} ALS tiles")

            # Validate discovered tiles
            _check_modality('DOP tiles', bool(tiles['dop']),
                            f'no DOP tiles intersect trajectory bbox in {geodata_dir}')
            _check_modality('DSM tiles', bool(tiles['dsm']),
                            f'no DSM tiles intersect trajectory bbox in {geodata_dir}')
            if not tiles.get('als', []):
                print(f"   ⚠ WARNING: No ALS tiles intersect trajectory bbox in {geodata_dir} — LiDAR will be missing")
            _check_modality('LoD2 tiles', bool(tiles['lod2']),
                            f'no LoD2 tiles intersect trajectory bbox in {geodata_dir}')
            
            # Convert LoD if requested
            if convert_lod:
                citygml2obj_script = _citygml2obj_script()
                if citygml2obj_script.exists():
                    # LoD1
                    if tiles['lod1']:
                        print("\n   Converting LoD1 tiles to OBJ...")
                        lod1_obj_dir = geodata_path / 'LoDv1_obj'
                        converted = convert_lod_tiles_parallel(
                            [geodata_path / t for t in tiles['lod1']], lod1_obj_dir, citygml2obj_script, lod_level=1)
                        lod1_obj_paths.extend([f"LoDv1_obj/{stem}" for stem in converted])
                    # LoD2
                    if tiles['lod2']:
                        print("\n   Converting LoD2 tiles to OBJ...")
                        lod2_obj_dir = geodata_path / 'LoDv2_obj'
                        converted = convert_lod_tiles_parallel(
                            [geodata_path / t for t in tiles['lod2']], lod2_obj_dir, citygml2obj_script, lod_level=2)
                        lod2_obj_paths.extend([f"LoDv2_obj/{stem}" for stem in converted])
                elif tiles['lod1'] or tiles['lod2']:
                    raise FileNotFoundError(
                        f"CityGML2OBJ script not found at {citygml2obj_script}. "
                        f"Cannot convert {len(tiles['lod1'])} LoD1 + {len(tiles['lod2'])} LoD2 GML tiles to OBJ. "
                        f"Run: git submodule update --init --recursive")

            geodata_refs = {
                'geodata_base_path': str(geodata_path.absolute()),
                'trajectory_bbox': {'min_x': bbox[0], 'min_y': bbox[1], 'max_x': bbox[2], 'max_y': bbox[3]},
            }
        else:
            print(f"\n1. Geodata directory not found: {geodata_dir}")
    else:
        # Legacy behavior: check for local TIFs in input
        print("\n1. Checking for GeoTIFF files in input...")
        for tif_file in input_path.parent.glob("*.tif"):
            dst_name = "dsm.tif" if ('DOM' in tif_file.name or 'DSM' in tif_file.name) else "dop.tif"
            dst = output_path / dst_name
            if dst.exists(): dst.unlink()
            dst.symlink_to(tif_file.absolute())
            print(f"   Linked: {tif_file.name} -> {dst.name}")



    
    # ===== Validation: Ensure proper configuration for rendering =====
    if render_mesh:
        if not auto_geodata and not mesh_dir:
            raise ValueError(
                "Error: --render requires either --auto-geodata OR --mesh-dir.\n"
                "  Use --auto-geodata to download mesh tiles, OR\n"
                "  Use --mesh-dir /path/to/meshes, OR\n"
                "  Omit --render and pass --frames-dir / existing GES frames / --video / --priors-dir."
            )
    
    # ===== 0. Rendering from Mesh (Optional) =====
    if render_mesh:
        if not mesh_dir:
            print("Warning: --render requested but no --mesh-dir provided. Skipping rendering.")
        elif not OPEN3D_AVAILABLE:
            print("Warning: --render requested but Open3D not installed. Skipping rendering.")
        else:
            print("\n0. Rendering frames from mesh...")
            origin_shift_val = None
            
            # Setup Super-Resolution
            sr_enhancer = None
            if sr_mode != 'none':
                try:
                    from utils.sr import TextureEnhancer
                except ImportError as e:
                    raise ImportError(
                        "Super-resolution (--sr-mode) requires utils.sr "
                        "(Real-ESRGAN). Use --sr-mode none to skip, or install "
                        "torch/torchvision and ensure utils/sr.py is present."
                    ) from e
                sr_enhancer = TextureEnhancer()
            
            do_sr_textures = sr_mode in ['textures', 'both']
            do_sr_frames = sr_mode in ['frames', 'both']

            # Filter large VirtualCity mesh tile trees to trajectory footprint
            mesh_load_arg = mesh_dir
            mesh_root = Path(mesh_dir) if not isinstance(mesh_dir, (list, tuple)) else None
            if mesh_root is not None and mesh_root.is_dir():
                # Build UTM positions from trajectory for tile selection
                _utm = []
                for frame in json_data['cameraFrames']:
                    coord = frame['coordinate']
                    x, y = transformer.transform(coord['longitude'], coord['latitude'])
                    _utm.append([x, y])
                tile_dirs = select_mesh_dirs_for_trajectory(mesh_root, _utm, margin_m=800.0)
                n_sub = sum(1 for p in mesh_root.iterdir() if p.is_dir())
                if tile_dirs and n_sub > 8:
                    logger.info(
                        f"Selected {len(tile_dirs)}/{n_sub} mesh tiles near trajectory "
                        f"(margin=800m) from {mesh_root}"
                    )
                    mesh_load_arg = tile_dirs
                elif not tile_dirs and n_sub > 0:
                    logger.warning(
                        f"No mesh tiles intersect trajectory in {mesh_root}; "
                        f"attempting to load entire tree (may be very slow)"
                    )

            models = load_meshes(mesh_load_arg, sr_enhancer=sr_enhancer, 
                                 sr_batch_size=sr_batch_size, sr_textures=do_sr_textures)
            if models:
                # Write a temp JSON with only the frames we want rendered.
                # This ensures render_sequence's PoseLoader reads exactly
                # the (downsampled + truncated) frames, keeping rendering,
                # poses.csv, depth, and video all in sync.
                render_json_path = output_path / '_temp_render_trajectory.json'
                with open(render_json_path, 'w') as f_tmp:
                    json.dump(json_data, f_tmp)

                origin_shift_val = render_sequence(models, str(render_json_path), str(output_path), 
                                utm_zone=utm_zone,
                                high_quality=high_quality,
                                realism=realism, lighting=lighting,
                                temperature=temperature, sun_intensity=sun_intensity,
                                ibl_intensity=ibl_intensity, sr_enhancer=sr_enhancer,
                                sr_frames=do_sr_frames, max_frames=None,
                                background_color=background_color, save_depth=save_depth,
                                save_normals=save_normals,
                                random_sun=random_sun, sun_seed=sun_seed,
                                low_light=low_light, high_bright=high_bright,
                                exposure=exposure)

                # Clean up temp JSON
                render_json_path.unlink(missing_ok=True)

                # ===== Motion Blur (post-rendering) =====
                if motion_blur:
                    rendering_dir = output_path / "rendering"
                    if rendering_dir.exists():
                        print("\n   Applying motion blur to rendered frames...")
                        n_blurred = apply_motion_blur(
                            rendering_dir, json_data['cameraFrames'],
                            blur_strength=blur_strength,
                            speed_threshold_ms=15.0,
                            max_kernel_size=9,
                            frame_rate=frame_rate,
                        )
                        print(f"   Motion blur applied to {n_blurred}/{len(list(rendering_dir.glob('frame_*.jpg')))} frames")
            else:
                print(f"   No meshes found in {mesh_dir}. Skipping rendering.")

    # ===== 1B. Geodata download Phase 2 (depth-based bbox) =====
    # After rendering, depth maps are available. Back-project them to find the
    # actual visible ground region, which can be MUCH larger than the camera
    # position envelope (e.g. tilted cameras at 500m+ altitude).
    if auto_geodata:
        depth_dir = output_path / "depth"
        
        # Write a temporary poses.csv so compute_sequence_bbox can read it
        _temp_poses_path = output_path / "_temp_poses_for_bbox.csv"
        _write_poses_csv(json_data, transformer, _temp_poses_path)
        
        if depth_dir.exists() and list(depth_dir.glob('depth_*.npz')):
            print("\n1B. Computing depth-based visible region bbox...")
            # Use geodata_margin + 50 for tile discovery (wider than final crop)
            depth_bbox = compute_sequence_bbox(
                _temp_poses_path, margin_m=geodata_margin + 50,
                depth_dir=depth_dir,
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=width, height=height,
                subsample_frames=4)
        else:
            print("\n1B. No depth maps available — using camera positions for geodata bbox")
            depth_bbox = None
        
        _temp_poses_path.unlink(missing_ok=True)
        
        # Discover tiles using depth-based bbox (or fallback to camera positions)
        print("   Discovering DOP/DSM/ALS/LoD tiles...")
        needed_geo = gm.discover_tiles(
            utm_positions,
            data_types=["dop", "dsm", "lod1", "lod2", "als"],
            bbox=depth_bbox)
        
        # Use depth bbox for bounds filtering if available, else camera positions + padding
        if depth_bbox is not None:
            pad = 100  # small extra margin on top of depth bbox
            f_bbox = (depth_bbox[0] - pad, depth_bbox[1] - pad,
                      depth_bbox[2] + pad, depth_bbox[3] + pad)
        else:
            pad = 1000
            f_bbox = (utm_positions[:, 0].min()-pad, utm_positions[:, 1].min()-pad,
                      utm_positions[:, 0].max()+pad, utm_positions[:, 1].max()+pad)
        
        # Skip GDI DOP region ZIP downloads (dop_YYYY) — they are 300MB-1GB each
        # and pre-2025 ZIPs contain ECW files unreadable by GDAL.
        # preprocess_dop_multiyear() fetches GDI DOPs via fast WMS crops instead.
        download_geo = {k: v for k, v in needed_geo.items()
                        if not (k.startswith('dop_') and k != 'dop_bb_fallback')}
        # Keep GDI dop_YYYY keys for metadata (empty tile list → WMS fallback)
        gdi_dop_keys = {k: [] for k, v in needed_geo.items()
                        if k.startswith('dop_') and k != 'dop_bb_fallback'}
        
        fetched_files = gm.download_all(download_geo, max_workers=8, bounds=f_bbox, per_tile_timeout=600)
        
        # Build tile references
        rel_fetched = {t: [os.path.relpath(f, geodata_dir) for f in files] for t, files in fetched_files.items()}
        tiles = {
            'dsm': rel_fetched.get('dsm', []), 
            'lod1': rel_fetched.get('lod1', []), 
            'lod2': rel_fetched.get('lod2', []),
            'als': rel_fetched.get('als', [])
        }
        for key, paths in rel_fetched.items():
            if key.startswith('dop'):
                tiles[key] = paths
        # Add GDI DOP year keys with empty tile lists (triggers WMS in preprocess_dop_multiyear)
        for k in gdi_dop_keys:
            if k not in tiles:
                tiles[k] = []
        # Ensure 'dop' key exists (some code expects it)
        if 'dop' not in tiles:
            tiles['dop'] = []
        
        # Convert LoD (parallel)
        citygml2obj_script = _citygml2obj_script()
        if citygml2obj_script.exists():
            if tiles['lod1']:
                print("\n   Converting LoD1 tiles to OBJ...")
                lod1_obj_dir = map_dir / 'LoDv1_obj'
                converted = convert_lod_tiles_parallel(
                    [map_dir / r for r in tiles['lod1']], lod1_obj_dir, citygml2obj_script, lod_level=1)
                lod1_obj_paths.extend([f"LoDv1_obj/{stem}" for stem in converted])
            if tiles['lod2']:
                print("\n   Converting LoD2 tiles to OBJ...")
                lod2_obj_dir = map_dir / 'LoDv2_obj'
                converted = convert_lod_tiles_parallel(
                    [map_dir / r for r in tiles['lod2']], lod2_obj_dir, citygml2obj_script, lod_level=2)
                lod2_obj_paths.extend([f"LoDv2_obj/{stem}" for stem in converted])
        elif tiles['lod1'] or tiles['lod2']:
            raise FileNotFoundError(
                f"CityGML2OBJ script not found at {citygml2obj_script}. "
                f"Cannot convert {len(tiles['lod1'])} LoD1 + {len(tiles['lod2'])} LoD2 GML tiles to OBJ. "
                f"Run: git submodule update --init --recursive")
        
        # Summary
        bb_dop_count = len(tiles.get('dop', [])) + len(tiles.get('dop_bb_fallback', []))
        gdi_dop_year_count = len([k for k in tiles.keys() if k.startswith('dop_') and k != 'dop_bb_fallback'])
        dop_year_keys = sorted([k for k in tiles.keys() if k.startswith('dop_') and k != 'dop_bb_fallback'])
        print(f"   Phase 1B: {bb_dop_count} BB DOP tiles, {gdi_dop_year_count} GDI DOP years (via WMS), "
              f"{len(tiles['dsm'])} DSM, {len(tiles['lod1'])} LoD1, {len(tiles['lod2'])} LoD2, "
              f"{len(tiles['als'])} ALS")

        # Validate — DOP check includes GDI years (fetched via WMS, not downloaded)
        has_any_dop = bb_dop_count > 0 or gdi_dop_year_count > 0
        _check_modality('DOP sources', has_any_dop,
                        'auto-geodata found no DOP tiles or WMS years for trajectory')
        _check_modality('DSM tiles', bool(tiles['dsm']),
                        'auto-geodata found 0 DSM tiles for trajectory')
        if not tiles['als']:
            print("   ⚠ WARNING: auto-geodata found 0 ALS tiles for trajectory — LiDAR will be missing")
        _check_modality('LoD2 tiles', bool(tiles['lod2']),
                        'auto-geodata found 0 LoD2 tiles for trajectory')

        bbox = [utm_positions[:, 0].min(), utm_positions[:, 1].min(), utm_positions[:, 0].max(), utm_positions[:, 1].max()]
        geodata_refs = {
            'geodata_base_path': str(Path(geodata_dir).absolute()),
            'trajectory_bbox': {'min_x': bbox[0], 'min_y': bbox[1], 'max_x': bbox[2], 'max_y': bbox[3]},
        }

    # ===== 1. Create MP4 video (render / frames-dir / GES frames / --video) =====
    logger.info("1. Creating MP4 video...")
    video_path = output_path / "video.mp4"
    footage_dir = output_path / "rendering"
    video_created = False

    # Optional: copy an existing video
    if video_src:
        src_vid = Path(video_src)
        if not src_vid.exists():
            raise FileNotFoundError(f"--video not found: {src_vid}")
        shutil.copy2(src_vid, video_path)
        used_existing_video = True
        video_created = True
        logger.info(f"Copied existing video → {video_path}")

    # Resolve offline frames if we did not mesh-render into output/rendering
    resolved_frames = resolve_frames_dir(
        frames_dir=frames_dir,
        input_path=input_path,
        output_path=output_path,
    )

    if not video_created and resolved_frames is not None:
        # Stage into output/rendering as frame_%04d.jpg for encoding / trim logic
        if resolved_frames.resolve() != footage_dir.resolve() or not list(footage_dir.glob('frame_*.jpg')):
            logger.info(f"Staging frames from {resolved_frames} → {footage_dir}")
            stage_frames_for_encoding(resolved_frames, footage_dir, max_frames=max_frames)

    if not video_created and footage_dir.exists():
        frame_files = sorted(footage_dir.glob("frame_*.jpg"))
        if not frame_files:
            frame_files = (
                sorted(footage_dir.glob("*.jpeg"))
                + sorted(footage_dir.glob("*.jpg"))
                + sorted(footage_dir.glob("*.png"))
            )
        if max_frames and len(frame_files) > max_frames:
            frame_files = frame_files[:max_frames]

        if frame_files:
            logger.info(f"Found {len(frame_files)} frames in {footage_dir}")

            # --- Invalid Frame Check and Trimming ---
            first_idx = 0
            last_idx = len(frame_files) - 1
            for i in range(len(frame_files)):
                frame = cv2.imread(str(frame_files[i]))
                if not is_frame_invalid(frame):
                    first_idx = i
                    break
            else:
                raise ValueError(
                    f"All {len(frame_files)} frames in {footage_dir} are invalid "
                    f"(black or sky-colored). Cannot create a sequence."
                )

            for i in range(len(frame_files) - 1, first_idx - 1, -1):
                frame = cv2.imread(str(frame_files[i]))
                if not is_frame_invalid(frame):
                    last_idx = i
                    break

            trimmed = first_idx > 0 or last_idx < len(frame_files) - 1
            original_first_idx = first_idx
            original_last_idx = last_idx
            if trimmed:
                logger.warning(
                    f"Trimming {first_idx} frames from start and "
                    f"{len(frame_files) - 1 - last_idx} from end"
                )
                frame_files = frame_files[first_idx:last_idx + 1]
                json_data['cameraFrames'] = json_data['cameraFrames'][first_idx:last_idx + 1]
                num_frames = len(frame_files)

            # Validate middle frames (mesh-render failures)
            if render_mesh:
                logger.info(f"Validating all {len(frame_files)} frames...")
                invalid_frames = []
                for i, fpath in enumerate(frame_files):
                    frame = cv2.imread(str(fpath))
                    if is_frame_invalid(frame):
                        invalid_frames.append(i)
                if invalid_frames:
                    shown = invalid_frames[:10]
                    more = f" (and {len(invalid_frames) - 10} more)" if len(invalid_frames) > 10 else ""
                    raise ValueError(
                        f"{len(invalid_frames)} invalid frames in the middle of sequence. "
                        f"Indices: {shown}{more}."
                    )

            if trimmed:
                # Keep depth/normals/frames indices in sync after trim
                if save_depth:
                    depth_dir = output_path / "depth"
                    if depth_dir.exists():
                        all_depth_files = sorted(list(depth_dir.glob("depth_*.npz")))
                        depth_dict = {}
                        for d_file in all_depth_files:
                            try:
                                f_idx = int(d_file.stem.split('_')[1])
                                depth_dict[f_idx] = d_file
                            except Exception:
                                pass
                        for f_idx, d_file in depth_dict.items():
                            if f_idx < original_first_idx or f_idx > original_last_idx:
                                d_file.unlink()
                        temp_mapping = {}
                        for i in range(num_frames):
                            old_idx = i + original_first_idx
                            if old_idx in depth_dict and depth_dict[old_idx].exists():
                                temp_file = depth_dir / f"depth_temp_{i:04d}.npz"
                                depth_dict[old_idx].rename(temp_file)
                                temp_mapping[i] = temp_file
                        for i, temp_file in temp_mapping.items():
                            temp_file.rename(depth_dir / f"depth_{i:04d}.npz")

                if save_normals:
                    normals_dir = output_path / "normals"
                    if normals_dir.exists():
                        all_normal_files = sorted(list(normals_dir.glob("normal_*.npz")))
                        normal_dict = {}
                        for n_file in all_normal_files:
                            try:
                                f_idx = int(n_file.stem.split('_')[1])
                                normal_dict[f_idx] = n_file
                            except Exception:
                                pass
                        for f_idx, n_file in normal_dict.items():
                            if f_idx < original_first_idx or f_idx > original_last_idx:
                                n_file.unlink()
                        temp_mapping = {}
                        for i in range(num_frames):
                            old_idx = i + original_first_idx
                            if old_idx in normal_dict and normal_dict[old_idx].exists():
                                temp_file = normals_dir / f"normal_temp_{i:04d}.npz"
                                normal_dict[old_idx].rename(temp_file)
                                temp_mapping[i] = temp_file
                        for i, temp_file in temp_mapping.items():
                            temp_file.rename(normals_dir / f"normal_{i:04d}.npz")

                # Re-stage trimmed frames as contiguous frame_%04d.jpg
                tmp_stage = output_path / "_frames_trim_stage"
                if tmp_stage.exists():
                    shutil.rmtree(tmp_stage)
                tmp_stage.mkdir(parents=True)
                for i, src in enumerate(frame_files):
                    img = cv2.imread(str(src))
                    if img is None:
                        raise ValueError(f"Failed to read trimmed frame: {src}")
                    cv2.imwrite(str(tmp_stage / f"frame_{i:04d}.jpg"), img,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if footage_dir.exists():
                    shutil.rmtree(footage_dir)
                tmp_stage.rename(footage_dir)
                frame_files = sorted(footage_dir.glob("frame_*.jpg"))
                logger.info(f"Trimmed sequence length: {num_frames}")

            # Align cameraFrames length with available frames if offline frames shorter
            if len(json_data['cameraFrames']) > len(frame_files):
                json_data['cameraFrames'] = json_data['cameraFrames'][:len(frame_files)]
                num_frames = len(frame_files)
                logger.warning(f"Truncated trajectory to {num_frames} frames to match footage")
            elif len(json_data['cameraFrames']) < len(frame_files):
                frame_files = frame_files[:len(json_data['cameraFrames'])]
                num_frames = len(frame_files)
                logger.warning(f"Truncated footage to {num_frames} frames to match trajectory")

            video_created = encode_video_from_frames(
                footage_dir, video_path, frame_rate, width, height,
                frame_files=frame_files, keep_rendering=keep_rendering,
            )

    if not video_created and not video_path.exists():
        logger.warning(
            "No video.mp4 created. Provide --frames-dir, GES frames next to the JSON, "
            "--video, or --render --mesh-dir."
        )

    # ===== 2. Create poses.csv =====
    logger.info("2. Creating poses.csv...")
    poses_path = output_path / "poses.csv"
    _write_poses_csv(json_data, transformer, poses_path)
    logger.info(f"Saved: {poses_path}")
    
    # ===== 3. Intrinsics (always written — required by MovingDrone / tracking) =====
    logger.info(f"3. Intrinsics: fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")
    save_intrinsics_json(output_path, width, height, fx, fy, cx, cy)
    
    # ===== 4.5 Compute Z-offset and apply to poses =====
    # The trajectory altitude (from GES) may differ from the DSM/mesh reference.
    # We compute the offset empirically and apply it to poses.csv to ensure rendered
    # point maps match DSM elevations.
    if geodata_refs and (output_path / "depth").exists():
        print("\n4.5. Computing Z-offset (rendered depth vs DSM)...")
        dsm_abs_paths_zoff = [Path(geodata_dir) / t for t in tiles['dsm']]
        
        # Get origin shift if rendering was done
        origin_shift_for_zoff = None
        if 'origin_shift_val' in locals() and origin_shift_val is not None:
            origin_shift_for_zoff = origin_shift_val
        
        # Load poses for z-offset computation
        import pandas as pd
        df_poses = pd.read_csv(poses_path)
        
        z_offset_val = compute_z_offset_from_rendered_depth(
            depth_dir=output_path / "depth",
            poses_df=df_poses,
            dsm_paths=dsm_abs_paths_zoff,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            origin_shift=origin_shift_for_zoff,
            num_samples=5
        )
        
        # Apply offset to poses.csv if significant
        if z_offset_val is not None and abs(z_offset_val) > 0.1:
            print(f"   Applying Z-offset correction: subtracting {z_offset_val:.2f}m from all pose Z values")
            df_poses['z'] = df_poses['z'] - z_offset_val
            df_poses['altitude'] = df_poses['altitude'] - z_offset_val
            df_poses.to_csv(poses_path, index=False)
            print(f"   Updated poses.csv with corrected Z values")
        elif z_offset_val is not None:
            print(f"   Z-offset negligible ({z_offset_val:.2f}m), no correction needed")
    
    # ===== 5. Creating meta.json with metadata =====
    print("\n5. Creating meta.json...")
    meta = {
        'name': name,
        'width': width,
        'height': height,
        'num_frames': num_frames,
        'frame_rate': frame_rate,
        'fov_vertical': fov_vertical,
        'utm_zone': utm_zone,
    }

    # Record realism augmentation settings
    if trajectory_noise or motion_blur or random_sun or sudden_jitter or low_light or high_bright or wind_gusts:
        augmentation_meta = {}
        if trajectory_noise:
            augmentation_meta['trajectory_noise'] = {
                'position_noise_m': position_noise_m,
                'rotation_noise_deg': rotation_noise_deg,
                'roll_noise_deg': roll_noise_deg,
                'bank_on_turns': bank_on_turns,
                'bank_factor': bank_factor,
                'smoothness': noise_smoothness,
                'seed': noise_seed,
            }
        if motion_blur:
            augmentation_meta['motion_blur'] = {
                'blur_strength': blur_strength,
            }
        if random_sun:
            augmentation_meta['random_sun'] = {
                'seed': sun_seed,
            }
        if sudden_jitter and jitter_applied_frame is not None:
            augmentation_meta['sudden_jitter'] = {
                'frame': jitter_applied_frame,
                'magnitude_deg': jitter_magnitude_deg,
                'seed': jitter_seed,
            }
        if wind_gusts and gust_episodes is not None:
            augmentation_meta['wind_gusts'] = {
                'episodes': gust_episodes,
                'magnitude_deg': gust_magnitude_deg,
                'seed': gust_seed,
            }
        if low_light:
            augmentation_meta['low_light'] = True
        if high_bright:
            augmentation_meta['high_bright'] = True
        meta['realism_augmentation'] = augmentation_meta

    if 'origin_shift_val' in locals() and origin_shift_val is not None:
        meta['render_origin'] = origin_shift_val.tolist()

    # Add geodata tile references and sources
    if geodata_refs:
        meta['tiles'] = {
            'dsm': tiles.get('dsm', []),
            'lod1': tiles.get('lod1', []),
            'lod2': tiles.get('lod2', []),
            'als': tiles.get('als', []),
            'mesh': tiles.get('mesh', [])
        }
        # Add all DOP keys (e.g., 'dop', 'dop_2024', 'dop_bb_fallback')
        for key in tiles.keys():
            if key.startswith('dop'):
                meta['tiles'][key] = tiles[key]

        if lod1_obj_paths:
            meta['tiles']['lod1_obj'] = lod1_obj_paths
        if lod2_obj_paths:
            meta['tiles']['lod2_obj'] = lod2_obj_paths
            
        # Extract geodata metadata (licenses, capture dates, urls)
        geodata_sources = {}
        if 'gm' in locals() and gm is not None:
            for d_type, d_tiles in meta['tiles'].items():
                if not d_tiles or d_type.endswith('_obj'):
                    continue
                # Use sets to deduplicate metadata across tiles of the same type
                sources = {}
                for t_path in d_tiles:
                    # Extracts '33391-5819' from 'dop/33391-5819/dop_33391-5819.tif'
                    # or 'LoD1_392_5820' from 'lod1/berlin_full/LoD1_392_5820.xml'
                    try:
                        pts = Path(t_path).parts
                        if len(pts) > 1 and pts[0] == d_type:
                            if pts[1] == "berlin_full":
                                tid = Path(t_path).stem
                            else:
                                tid = pts[1]
                        elif len(pts) > 1 and "mesh" in d_type:
                            tid = pts[1]
                        else:
                            tid = Path(t_path).stem
                    except:
                        tid = str(t_path)
                        
                    t_meta = gm.get_tile_metadata(d_type, tid)
                    key = (t_meta["source_url"], t_meta["license"], t_meta["capture_date"])
                    sources[key] = t_meta
                geodata_sources[d_type] = list(sources.values())
        meta['geodata_sources'] = geodata_sources

        # Add trajectory bbox
        meta['trajectory_bbox'] = {
            'min_x': bbox[0],
            'min_y': bbox[1],
            'max_x': bbox[2],
            'max_y': bbox[3]
        }

    meta_path = output_path / "meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, cls=NumpyJSONEncoder)
    print(f"   Saved: {meta_path}")

    # --- Validate core outputs ---
    if render_mesh:
        _check_modality('Video', (output_path / 'video.mp4').exists(),
                        f'video.mp4 not created in {output_path}')
        _check_modality('Depth maps', (output_path / 'depth').exists() and
                        len(list((output_path / 'depth').glob('depth_*.npz'))) > 0,
                        f'no depth maps created in {output_path}/depth/')

    _check_modality('Poses', (output_path / 'poses.csv').exists(),
                    f'poses.csv not created in {output_path}')

    print(f"\n✓ Conversion complete! Output: {output_path}")

    # --- Statistics & Geocoding ---
    print("\n--- Generating Statistics & Region Name ---")

    # 1. Trajectory Stats
    import pandas as pd
    df_poses = pd.read_csv(poses_path)

    positions = df_poses[['x', 'y', 'z']].values
    # Length
    diffs_xy = np.diff(positions[:, :2], axis=0)
    dists_xy = np.linalg.norm(diffs_xy, axis=1)
    trajectory_length = float(np.sum(dists_xy))

    # Speed (3D)
    diffs_3d = np.diff(positions, axis=0)
    dists_3d = np.linalg.norm(diffs_3d, axis=1)
    # Speed = distance / time_per_frame = distance * fps
    speeds = dists_3d * frame_rate

    speed_stats = {
        'min': float(speeds.min()) if len(speeds) > 0 else 0.0,
        'max': float(speeds.max()) if len(speeds) > 0 else 0.0,
        'mean': float(speeds.mean()) if len(speeds) > 0 else 0.0
    }

    # Area (BBox)
    min_x, max_x = positions[:, 0].min(), positions[:, 0].max()
    min_y, max_y = positions[:, 1].min(), positions[:, 1].max()
    area_bbox = (max_x - min_x) * (max_y - min_y)

    # Altitude
    alt_min, alt_max = positions[:, 2].min(), positions[:, 2].max()

    # 2. Pose Distribution
    pose_stats = {
        'roll': {'mean': float(df_poses['roll'].mean()), 'std': float(df_poses['roll'].std()), 'min': float(df_poses['roll'].min()), 'max': float(df_poses['roll'].max())},
        'pitch': {'mean': float(df_poses['pitch'].mean()), 'std': float(df_poses['pitch'].std()), 'min': float(df_poses['pitch'].min()), 'max': float(df_poses['pitch'].max())},
        'yaw': {'mean': float(df_poses['yaw'].mean()), 'std': float(df_poses['yaw'].std()), 'min': float(df_poses['yaw'].min()), 'max': float(df_poses['yaw'].max())},
        'altitude': {'mean': float(df_poses['altitude'].mean()), 'std': float(df_poses['altitude'].std())}
    }

    # 3. Reverse Geocoding - Try to get a better name for the region
    region_name = name  # Default to original name
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="orthotrack_converter", timeout=10)
        
        # Use center frame
        center_frame = df_poses.iloc[len(df_poses)//2]
        lat, lon = center_frame['latitude'], center_frame['longitude']

        print(f"   Attempting reverse geocoding at ({lat:.6f}, {lon:.6f})...")
        location = geolocator.reverse((lat, lon), language='en')
        
        if location and location.raw:
            addr = location.raw.get('address', {})
            # Try to extract the most specific useful name
            # Priority: building/attraction > neighborhood > suburb > city/town > county
            candidates = [
                addr.get('building'),
                addr.get('attraction'),
                addr.get('tourism'),
                addr.get('neighbourhood'),
                addr.get('suburb'),
                addr.get('city'),
                addr.get('town'),
                addr.get('village'),
                addr.get('county'),
            ]
            
            # Get first non-None candidate
            for candidate in candidates:
                if candidate:
                    region_name = candidate
                    print(f"   ✓ Detected Region: {region_name}")
                    break
            else:
                print(f"   ⚠ Geocoding returned address but no useful name components")
        else:
            print(f"   ⚠ Geocoding returned no results")
            
    except Exception as e:
        print(f"   ⚠ Reverse geocoding failed: {e}")

    # Update meta.json with stats
    meta['name'] = region_name
    meta['statistics'] = {
        'trajectory_length_m': trajectory_length,
        'coverage_area_m2': area_bbox,
        'altitude_range_m': [float(alt_min), float(alt_max)],
        'speed_m_s': speed_stats,
        'pose_distribution': pose_stats
    }

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, cls=NumpyJSONEncoder)
    print(f"   Updated meta.json with stats and name: {region_name}")

    # ===== 6. Geodata: either crop from tiles OR copy pre-extracted priors =====
    if priors_dir:
        logger.info("6. Copying pre-extracted scene priors (--priors-dir)...")
        priors_copied = copy_scene_priors(Path(priors_dir), output_path)

        # Prefer utm_offset from priors meta; else derive from pose centroid so
        # absolute UTM poses become local and match typical MovingDrone priors.
        utm_offset = priors_copied.get('utm_offset')
        if utm_offset is None:
            df_abs = pd.read_csv(poses_path)
            utm_offset = [
                float(df_abs['x'].mean()),
                float(df_abs['y'].mean()),
                float(df_abs['z'].min()),
            ]
            logger.info(
                f"No utm_offset in priors meta; using pose-derived offset "
                f"[{utm_offset[0]:.2f}, {utm_offset[1]:.2f}, {utm_offset[2]:.2f}]"
            )
        else:
            utm_offset = [float(v) for v in utm_offset]

        df_poses = pd.read_csv(poses_path)
        # Only shift if poses still look absolute (far from local origin)
        if abs(df_poses['x'].mean()) > 10_000 or abs(df_poses['y'].mean()) > 10_000:
            df_poses['x'] = df_poses['x'] - utm_offset[0]
            df_poses['y'] = df_poses['y'] - utm_offset[1]
            df_poses['z'] = df_poses['z'] - utm_offset[2]
            df_poses['altitude'] = df_poses['altitude'] - utm_offset[2]
            df_poses.to_csv(poses_path, index=False)
            logger.info(f"Poses shifted into local frame using utm_offset={utm_offset}")
        else:
            logger.info("Poses already look local; leaving coordinates unchanged")

        meta['utm_offset'] = utm_offset
        if priors_copied.get('dsm_meta'):
            meta['dsm'] = priors_copied['dsm_meta']
        else:
            try:
                dsm_npz = np.load(output_path / 'dsm.npz')
                meta['dsm'] = {
                    'file': 'dsm.npz',
                    'bounds': dsm_npz['bounds'].tolist() if 'bounds' in dsm_npz else None,
                    'gsd': float(dsm_npz['gsd']) if 'gsd' in dsm_npz else None,
                }
            except Exception as e:
                logger.warning(f"Could not read dsm.npz metadata: {e}")

        if priors_copied.get('dops'):
            meta['dops'] = priors_copied['dops']
        else:
            dop_dir_out = output_path / 'dop'
            if dop_dir_out.exists():
                dops = {}
                for jpg in sorted(dop_dir_out.glob('*.jpg')):
                    year = jpg.stem
                    dops[f'dop_{year}'] = {'year': year, 'file': f'dop/{jpg.name}'}
                if dops:
                    meta['dops'] = dops

        meta['num_frames'] = len(df_poses)
        seq_files = {}
        for f in output_path.iterdir():
            if f.is_file():
                seq_files[f.name] = {
                    'size_bytes': f.stat().st_size,
                    'size_mb': round(f.stat().st_size / 1024 / 1024, 2),
                }
        dop_out = output_path / 'dop'
        if dop_out.exists():
            seq_files['dop/'] = {'num_files': len(list(dop_out.glob('*.jpg')))}
        meta['files'] = seq_files
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, cls=NumpyJSONEncoder)
        logger.info("Priors ingested and meta.json updated")

    elif geodata_refs:
        geodata_base_path = geodata_refs.get('geodata_base_path', '')
        if geodata_base_path:
            preprocess_result = preprocess_geodata(
                output_path, geodata_base_path, tiles,
                lod1_obj_paths, lod2_obj_paths,
                poses_path, width, height, fx, fy, cx, cy,
                margin=geodata_margin, gm=gm if 'gm' in locals() else None)

            utm_offset = preprocess_result.get('utm_offset', [0.0, 0.0, 0.0])
            dsm_meta = preprocess_result.get('dsm')
            dop_years = preprocess_result.get('dop_years', [])

            # Update tiles with re-discovered set from preprocess_geodata
            if 'tiles' in preprocess_result:
                tiles = preprocess_result['tiles']

            # Store utm_offset and geodata metadata in meta.json
            meta['utm_offset'] = utm_offset
            if dsm_meta:
                meta['dsm'] = dsm_meta
            
            # Store multi-year DOP metadata as dops dict keyed by dop_<year>
            if dop_years:
                meta['dops'] = {
                    f"dop_{d['year']}": d for d in dop_years
                }

            # Store paths to all sequence files (for easy inspection)
            seq_files = {}
            for f in output_path.iterdir():
                if f.is_file():
                    seq_files[f.name] = {
                        'size_bytes': f.stat().st_size,
                        'size_mb': round(f.stat().st_size / 1024 / 1024, 2),
                    }
            # Add depth directory info
            depth_dir = output_path / 'depth'
            if depth_dir.exists():
                depth_count = len(list(depth_dir.glob('depth_*.npz')))
                seq_files['depth/'] = {'num_files': depth_count}
            normals_dir = output_path / 'normals'
            if normals_dir.exists():
                normals_count = len(list(normals_dir.glob('normal_*.npz')))
                seq_files['normals/'] = {'num_files': normals_count}
            # Add DOP directory info
            dop_dir = output_path / 'dop'
            if dop_dir.exists():
                dop_count = len(list(dop_dir.glob('*.jpg')))
                seq_files['dop/'] = {'num_files': dop_count}
            meta['files'] = seq_files

            # Update render_origin to be relative to utm_offset
            if 'origin_shift_val' in locals() and origin_shift_val is not None:
                render_origin_local = (origin_shift_val - np.array(utm_offset)).tolist()
                meta['render_origin'] = render_origin_local

            # ===== 7. Convert poses to local coordinates =====
            logger.info("7. Converting poses to local coordinates...")
            df_poses = pd.read_csv(poses_path)
            df_poses['x'] = df_poses['x'] - utm_offset[0]
            df_poses['y'] = df_poses['y'] - utm_offset[1]
            df_poses['z'] = df_poses['z'] - utm_offset[2]
            df_poses['altitude'] = df_poses['altitude'] - utm_offset[2]
            df_poses.to_csv(poses_path, index=False)
            logger.info(f"Poses shifted by utm_offset: [{utm_offset[0]:.2f}, {utm_offset[1]:.2f}, {utm_offset[2]:.2f}]")
            meta['num_frames'] = len(df_poses)

            # Re-save meta.json with all updates
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2, cls=NumpyJSONEncoder)
    else:
        logger.warning(
            "No --geodata-dir / --auto-geodata / --priors-dir: scene has poses/video/intrinsics "
            "but no dop/dsm. OrthoTrack tracking needs map priors."
        )

    # Print summary
    logger.info("--- Trajectory Summary ---")
    logger.info(f"Result Name: {region_name}")
    logger.info(f"Length: {trajectory_length:.1f}m")
    logger.info(f"Area: {area_bbox:.0f}m²")
    logger.info(f"Altitude: {alt_min:.1f}m - {alt_max:.1f}m")
    logger.info(f"Yaw Range: {pose_stats['yaw']['min']:.1f}° - {pose_stats['yaw']['max']:.1f}°")

    # ===== Final Verification =====
    has_geodata = bool(priors_copied) or bool(geodata_refs)
    has_lod = (output_path / 'lod1.npz').exists() and (output_path / 'lod2.npz').exists()
    expect_video = bool(
        render_mesh or used_existing_video or (output_path / 'video.mp4').exists()
    )
    # Tile pipeline historically requires LoD; priors path only if both lod files exist
    if priors_copied is not None:
        expect_lod = has_lod
    else:
        expect_lod = bool(geodata_refs)

    is_valid, issues = verify_sequence(
        output_path,
        expect_video=expect_video,
        expect_depth=render_mesh and save_depth,
        expect_normals=render_mesh and save_normals,
        expect_geodata=has_geodata,
        expect_lod=expect_lod,
        destroy_on_fail=True,
    )
    if not is_valid:
        raise MissingModalityError(
            f"Sequence '{output_path.name}' failed verification and was removed. "
            f"Issues: {'; '.join(issues)}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Create a MovingDrone scene from a trajectory JSON + frames/mesh + geodata/priors. '
            'Google Earth Studio is optional upstream for footage/poses; offline mesh/geodata '
            'authoring is the primary path.'
        ),
    )

    # Required arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to trajectory JSON (GES-compatible cameraFrames)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output scene directory (use data/MovingDrone/scenes/<name>)')

    # Offline footage / priors
    offline_group = parser.add_argument_group('Offline footage & priors')
    offline_group.add_argument('--frames-dir', type=str, default=None,
                               help='Directory of UAV frames (jpg/png). Also auto-detected '
                                    'next to the trajectory as frames/, footage/, or loose JPEGs.')
    offline_group.add_argument('--video', type=str, default=None,
                               help='Existing video.mp4 to copy into the scene (skips frame encoding)')
    offline_group.add_argument('--priors-dir', type=str, default=None,
                               help='Directory with pre-extracted MovingDrone priors '
                                    '(dop/, dsm.npz, optional lod1.npz/lod2.npz). '
                                    'Preferred offline alternative to --geodata-dir tile cropping.')

    # Geodata source arguments
    geodata_group = parser.add_argument_group('Geodata sources')
    geodata_group.add_argument('--geodata-dir', '-g', type=str, default=None,
                               help='Path to geodata directory containing DOP/, DSM/, LoDv2/')
    geodata_group.add_argument('--dop-dir', type=str, default=None, help='Explicit path to DOP directory')
    geodata_group.add_argument('--dsm-dir', type=str, default=None, help='Explicit path to DSM directory')
    geodata_group.add_argument('--lod1-dir', type=str, default=None, help='Explicit path to LoD1 directory')
    geodata_group.add_argument('--lod2-dir', type=str, default=None, help='Explicit path to LoD2 directory')
    geodata_group.add_argument('--als-dir', type=str, default=None, help='Explicit path to ALS directory')
    geodata_group.add_argument('--auto-geodata', action='store_true',
                               help='Automatically fetch geodata from Geobasis-BB and VirtualCityMap (Berlin)')
    geodata_group.add_argument('--map-dir', type=str,
                               help='Centralized directory for geodata storage and reuse')

    # Rendering arguments
    render_group = parser.add_argument_group('Rendering')
    render_group.add_argument('--mesh-dir', type=str, default=None,
                              help='Directory containing OBJ meshes for rendering')
    render_group.add_argument('--high-quality', action='store_true',
                              help='Enable all high quality features')
    render_group.add_argument('--realism', action='store_true',
                              help='Enable ACES tone mapping and SSAO')
    render_group.add_argument('--lighting', action='store_true',
                              help='Enable Sun and Indirect lighting')
    render_group.add_argument('--temperature', type=float, default=6500.0,
                              help='Color temperature in Kelvin')
    render_group.add_argument('--sun-intensity', type=int, default=None)
    render_group.add_argument('--ibl_intensity', type=int, default=None)
    render_group.add_argument('--background_color', type=float, nargs=4, default=DEFAULT_SKY_COLOR,
                              help='Background RGBA color (default: sky blue)')
    render_group.add_argument('--exposure', type=float, default=1.0,
                              help='Exposure value for rendering. Default: 1.0')

    # Super Resolution
    sr_group = parser.add_argument_group('Super Resolution')
    sr_group.add_argument('--sr-mode', type=str, choices=['none', 'textures', 'frames', 'both'], default='none')
    sr_group.add_argument('--sr-batch-size', type=int, default=1)

    # Processing limits
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Max number of frames to process')
    parser.add_argument('--target-fps', type=float, default=None,
                        help='Downsample trajectory to this frame rate (e.g. 5 for 5fps from 30fps)')

    # Feature toggles
    toggle_group = parser.add_argument_group('Feature toggles')
    toggle_group.add_argument('--render', action='store_true', dest='save_render',
                              help='Render frames from --mesh-dir / --auto-geodata meshes (opt-in)')
    toggle_group.add_argument('--no-render', action='store_false', dest='save_render',
                              help='Do not mesh-render (default). Use with --frames-dir / GES frames / --video.')
    toggle_group.add_argument('--no-depth', action='store_false', dest='save_depth',
                              help='Disable saving depth maps (only relevant with --render)')
    toggle_group.add_argument('--keep-rendering', action='store_true', dest='keep_rendering',
                              help='Keep the rendering/ folder of individual JPEG frames after '
                                   'video.mp4 is encoded. By default it is deleted to save space.')
    toggle_group.add_argument('--save-normals', action='store_true', dest='save_normals',
                              help='Save pre-rendered surface normals (normally not needed: '
                                   'normals are synthesised on-the-fly from depth at load time).')
    toggle_group.add_argument('--geodata-margin', type=float, default=50.0,
                              help='Extra margin in meters around the visible-area bbox when '
                                   'cropping geodata (DOP, DSM, etc.). Default: 50')
    toggle_group.add_argument('--no-normals', action='store_false', dest='save_normals',
                              help=argparse.SUPPRESS)
    toggle_group.add_argument('--no-convert-lod', action='store_false', dest='convert_lod',
                              help='Disable automatic conversion of LoD GML tiles to OBJ')
    parser.set_defaults(
        save_render=False,
        save_normals=False,
        keep_rendering=False,
        trajectory_noise=False,
        motion_blur=False,
        random_sun=False,
        sudden_jitter=False,
        wind_gusts=False,
        bank_on_turns=True,
    )
    realism_group = parser.add_argument_group('Realism augmentation',
        'Mesh-render realism (opt-in). Enabled automatically with --render unless --no-realism.')
    realism_group.add_argument('--no-trajectory-noise', action='store_false', dest='trajectory_noise',
                               help='Disable correlated pose noise when realism is on')
    realism_group.add_argument('--position-noise', type=float, default=0.15, metavar='M',
                               help='Position noise std-dev in meters for XY (Z is ~67%%). Default: 0.15')
    realism_group.add_argument('--rotation-noise', type=float, default=0.03, metavar='DEG',
                               help='Pitch/yaw gimbal drift std-dev in degrees. Default: 0.03')
    realism_group.add_argument('--roll-noise', type=float, default=0.05, metavar='DEG',
                               help='Roll gimbal drift std-dev in degrees. Default: 0.05')
    realism_group.add_argument('--no-bank-on-turns', action='store_false', dest='bank_on_turns',
                               help='Disable roll banking during turns (default: enabled when realism on)')
    realism_group.add_argument('--bank-factor', type=float, default=0.15,
                               help='Roll banking multiplier (deg roll per deg/frame yaw rate). Default: 0.15')
    realism_group.add_argument('--noise-smoothness', type=int, default=15,
                               help='Temporal smoothing kernel (higher = slower drift). Default: 15')
    realism_group.add_argument('--noise-seed', type=int, default=None,
                               help='Random seed for trajectory noise (for reproducibility)')
    realism_group.add_argument('--no-motion-blur', action='store_false', dest='motion_blur',
                               help='Disable directional motion blur on high-speed frames')
    realism_group.add_argument('--blur-strength', type=float, default=1.0,
                               help='Motion blur intensity multiplier (1.0 = realistic). Default: 1.0')
    realism_group.add_argument('--no-random-sun', action='store_false', dest='random_sun',
                               help='Disable random sun direction')
    realism_group.add_argument('--sun-seed', type=int, default=None,
                               help='Random seed for sun direction (for reproducibility)')
    realism_group.add_argument('--low-light', action='store_true',
                               help='Force low-light / dusk conditions. If not set, 2%% chance when random-sun is ON')
    realism_group.add_argument('--low-light-probability', type=float, default=0.02,
                               help='Probability of low-light conditions when random-sun is ON. Default: 0.02')
    realism_group.add_argument('--high-bright-probability', type=float, default=0.30,
                               help='Probability of high-bright (overexposed) conditions when random-sun is ON. Default: 0.30')
    realism_group.add_argument('--no-sudden-jitter', action='store_false', dest='sudden_jitter',
                               help='Disable sudden jitter events')
    realism_group.add_argument('--jitter-frame', type=int, default=None,
                               help='Specific frame for jitter. If omitted, placed randomly with --jitter-probability')
    realism_group.add_argument('--jitter-probability', type=float, default=0.2,
                               help='Probability of jitter occurring when --jitter-frame is not set. Default: 0.2')
    realism_group.add_argument('--jitter-magnitude', type=float, default=0.8, metavar='DEG',
                               help='Peak jitter rotation in degrees (Z-axis biased). Default: 0.8')
    realism_group.add_argument('--jitter-seed', type=int, default=None,
                               help='Random seed for jitter placement/direction')
    realism_group.add_argument('--no-wind-gusts', action='store_false', dest='wind_gusts',
                               help='Disable episodic wind gust shake bursts')
    realism_group.add_argument('--gust-probability', type=float, default=0.4,
                               help='Probability that wind gust episodes occur. Default: 0.4 (40%%)')
    realism_group.add_argument('--gust-magnitude', type=float, default=0.35, metavar='DEG',
                               help='Rotation std-dev during gust episodes (degrees). Default: 0.35')
    realism_group.add_argument('--gust-episodes-max', type=int, default=2,
                               help='Max number of gust episodes per sequence. Default: 2')
    realism_group.add_argument('--gust-duration', type=float, default=4.0, metavar='SEC',
                               help='Duration of each gust episode in seconds (excl. fade). Default: 4.0')
    realism_group.add_argument('--gust-seed', type=int, default=None,
                               help='Random seed for gust placement/noise')
    realism_group.add_argument('--no-realism', action='store_true',
                               help='Disable ALL realism augmentations (clean/baseline)')

    # Velocity bounding
    velocity_group = parser.add_argument_group('Velocity bounding',
        'Resample trajectory so max speed stays below a threshold')
    velocity_group.add_argument('--max-speed', type=float, default=100.0, metavar='KMH',
                                help='Maximum allowed speed in km/h. Trajectories exceeding this '
                                     'are resampled via cubic-spline interpolation. Default: 100')
    velocity_group.add_argument('--no-velocity-bound', action='store_false', dest='velocity_bound',
                                help='Disable velocity bounding entirely')
    parser.set_defaults(velocity_bound=True)

    args = parser.parse_args()

    # Mesh-render path: enable research-style realism unless --no-realism
    if args.save_render and not args.no_realism:
        args.trajectory_noise = True
        args.motion_blur = True
        args.random_sun = True
        args.sudden_jitter = True
        args.wind_gusts = True

    # Handle --no-realism master switch (also clears render-enabled defaults)
    if args.no_realism:
        args.trajectory_noise = False
        args.motion_blur = False
        args.random_sun = False
        args.sudden_jitter = False
        args.wind_gusts = False
        args.low_light = False
        args.high_bright = False

    # Auto-trigger low-light and high-bright with probability when random-sun is ON
    args.high_bright = False
    if args.random_sun and not args.low_light:
        rng = np.random.default_rng(args.sun_seed)
        r = rng.random()
        if r < args.low_light_probability:
            args.low_light = True
            logger.info(f"[Realism] Low-light randomly activated ({args.low_light_probability*100:.0f}% chance)")
        elif r < args.low_light_probability + args.high_bright_probability:
            args.high_bright = True
            logger.info(f"[Realism] High-bright randomly activated ({args.high_bright_probability*100:.0f}% chance)")

    convert_dataset(args.input, args.output,
                    args.geodata_dir, args.convert_lod,
                    dop_dir=args.dop_dir, dsm_dir=args.dsm_dir,
                    lod1_dir=args.lod1_dir, lod2_dir=args.lod2_dir, als_dir=args.als_dir,
                    render_mesh=args.save_render, mesh_dir=args.mesh_dir,
                    high_quality=args.high_quality, realism=args.realism,
                    lighting=args.lighting, temperature=args.temperature,
                    sun_intensity=args.sun_intensity, ibl_intensity=args.ibl_intensity,
                    sr_mode=args.sr_mode, sr_batch_size=args.sr_batch_size,
                    save_depth=args.save_depth, save_normals=args.save_normals,
                    max_frames=args.max_frames,
                    background_color=args.background_color,
                    auto_geodata=args.auto_geodata, map_dir=args.map_dir,
                    geodata_margin=args.geodata_margin,
                    trajectory_noise=args.trajectory_noise,
                    position_noise_m=args.position_noise,
                    rotation_noise_deg=args.rotation_noise,
                    roll_noise_deg=args.roll_noise,
                    bank_on_turns=args.bank_on_turns,
                    bank_factor=args.bank_factor,
                    noise_smoothness=args.noise_smoothness,
                    noise_seed=args.noise_seed,
                    motion_blur=args.motion_blur,
                    blur_strength=args.blur_strength,
                    random_sun=args.random_sun,
                    sun_seed=args.sun_seed,
                    low_light=args.low_light,
                    high_bright=args.high_bright,
                    sudden_jitter=args.sudden_jitter,
                    jitter_frame=args.jitter_frame,
                    jitter_probability=args.jitter_probability,
                    jitter_magnitude_deg=args.jitter_magnitude,
                    jitter_seed=args.jitter_seed,
                    wind_gusts=args.wind_gusts,
                    gust_probability=args.gust_probability,
                    n_episodes_max=args.gust_episodes_max,
                    episode_duration_s=args.gust_duration,
                    gust_magnitude_deg=args.gust_magnitude,
                    gust_seed=args.gust_seed,
                    target_fps=args.target_fps,
                    keep_rendering=args.keep_rendering,
                    velocity_bound=args.velocity_bound,
                    max_speed_kmh=args.max_speed,
                    frames_dir=args.frames_dir,
                    priors_dir=args.priors_dir,
                    video_src=args.video)

if __name__ == '__main__':
    main()
