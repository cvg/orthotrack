"""
COLMAP and GLOMAP (SfM) baseline wrappers for OrthoTrack evaluation.

Unlike VO/SLAM methods which process frames sequentially, SfM methods
(COLMAP, GLOMAP) perform offline reconstruction from all frames.  Both
produce poses in an arbitrary coordinate frame, so we apply the same
alignment protocol as the VO baselines (first_frame, first_frame_scale,
ate_sim3).

These wrappers call the COLMAP / GLOMAP command-line binaries.  By default
the search order for the binary is:
  1.  PROJECT_DIR/tmp/sfm_install/bin/{colmap,glomap}   (custom-built)
  2.  system PATH"""

import os
import shutil
import struct
import subprocess
import time
import json
import numpy as np
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from orthotrack.baselines.vo_wrapper import VOBaselineWrapper


# ===================================================================== #
#  Utility: extract video frames to a directory                         #
# ===================================================================== #

def extract_video_frames(
    video_path: str,
    frame_indices: List[int],
    output_dir: str,
    target_width: Optional[int] = None,
) -> Tuple[List[str], float]:
    """Extract specific frames from a video or image directory and save as images.

    Returns
    -------
    image_names : list[str]  — filenames in output_dir (e.g. "frame_000123.jpg")
    scale_factor : float     — resize scale applied (1.0 if no resize)"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    source_path = Path(video_path)

    if source_path.is_dir():
        # Image directory mode: read images directly
        # Get dimensions from first available image
        first_img = None
        for fi in frame_indices:
            for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
                candidate = source_path / f"{fi:06d}{ext}"
                if candidate.exists():
                    first_img = candidate
                    break
            if first_img:
                break
        if first_img is None:
            return [], 1.0
        sample = cv2.imread(str(first_img))
        orig_w = sample.shape[1]
        if target_width is not None and target_width < orig_w:
            scale_factor = target_width / orig_w
        else:
            scale_factor = 1.0

        image_names = []
        for fi in frame_indices:
            name = f"frame_{fi:06d}.jpg"
            dst = out / name
            if dst.exists():
                image_names.append(name)
                continue
            img_path = None
            for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
                candidate = source_path / f"{fi:06d}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                continue
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            if scale_factor != 1.0:
                h, w = frame.shape[:2]
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), frame)
            image_names.append(name)
        return image_names, scale_factor

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if target_width is not None and target_width < orig_w:
        scale_factor = target_width / orig_w
    else:
        scale_factor = 1.0

    image_names = []
    for fi in frame_indices:
        name = f"frame_{fi:06d}.jpg"
        dst = out / name
        if dst.exists():
            image_names.append(name)
            continue

        if fi >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        if scale_factor != 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(dst), frame)
        image_names.append(name)

    cap.release()
    return image_names, scale_factor


# ===================================================================== #
#  Utility: read COLMAP binary model                                    #
# ===================================================================== #

def read_colmap_images_binary(path: str) -> dict:
    """Read images.bin from a COLMAP binary model.

    Returns dict[image_name] -> {
        'qw', 'qx', 'qy', 'qz': float  (W2C quaternion, scalar-first),
        'tx', 'ty', 'tz': float  (W2C translation),
        'camera_id': int,
        'image_id': int,
    }"""
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            # Read image name (null-terminated)
            name_chars = []
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name_chars.append(ch.decode("ascii"))
            image_name = "".join(name_chars)
            # Read 2D points (skip for now)
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2D * 24)  # x, y, point3D_id

            images[image_name] = {
                "image_id": image_id,
                "camera_id": camera_id,
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                "tx": tx, "ty": ty, "tz": tz,
            }
    return images


def read_colmap_images_text(path: str) -> dict:
    """Read images.txt from a COLMAP text model."""
    images = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue  # Skip POINTS2D lines
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8])
            image_name = parts[9]

            images[image_name] = {
                "image_id": image_id,
                "camera_id": camera_id,
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                "tx": tx, "ty": ty, "tz": tz,
            }
            # Next line is POINTS2D — skip it
            next(f, None)
    return images


def read_colmap_model_images(model_dir: str) -> dict:
    """Read images from a COLMAP model (binary or text)."""
    p = Path(model_dir)
    if (p / "images.bin").exists():
        return read_colmap_images_binary(str(p / "images.bin"))
    elif (p / "images.txt").exists():
        return read_colmap_images_text(str(p / "images.txt"))
    else:
        raise FileNotFoundError(f"No images.bin or images.txt in {model_dir}")


def colmap_image_to_pose(img_data: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Convert COLMAP image quaternion+translation to rotation matrix and camera center.

    COLMAP stores W2C: p_cam = R @ p_world + t
    Camera center (C2W position): C = -R^T @ t

    Returns
    -------
    R_c2w : (3, 3) — camera-to-world rotation
    center : (3,) — camera center in world frame"""
    from scipy.spatial.transform import Rotation as R

    qw, qx, qy, qz = img_data["qw"], img_data["qx"], img_data["qy"], img_data["qz"]
    tx, ty, tz = img_data["tx"], img_data["ty"], img_data["tz"]

    # scipy expects [qx, qy, qz, qw]
    R_w2c = R.from_quat([qx, qy, qz, qw]).as_matrix()
    t_w2c = np.array([tx, ty, tz])

    R_c2w = R_w2c.T
    center = -R_c2w @ t_w2c

    return R_c2w, center


# ===================================================================== #
#  Find binaries                                                        #
# ===================================================================== #

PROJECT_DIR = Path(__file__).resolve().parents[2]  # OrthoTrack root

def find_binary(name: str) -> str:
    """Find COLMAP or GLOMAP binary: first in our custom build, then system."""
    custom = PROJECT_DIR / "tmp" / "sfm_install" / "bin" / name
    if custom.exists():
        return str(custom)
    system = shutil.which(name)
    if system:
        return system
    raise FileNotFoundError(
        f"'{name}' binary not found. Build it with: sbatch slurm/build_colmap_glomap.sbatch"
    )


# ===================================================================== #
#  Base SfM Wrapper                                                     #
# ===================================================================== #

class SfMBaseWrapper(VOBaselineWrapper):
    """Base class for offline SfM methods (COLMAP, GLOMAP).

    Handles common steps:
    1. Extract video frames to a working directory
    2. Create COLMAP database
    3. Extract SIFT features
    4. Run sequential matching
    5. [subclass] Run mapping (incremental or global)
    6. Parse reconstructed poses"""

    name: str = "sfm_base"

    def __init__(
        self,
        use_gpu: bool = True,
        target_width: Optional[int] = None,
        max_num_features: int = 8192,
        matching_type: str = "sequential",
    ):
        self._use_gpu = use_gpu
        self._target_width = target_width
        self._max_num_features = max_num_features
        self._matching_type = matching_type  # "sequential" or "exhaustive"

    @abstractmethod
    def _run_mapper(
        self,
        database_path: str,
        image_path: str,
        output_path: str,
        intrinsics: np.ndarray,
        image_names: List[str],
    ) -> str:
        """Run the SfM mapper. Return path to the best model directory."""
        ...

    def _run_vo(
        self,
        video_path: str,
        intrinsics: np.ndarray,
        frame_indices: List[int],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[float]]:
        """Run SfM pipeline: extract frames → SIFT → match → map → parse.

        Returns positions, rotations, timings (same interface as VO wrappers)."""
        # Working directory: ./tmp/sfm_work/<method>/<sequence_hash>/
        # Always cleared to avoid stale images/database from a previous run
        # with a different stride or frame range.
        import shutil
        seq_name = Path(video_path).parent.name
        work_dir = PROJECT_DIR / "tmp" / "sfm_work" / self.name / seq_name
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        image_dir = work_dir / "images"
        database_path = str(work_dir / "database.db")
        output_path = str(work_dir / "sparse")

        # ── Step 1: Extract frames ──────────────────────────────
        print(f"[{self.name}] Extracting {len(frame_indices)} frames ...")
        t0 = time.time()
        image_names, scale_factor = extract_video_frames(
            video_path, frame_indices, str(image_dir),
            target_width=self._target_width,
        )
        t_extract = time.time() - t0
        print(f"[{self.name}] Extracted {len(image_names)} frames in {t_extract:.1f}s "
              f"(scale={scale_factor:.3f})")

        if len(image_names) < 3:
            raise RuntimeError(f"Too few frames extracted ({len(image_names)})")

        # Scale intrinsics if resized
        K = intrinsics.copy()
        K[0, 0] *= scale_factor
        K[0, 2] *= scale_factor
        K[1, 1] *= scale_factor
        K[1, 2] *= scale_factor

        # ── Step 2: Feature extraction ──────────────────────────
        print(f"[{self.name}] Running SIFT feature extraction ...")
        t1 = time.time()
        self._run_feature_extraction(database_path, str(image_dir), K)
        t_feat = time.time() - t1
        print(f"[{self.name}] Feature extraction: {t_feat:.1f}s")

        # ── Step 3: Matching ────────────────────────────────────
        print(f"[{self.name}] Running {self._matching_type} matching ...")
        t2 = time.time()
        self._run_matching(database_path)
        t_match = time.time() - t2
        print(f"[{self.name}] Matching: {t_match:.1f}s")

        # ── Step 4: Mapping ─────────────────────────────────────
        print(f"[{self.name}] Running mapper ...")
        t3 = time.time()
        best_model = self._run_mapper(
            database_path, str(image_dir), output_path, K, image_names,
        )
        t_map = time.time() - t3
        print(f"[{self.name}] Mapping: {t_map:.1f}s")

        total_time = t_feat + t_match + t_map

        # ── Step 5: Parse poses ─────────────────────────────────
        model_images = read_colmap_model_images(best_model)
        print(f"[{self.name}] Reconstructed {len(model_images)}/{len(image_names)} images")

        # Map frame_index -> pose
        positions = np.zeros((len(frame_indices), 3))
        rotations = np.zeros((len(frame_indices), 3, 3))
        timings = [total_time / len(frame_indices)] * len(frame_indices)

        n_registered = 0
        for idx, fi in enumerate(frame_indices):
            name = f"frame_{fi:06d}.jpg"
            if name in model_images:
                R_c2w, center = colmap_image_to_pose(model_images[name])
                positions[idx] = center
                rotations[idx] = R_c2w
                n_registered += 1
            else:
                # Unregistered frame — interpolate from neighbours
                positions[idx] = np.nan
                rotations[idx] = np.nan

        # Interpolate unregistered frames
        registered = ~np.isnan(positions[:, 0])
        if registered.sum() > 0 and (~registered).sum() > 0:
            positions = self._interpolate_poses(positions, registered)
            rotations = self._interpolate_rotations(rotations, registered)

        print(f"[{self.name}] Registered: {n_registered}/{len(frame_indices)} "
              f"({100 * n_registered / len(frame_indices):.1f}%)")

        return positions, rotations, timings

    def _run_feature_extraction(
        self,
        database_path: str,
        image_path: str,
        K: np.ndarray,
    ):
        """Run SIFT feature extraction with known intrinsics."""
        colmap_bin = find_binary("colmap")

        # Remove old database if it exists (re-run)
        if Path(database_path).exists():
            Path(database_path).unlink()

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        cmd = [
            colmap_bin, "feature_extractor",
            "--database_path", database_path,
            "--image_path", image_path,
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "PINHOLE",
            f"--ImageReader.camera_params", f"{fx},{fy},{cx},{cy}",
            f"--SiftExtraction.max_num_features", str(self._max_num_features),
            f"--SiftExtraction.use_gpu", "1" if self._use_gpu else "0",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            # If GPU SIFT fails (any reason: CUDA error, no display, OpenGL, etc.), retry with CPU
            if self._use_gpu:
                print(f"[{self.name}] GPU SIFT failed (exit {result.returncode}), falling back to CPU ...")
                print(f"[{self.name}] GPU SIFT stderr: {result.stderr[-500:]}")
                if Path(database_path).exists():
                    Path(database_path).unlink()
                cmd[-1] = "0"  # use_gpu = 0
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Feature extraction failed (exit {result.returncode}):\n"
                    f"{result.stderr[-1000:]}"
                )

    def _run_matching(self, database_path: str):
        """Run feature matching."""
        colmap_bin = find_binary("colmap")

        if self._matching_type == "sequential":
            cmd = [
                colmap_bin, "sequential_matcher",
                "--database_path", database_path,
                "--SiftMatching.use_gpu", "1" if self._use_gpu else "0",
                "--SequentialMatching.overlap", "10",
                "--SequentialMatching.loop_detection", "0",
            ]
        elif self._matching_type == "exhaustive":
            cmd = [
                colmap_bin, "exhaustive_matcher",
                "--database_path", database_path,
                "--SiftMatching.use_gpu", "1" if self._use_gpu else "0",
            ]
        else:
            raise ValueError(f"Unknown matching type: {self._matching_type}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            # GPU fallback
            if self._use_gpu:
                print(f"[{self.name}] GPU matching failed, falling back to CPU ...")
                cmd_cpu = [c if c != "1" or i == 0 else "0"
                           for i, c in enumerate(cmd)]
                # More precisely, replace use_gpu value
                for i, c in enumerate(cmd):
                    if c == "--SiftMatching.use_gpu":
                        cmd[i + 1] = "0"
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Matching failed (exit {result.returncode}):\n"
                    f"{result.stderr[-1000:]}"
                )

    @staticmethod
    def _interpolate_poses(
        positions: np.ndarray, registered: np.ndarray,
    ) -> np.ndarray:
        """Fill NaN positions via linear interpolation."""
        result = positions.copy()
        reg_indices = np.where(registered)[0]
        unreg_indices = np.where(~registered)[0]

        for dim in range(3):
            result[unreg_indices, dim] = np.interp(
                unreg_indices, reg_indices, positions[reg_indices, dim],
            )
        return result

    @staticmethod
    def _interpolate_rotations(
        rotations: np.ndarray, registered: np.ndarray,
    ) -> np.ndarray:
        """Fill NaN rotations via SLERP interpolation."""
        from scipy.spatial.transform import Rotation as R, Slerp

        result = rotations.copy()
        reg_indices = np.where(registered)[0]
        unreg_indices = np.where(~registered)[0]

        if len(reg_indices) < 2:
            # Can't interpolate, fill with nearest
            for ui in unreg_indices:
                nearest = reg_indices[np.argmin(np.abs(reg_indices - ui))]
                result[ui] = rotations[nearest]
            return result

        # Build slerp from registered rotations
        rots_reg = R.from_matrix(rotations[reg_indices])
        slerp = Slerp(reg_indices.astype(float), rots_reg)

        # Clamp unregistered indices to the registered range
        lo, hi = reg_indices[0], reg_indices[-1]
        clamped = np.clip(unreg_indices, lo, hi).astype(float)
        interp_rots = slerp(clamped)
        result[unreg_indices] = interp_rots.as_matrix()

        return result


# ===================================================================== #
#  COLMAP — Incremental SfM                                             #
# ===================================================================== #

class COLMAPWrapper(SfMBaseWrapper):
    """COLMAP incremental SfM wrapper."""

    name = "colmap"

    def __init__(
        self,
        use_gpu: bool = True,
        target_width: Optional[int] = None,
        max_num_features: int = 8192,
        matching_type: str = "sequential",
    ):
        super().__init__(
            use_gpu=use_gpu,
            target_width=target_width,
            max_num_features=max_num_features,
            matching_type=matching_type,
        )

    def _run_mapper(
        self,
        database_path: str,
        image_path: str,
        output_path: str,
        intrinsics: np.ndarray,
        image_names: List[str],
    ) -> str:
        """Run COLMAP incremental mapper. Returns path to best model."""
        colmap_bin = find_binary("colmap")

        # Clean previous output
        out = Path(output_path)
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)

        cmd = [
            colmap_bin, "mapper",
            "--database_path", database_path,
            "--image_path", image_path,
            "--output_path", output_path,
            "--Mapper.ba_refine_focal_length", "0",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params", "0",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
        if result.returncode != 0:
            raise RuntimeError(
                f"COLMAP mapper failed (exit {result.returncode}):\n"
                f"{result.stderr[-1000:]}"
            )

        # COLMAP produces numbered sub-models: sparse/0/, sparse/1/, ...
        # Pick the largest one (most registered images)
        return self._select_best_model(output_path)

    @staticmethod
    def _select_best_model(output_path: str) -> str:
        """Select the sub-model with the most registered images."""
        models = sorted(Path(output_path).iterdir())
        if not models:
            raise RuntimeError(f"No models produced in {output_path}")

        best = None
        best_count = -1
        for m in models:
            if not m.is_dir():
                continue
            try:
                imgs = read_colmap_model_images(str(m))
                if len(imgs) > best_count:
                    best_count = len(imgs)
                    best = str(m)
            except FileNotFoundError:
                continue

        if best is None:
            raise RuntimeError(f"No valid sub-models in {output_path}")

        print(f"[colmap] Selected model with {best_count} images from {best}")
        return best


# ===================================================================== #
#  GLOMAP — Global SfM                                                  #
# ===================================================================== #

class GLOMAPWrapper(SfMBaseWrapper):
    """GLOMAP global SfM wrapper."""

    name = "glomap"

    def __init__(
        self,
        use_gpu: bool = True,
        target_width: Optional[int] = None,
        max_num_features: int = 8192,
        matching_type: str = "sequential",
    ):
        super().__init__(
            use_gpu=use_gpu,
            target_width=target_width,
            max_num_features=max_num_features,
            matching_type=matching_type,
        )

    def _run_mapper(
        self,
        database_path: str,
        image_path: str,
        output_path: str,
        intrinsics: np.ndarray,
        image_names: List[str],
    ) -> str:
        """Run GLOMAP global mapper. Returns path to best model."""
        glomap_bin = find_binary("glomap")

        # Clean previous output
        out = Path(output_path)
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)

        cmd = [
            glomap_bin, "mapper",
            "--database_path", database_path,
            "--image_path", image_path,
            "--output_path", output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
        if result.returncode != 0:
            raise RuntimeError(
                f"GLOMAP mapper failed (exit {result.returncode}):\n"
                f"{result.stderr[-1000:]}"
            )

        # GLOMAP may produce numbered sub-models similar to COLMAP
        return self._select_best_model(output_path)

    @staticmethod
    def _select_best_model(output_path: str) -> str:
        """Select the sub-model with the most registered images."""
        out = Path(output_path)

        # GLOMAP might output directly to output_path (model files at top level)
        if (out / "images.bin").exists() or (out / "images.txt").exists():
            return str(out)

        # Or numbered sub-models like COLMAP
        models = sorted(out.iterdir())
        if not models:
            raise RuntimeError(f"No models produced in {output_path}")

        best = None
        best_count = -1
        for m in models:
            if not m.is_dir():
                continue
            try:
                imgs = read_colmap_model_images(str(m))
                if len(imgs) > best_count:
                    best_count = len(imgs)
                    best = str(m)
            except FileNotFoundError:
                continue

        if best is None:
            raise RuntimeError(f"No valid sub-models in {output_path}")

        print(f"[glomap] Selected model with {best_count} images from {best}")
        return best
