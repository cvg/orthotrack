"""
ORB-SLAM3 wrapper for OrthoTrack VO baseline evaluation.

Uses the compiled mono_headless binary (headless, no viewer) to track
monocular video sequences and produce TUM-format trajectories.

Build ORB-SLAM3 first:
    bash thirdparty/ORB_SLAM3/build_headless.sh

The wrapper:
1. Extracts video frames to ./tmp/orbslam3_runs/<runid>/
2. Writes a TUM-format rgb.txt with per-frame timestamps
3. Generates an ORB-SLAM3 settings YAML from intrinsics
4. Runs: mono_headless <vocab> <settings.yaml> <frames_dir> <output_path>
5. Parses the saved trajectory and maps back to input frame indices"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from orthotrack.baselines.vo_wrapper import VOBaselineWrapper

# Paths relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ORBSLAM3_DIR = _PROJECT_ROOT / "thirdparty" / "ORB_SLAM3"
_VOCAB_PATH = str(_ORBSLAM3_DIR / "Vocabulary" / "ORBvoc.txt")
_BINARY = str(_ORBSLAM3_DIR / "Examples" / "Monocular" / "mono_headless")

# Retry with fewer features if initial run fails to initialize
_MIN_NFEATURES = 500


# ──────────────────────────────────────────────────────────────────────────────
#  Settings YAML generation
# ──────────────────────────────────────────────────────────────────────────────

def _write_settings_yaml(
    intrinsics: np.ndarray,
    width: int,
    height: int,
    fps: float,
    output_path: str,
    nFeatures: int = 1200,
    nLevels: int = 8,
    scaleFactor: float = 1.2,
    iniThFAST: int = 20,
    minThFAST: int = 7,
) -> None:
    """Write an ORB-SLAM3 monocular YAML settings file."""
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    # Our MovingDrone sequences have no radial distortion (rendered / undistorted)
    yaml_content = f"""%YAML:1.0

File.version: "1.0"

Camera.type: "PinHole"

# Camera calibration — MovingDrone (no distortion, rendered/undistorted videos)
Camera1.fx: {fx:.6f}
Camera1.fy: {fy:.6f}
Camera1.cx: {cx:.6f}
Camera1.cy: {cy:.6f}

Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0
Camera1.k3: 0.0

Camera.fps: {int(round(fps)):d}

# Image color order: 0=BGR, 1=RGB (OpenCV reads BGR, ORB-SLAM3 expects BGR)
Camera.RGB: 0

Camera.width: {width:d}
Camera.height: {height:d}

# ORB Extractor Parameters
ORBextractor.nFeatures: {nFeatures:d}
ORBextractor.scaleFactor: {scaleFactor:.2f}
ORBextractor.nLevels: {nLevels:d}
ORBextractor.iniThFAST: {iniThFAST:d}
ORBextractor.minThFAST: {minThFAST:d}

# Viewer (headless — these values are ignored at runtime)
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""
    with open(output_path, 'w') as f:
        f.write(yaml_content)


# ──────────────────────────────────────────────────────────────────────────────
#  Trajectory parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_tum_trajectory(filepath: str) -> dict:
    """Parse a TUM-format trajectory file.

    Format: ``timestamp tx ty tz qx qy qz qw``  (C2W, camera center + rotation)

    Returns
    -------
    dict mapping timestamp_float -> (position_xyz, rotation_matrix_C2W)"""
    poses = {}
    if not os.path.exists(filepath):
        return poses

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            # scipy quaternion convention: [qx, qy, qz, qw]
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            poses[ts] = (np.array([tx, ty, tz]), R)

    return poses


# ──────────────────────────────────────────────────────────────────────────────
#  Wrapper class
# ──────────────────────────────────────────────────────────────────────────────

class ORBSLAM3Wrapper(VOBaselineWrapper):
    """Wrapper for ORB-SLAM3 monocular tracking (headless server mode).

    Uses the compiled ``mono_headless`` binary which sets ``bUseViewer=false``.

    Build first:
        bash thirdparty/ORB_SLAM3/build_headless.sh

    Parameters
    ----------
    vocab_path : str
        Path to ORBvoc.txt (default: thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt)
    binary : str
        Path to mono_headless binary.
    max_image_dim : int or None
        If set, resize frames so the larger side <= max_image_dim.
        Reduces memory and speeds up tracking for high-res inputs.
    nFeatures : int
        Number of ORB features per frame (default 1200).
    fps : float or None
        Video FPS for camera settings YAML. Auto-detected from video if None."""

    name = "orb_slam3"

    def __init__(
        self,
        vocab_path: str = _VOCAB_PATH,
        binary: str = _BINARY,
        max_image_dim: Optional[int] = 1280,
        nFeatures: int = 1200,
        fps: Optional[float] = None,
    ):
        self._vocab_path = vocab_path
        self._binary = binary
        self._max_image_dim = max_image_dim
        self._nFeatures = nFeatures
        self._fps = fps

        if not Path(self._binary).exists():
            raise FileNotFoundError(
                f"ORB-SLAM3 mono_headless binary not found at: {self._binary}\n"
                "Build with:  bash thirdparty/ORB_SLAM3/build_headless.sh"
            )
        if not Path(self._vocab_path).exists():
            raise FileNotFoundError(
                f"ORB vocabulary not found at: {self._vocab_path}\n"
                "Build ORB-SLAM3 first to extract the vocabulary."
            )

    def _run_vo(
        self,
        video_path: str,
        intrinsics: np.ndarray,
        frame_indices: List[int],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[float]]:
        """Extract frames, run ORB-SLAM3, parse trajectory."""
        # Use ./tmp/ per project convention
        tmp_base = _PROJECT_ROOT / "tmp" / "orbslam3_runs"
        tmp_base.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(dir=str(tmp_base), prefix="orb_"))

        try:
            return self._run_in_dir(run_dir, video_path, intrinsics, frame_indices)
        finally:
            # Clean up temp directory to avoid disk filling up
            import shutil
            try:
                shutil.rmtree(str(run_dir))
            except Exception:
                pass

    def _run_in_dir(
        self,
        run_dir: Path,
        video_path: str,
        intrinsics: np.ndarray,
        frame_indices: List[int],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[float]]:
        # --- Step 1: Get source metadata ---
        source_path = Path(video_path)
        if source_path.is_dir():
            # Image directory: get dimensions from first image
            first_img = sorted(source_path.glob("*.jpg")) or sorted(source_path.glob("*.png"))
            if not first_img:
                raise RuntimeError(f"No images found in {video_path}")
            sample = cv2.imread(str(first_img[0]))
            orig_h, orig_w = sample.shape[:2]
            video_fps = 10.0  # default for image sequences
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()

        fps = self._fps if self._fps is not None else video_fps

        # Determine resize scale
        if self._max_image_dim is not None:
            max_dim = max(orig_w, orig_h)
            if max_dim > self._max_image_dim:
                scale = self._max_image_dim / max_dim
                target_w = int(orig_w * scale)
                target_h = int(orig_h * scale)
            else:
                scale = 1.0
                target_w = orig_w
                target_h = orig_h
        else:
            scale = 1.0
            target_w = orig_w
            target_h = orig_h

        # Scale intrinsics if resizing
        K = intrinsics.copy().astype(np.float64)
        if scale != 1.0:
            K[0, 0] *= scale  # fx
            K[0, 2] *= scale  # cx
            K[1, 1] *= scale  # fy
            K[1, 2] *= scale  # cy

        # --- Step 2: Extract frames ---
        img_dir = run_dir / "rgb"
        img_dir.mkdir()

        print(f"  Extracting {len(frame_indices)} frames to {img_dir} ...")
        t0 = time.time()
        self._extract_frames(str(video_path), frame_indices,
                             str(img_dir), target_w, target_h)
        print(f"  Extracted in {time.time() - t0:.1f}s")

        # --- Step 3: Write rgb.txt (TUM format) ---
        # Use frame_index / fps as timestamp so motion model runs correctly.
        rgb_txt = run_dir / "rgb.txt"
        with open(rgb_txt, 'w') as f:
            f.write("# MovingDrone sequence converted to TUM format\n")
            f.write("# timestamp filename\n")
            f.write("# ------\n")
            for fi in frame_indices:
                ts = fi / fps
                fname = f"rgb/{fi:06d}.png"
                f.write(f"{ts:.6f} {fname}\n")

        # --- Step 4: Write ORB-SLAM3 settings YAML ---
        settings_yaml = str(run_dir / "camera_settings.yaml")
        _write_settings_yaml(
            intrinsics=K,
            width=target_w,
            height=target_h,
            fps=fps,
            output_path=settings_yaml,
            nFeatures=self._nFeatures,
        )

        # --- Step 5: Run mono_headless ---
        output_traj = str(run_dir / "trajectory")
        cmd = [
            self._binary,
            self._vocab_path,
            settings_yaml,
            str(run_dir),
            output_traj,
            "--no-wait",
        ]

        # Build LD_LIBRARY_PATH so the binary can find bundled OpenCV 4.6 libs
        # and ORB-SLAM3's own shared libraries on any compute node.
        opencv_libs = str(_ORBSLAM3_DIR / "opencv_libs")
        orbslam_lib = str(_ORBSLAM3_DIR / "lib")
        dbow2_lib = str(_ORBSLAM3_DIR / "Thirdparty" / "DBoW2" / "lib")
        g2o_lib = str(_ORBSLAM3_DIR / "Thirdparty" / "g2o" / "lib")
        extra_ld_path = f"{opencv_libs}:{orbslam_lib}:{dbow2_lib}:{g2o_lib}"

        env = os.environ.copy()
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        # System libs first (for correct boost/libstdc++), then ORB-SLAM3 libs
        sys_libs = "/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
        env["LD_LIBRARY_PATH"] = f"{sys_libs}:{extra_ld_path}:{existing_ld}" if existing_ld else f"{sys_libs}:{extra_ld_path}"

        print(f"  Running ORB-SLAM3: {' '.join(cmd[:4])} ...")
        t0 = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            env=env,
        )
        elapsed = time.time() - t0

        # Report actual SLAM processing time (excluding frame extraction)
        # so that run_sequence() uses this for FPS instead of wall time.
        self._processing_time = elapsed

        if result.returncode != 0:
            print(f"  WARNING: ORB-SLAM3 exited with code {result.returncode}")
            print(f"  STDOUT (last 20 lines):\n" +
                  "\n".join(result.stdout.splitlines()[-20:]))
            print(f"  STDERR (last 20 lines):\n" +
                  "\n".join(result.stderr.splitlines()[-20:]))
        else:
            # Print key info from stdout
            for line in result.stdout.splitlines():
                if any(kw in line for kw in ['median tracking', 'mean tracking',
                                              'Trajectory saved', 'Start processing',
                                              'Images in the sequence']):
                    print(f"  {line.strip()}")

        # --- Step 6: Parse trajectories ---
        # Try full trajectory first; fall back to keyframe trajectory
        full_traj_path = output_traj + "_full.txt"
        kf_traj_path = output_traj

        # Build timestamp → pose map from best available trajectory
        traj_poses = {}
        if os.path.exists(full_traj_path) and os.path.getsize(full_traj_path) > 0:
            traj_poses = _parse_tum_trajectory(full_traj_path)
            print(f"  Parsed {len(traj_poses)} poses from full trajectory")
        if (not traj_poses) and os.path.exists(kf_traj_path) and os.path.getsize(kf_traj_path) > 0:
            traj_poses = _parse_tum_trajectory(kf_traj_path)
            print(f"  Parsed {len(traj_poses)} poses from keyframe trajectory")

        if not traj_poses:
            print("  WARNING: No trajectory produced! ORB-SLAM3 likely failed to initialize.")
            n = len(frame_indices)
            positions = np.zeros((n, 3))
            rotations = np.tile(np.eye(3), (n, 1, 1))
            timings = [elapsed / max(n, 1)] * n
            return positions, rotations, timings

        # --- Step 7: Load per-frame timings ---
        timing_file = output_traj + ".timing"
        timing_by_ts = {}
        if os.path.exists(timing_file):
            with open(timing_file) as tf:
                for line in tf:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        ts = float(parts[0])
                        t_val = float(parts[1])
                        timing_by_ts[ts] = t_val

        # --- Step 8: Map back to frame_indices ---
        # Match timestamps: frame fi → timestamp fi/fps
        positions = []
        rotations = []
        timings = []
        last_pos = np.zeros(3)
        last_rot = np.eye(3)

        for fi in frame_indices:
            ts = fi / fps
            # Find nearest timestamp in trajectory (within 0.5 frame tolerance)
            tol = 0.5 / fps
            best_match = None
            best_dist = float('inf')
            for ts_pred in traj_poses:
                d = abs(ts_pred - ts)
                if d < best_dist:
                    best_dist = d
                    best_match = ts_pred

            if best_match is not None and best_dist <= tol:
                pos, rot = traj_poses[best_match]
                last_pos = pos.copy()
                last_rot = rot.copy()
            # else: use last known pose (propagation for lost frames)

            positions.append(last_pos.copy())
            rotations.append(last_rot.copy())
            timings.append(timing_by_ts.get(ts, elapsed / len(frame_indices)))

        return np.array(positions), np.array(rotations), timings

    @staticmethod
    def _extract_frames(
        video_path: str,
        frame_indices: List[int],
        output_dir: str,
        target_w: int,
        target_h: int,
    ) -> None:
        """Extract specific frames from video or image directory to output_dir as <fi:06d>.png."""
        source_path = Path(video_path)

        if source_path.is_dir():
            # Image directory: read and optionally resize images
            n_extracted = 0
            for fi in sorted(frame_indices):
                # Try common extensions
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
                orig_h, orig_w = frame.shape[:2]
                if target_w != orig_w or target_h != orig_h:
                    frame = cv2.resize(frame, (target_w, target_h))
                out_path = os.path.join(output_dir, f"{fi:06d}.png")
                cv2.imwrite(out_path, frame)
                n_extracted += 1
            if n_extracted == 0:
                raise RuntimeError(f"No frames extracted from {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_set = set(frame_indices)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        do_resize = (target_w != orig_w or target_h != orig_h)

        # Sequential reading (avoid unreliable CAP_PROP_POS_FRAMES seek for H.264)
        # This matches the pattern described in MovingDroneBase._get_video_frame()
        sorted_indices = sorted(frame_indices)
        idx_set = set(sorted_indices)
        current_frame = 0
        n_extracted = 0

        for fi in sorted_indices:
            if fi >= total_frames:
                continue
            # Seek forward
            while current_frame <= fi:
                ret, frame = cap.read()
                if not ret:
                    break
                if current_frame == fi:
                    if do_resize:
                        frame = cv2.resize(frame, (target_w, target_h))
                    out_path = os.path.join(output_dir, f"{fi:06d}.png")
                    cv2.imwrite(out_path, frame)
                    n_extracted += 1
                current_frame += 1

        cap.release()
        if n_extracted == 0:
            raise RuntimeError(f"No frames extracted from {video_path}")
