"""
DSO (Direct Sparse Odometry) wrapper for OrthoTrack evaluation.

DSO is a C++ visual odometry system.  We run it as a subprocess:
1. Extract video frames to a temporary image directory.
2. Write a DSO-format calibration file from intrinsics.
3. Run the ``dso_headless`` binary.
4. Parse the TUM-format ``result.txt`` (timestamp x y z qx qy qz qw).
5. Align via Sim(3) Umeyama and compute evaluation metrics.

Requirements:
    - Build DSO headless:
        cd thirdparty/DSO && mkdir build && cd build
        CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake .. \\
            -DCMAKE_BUILD_TYPE=Release \\
            -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \\
            -DBoost_NO_BOOST_CMAKE=ON
        make -j$(nproc)"""

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

# Default DSO binary path (relative to project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DSO_BINARY = str(_PROJECT_ROOT / "thirdparty" / "DSO" / "build" / "bin" / "dso_headless")


class DSOWrapper(VOBaselineWrapper):
    """Wrapper for DSO (Direct Sparse Odometry, Engel et al. 2018).

    Runs the DSO headless binary as a subprocess on extracted video frames.

    Parameters
    ----------
    dso_binary : str
        Path to the ``dso_headless`` executable.
    preset : int
        DSO preset (0 = default, 2 = fast).
    mode : int
        Photometric mode (1 = no photometric calibration).
    max_image_dim : int or None
        If set, resize images so that the larger dimension is at most this value.
        Helps with very high-res inputs (e.g., 1920x1080 -> 960x540)."""

    name = "dso"

    def __init__(
        self,
        dso_binary: str = DSO_BINARY,
        preset: int = 0,
        mode: int = 1,
        max_image_dim: Optional[int] = None,
    ):
        self._dso_binary = dso_binary
        self._preset = preset
        self._mode = mode
        self._max_image_dim = max_image_dim

        if not Path(self._dso_binary).exists():
            raise FileNotFoundError(
                f"DSO binary not found at {self._dso_binary}.\n"
                "Build it with:\n"
                "  cd thirdparty/DSO && mkdir -p build && cd build\n"
                "  CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake .. "
                "-DCMAKE_BUILD_TYPE=Release "
                "-DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4\n"
                "  make -j$(nproc)"
            )

    def _run_vo(
        self,
        video_path: str,
        intrinsics: np.ndarray,
        frame_indices: List[int],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[float]]:
        """Run DSO on video frames.

        1. Extract frames to temp directory.
        2. Write DSO calibration file.
        3. Run DSO subprocess.
        4. Parse result.txt.
        5. Return positions, rotations, timings."""
        # Use ./tmp/ relative to project root per convention
        tmp_base = _PROJECT_ROOT / "tmp" / "dso_runs"
        tmp_base.mkdir(parents=True, exist_ok=True)

        # Create a unique temp directory for this run
        run_dir = Path(tempfile.mkdtemp(dir=str(tmp_base), prefix="dso_"))
        img_dir = run_dir / "images"
        img_dir.mkdir()

        try:
            # Step 1: Extract frames
            print(f"  Extracting {len(frame_indices)} frames to {img_dir} ...")
            t_extract_start = time.time()
            orig_w, orig_h = self._extract_frames(
                video_path, frame_indices, str(img_dir),
            )
            t_extract = time.time() - t_extract_start
            print(f"  Extracted in {t_extract:.1f}s")

            # Determine actual image size (after optional resize)
            sample_img = cv2.imread(str(next(img_dir.iterdir())))
            actual_h, actual_w = sample_img.shape[:2]

            # Scale intrinsics if images were resized
            fx, fy, cx, cy = (
                intrinsics[0, 0],
                intrinsics[1, 1],
                intrinsics[0, 2],
                intrinsics[1, 2],
            )
            scale_x = actual_w / orig_w
            scale_y = actual_h / orig_h
            fx_scaled = fx * scale_x
            fy_scaled = fy * scale_y
            cx_scaled = cx * scale_x
            cy_scaled = cy * scale_y

            # Step 2: Write DSO calibration file
            # DSO format: "Pinhole fx fy cx cy 0\nin_width in_height\ncrop\nout_width out_height"
            # When cx/cy > 1, DSO uses them directly in K
            calib_path = run_dir / "camera.txt"
            with open(calib_path, "w") as f:
                f.write(f"Pinhole {fx_scaled} {fy_scaled} {cx_scaled} {cy_scaled} 0\n")
                f.write(f"{actual_w} {actual_h}\n")
                f.write("crop\n")
                f.write(f"{actual_w} {actual_h}\n")

            # Step 3: Run DSO
            result_path = run_dir / "result.txt"
            cmd = [
                self._dso_binary,
                f"files={img_dir}",
                f"calib={calib_path}",
                f"preset={self._preset}",
                f"mode={self._mode}",
                f"result={result_path}",
                "quiet=1",
                "nogui=1",
                "nolog=1",
                "speed=0",
            ]

            print(f"  Running DSO: {' '.join(cmd[-6:])}")
            t_dso_start = time.time()

            env = os.environ.copy()
            # Ensure system libstdc++ is used (not conda's older version)
            # Filter out conda lib paths to avoid GLIBCXX version conflicts
            existing_ldpath = env.get('LD_LIBRARY_PATH', '')
            filtered_paths = [p for p in existing_ldpath.split(':') if p and 'conda' not in p]
            system_paths = ['/usr/lib/x86_64-linux-gnu', '/lib/x86_64-linux-gnu']
            env["LD_LIBRARY_PATH"] = ':'.join(system_paths + filtered_paths)

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=3600,  # 1 hour max
                env=env,
            )
            t_dso = time.time() - t_dso_start

            output_text = proc.stdout.decode("utf-8", errors="replace")
            if proc.returncode != 0:
                print(f"  DSO exited with code {proc.returncode}")
                for line in output_text.strip().split("\n")[-15:]:
                    print(f"    {line}")
            else:
                # Print key output lines (skip verbose re-tracking attempts)
                for line in output_text.strip().split("\n"):
                    lower = line.lower().strip()
                    if "re-track" in lower or "attempt" in lower:
                        continue
                    if any(k in lower for k in ["lost", "reset", "fps", "init",
                                                  "preset", "frames ("]):
                        print(f"    [DSO] {line.strip()}")

            # Check for LOST
            if "LOST" in output_text:
                print("  WARNING: DSO reported LOST during tracking")

            # Print timing info from DSO output
            for line in output_text.strip().split("\n"):
                if "Frames" in line and "fps" in line.lower():
                    print(f"  {line.strip()}")

            # Step 4: Parse result.txt
            positions, rotations, frame_ids_in_result = self._parse_result(
                str(result_path),
            )

            print(f"  DSO produced {len(positions)} poses for {len(frame_indices)} input frames")

            # Step 5: Map DSO output back to frame_indices
            # DSO uses frame indices as timestamps, so we map them
            all_positions, all_rotations, timings = self._map_results_to_frames(
                positions,
                rotations,
                frame_ids_in_result,
                frame_indices,
                t_dso,
            )

            return all_positions, all_rotations, timings

        finally:
            # Clean up temp directory
            import shutil
            shutil.rmtree(str(run_dir), ignore_errors=True)

    def _extract_frames(
        self,
        video_path: str,
        frame_indices: List[int],
        output_dir: str,
    ) -> Tuple[int, int]:
        """Extract frames from video to numbered image files.

        Returns (original_width, original_height)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Determine resize factor
        resize_factor = 1.0
        if self._max_image_dim is not None:
            max_dim = max(orig_w, orig_h)
            if max_dim > self._max_image_dim:
                resize_factor = self._max_image_dim / max_dim

        # Sort frame indices for sequential reading (we use sequential reading
        # since cv2 seeking can be unreliable with H.264 B-frames)
        sorted_indices = sorted(set(frame_indices))
        index_set = set(sorted_indices)

        current_frame = 0
        for fi in sorted_indices:
            # Seek to the right frame
            if fi != current_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)

            ret, frame = cap.read()
            if not ret:
                # Create a black frame as placeholder
                h = int(orig_h * resize_factor) if resize_factor != 1.0 else orig_h
                w = int(orig_w * resize_factor) if resize_factor != 1.0 else orig_w
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                if resize_factor != 1.0:
                    new_w = int(orig_w * resize_factor)
                    new_h = int(orig_h * resize_factor)
                    frame = cv2.resize(frame, (new_w, new_h))

            # DSO reads images sorted alphabetically, so zero-pad names
            img_path = os.path.join(output_dir, f"{fi:06d}.jpg")
            cv2.imwrite(img_path, frame)
            current_frame = fi + 1

        cap.release()

        # Write times.txt so DSO records frame indices as timestamps.
        # Without this, DSO sets all timestamps to 0.0 and result.txt
        # becomes unmappable.
        # DSO looks for times.txt at path.substr(0, path.find_last_of('/'))
        # where path is the images directory, so times.txt goes in the parent.
        parent_dir = str(Path(output_dir).parent)
        times_path = os.path.join(parent_dir, "times.txt")
        with open(times_path, "w") as f:
            for seq_idx, fi in enumerate(sorted_indices):
                # Format: id timestamp [exposure]
                # Use frame index as timestamp so result.txt contains frame IDs
                f.write(f"{seq_idx} {fi}\n")

        return orig_w, orig_h

    @staticmethod
    def _parse_result(result_path: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Parse DSO result.txt in TUM format.

        Format: timestamp x y z qx qy qz qw (camera-to-world)

        Returns
        -------
        positions : (N, 3) array
        rotations : (N, 3, 3) array — C2W rotation matrices
        frame_ids : list of int — frame indices (from timestamps)"""
        if not Path(result_path).exists():
            print(f"  WARNING: result.txt not found at {result_path}")
            return np.zeros((0, 3)), np.zeros((0, 3, 3)), []

        data = np.loadtxt(result_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if len(data) == 0 or data.size == 0:
            return np.zeros((0, 3)), np.zeros((0, 3, 3)), []

        frame_ids = [int(round(d)) for d in data[:, 0]]
        positions = data[:, 1:4]  # x, y, z

        # Quaternions: qx, qy, qz, qw — scipy format
        quats_xyzw = data[:, 4:8]  # [qx, qy, qz, qw]
        rotations = Rotation.from_quat(quats_xyzw).as_matrix()

        return positions, rotations, frame_ids

    @staticmethod
    def _map_results_to_frames(
        dso_positions: np.ndarray,
        dso_rotations: np.ndarray,
        dso_frame_ids: List[int],
        requested_frames: List[int],
        total_time: float,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Map DSO results (which may be sparse) to the requested frame list.

        DSO may skip frames or lose track, so not every requested frame
        has a pose.  For missing frames, we use the nearest available pose."""
        n = len(requested_frames)
        if len(dso_positions) == 0:
            return np.zeros((n, 3)), np.tile(np.eye(3), (n, 1, 1)), [0.0] * n

        # Build lookup: frame_id -> index in DSO output
        dso_lookup = {fid: i for i, fid in enumerate(dso_frame_ids)}

        positions = np.zeros((n, 3))
        rotations = np.tile(np.eye(3), (n, 1, 1))
        per_frame_time = total_time / max(n, 1)
        timings = [per_frame_time] * n

        last_known_pos = dso_positions[0]
        last_known_rot = dso_rotations[0]

        for i, fi in enumerate(requested_frames):
            if fi in dso_lookup:
                idx = dso_lookup[fi]
                positions[i] = dso_positions[idx]
                rotations[i] = dso_rotations[idx]
                last_known_pos = dso_positions[idx]
                last_known_rot = dso_rotations[idx]
            else:
                # Use last known pose (DSO may have lost/skipped this frame)
                positions[i] = last_known_pos
                rotations[i] = last_known_rot

        return positions, rotations, timings
