"""
Visual Odometry / SLAM baseline wrappers for OrthoTrack evaluation.

VO systems produce trajectories in an *arbitrary local frame* -- they
have no notion of absolute UTM coordinates.  To make them comparable
to OrthoTrack's metric evaluation, we:

1.  Run the VO system on the input video to obtain local poses.
2.  Evaluate with **all alignment modes** (same protocol as the
    foundation model comparison in scripts/evaluate_foundation_comparison.py):
    - first_frame:       anchor at first GT frame, no scale (metric drift)
    - first_frame_scale: anchor at first GT frame + LS scale (non-metric drift)
    - ate_sim3:          global Sim(3) Umeyama (non-metric trajectory shape)
3.  Report metrics for each alignment mode so users/paper can compare fairly.

Available wrappers:
    - FivePointVO       -- pure-OpenCV classical monocular VO (always works)
    - DROIDSLAMWrapper  -- DROID-SLAM (needs thirdparty/DROID-SLAM)
    - DPVOWrapper       -- DPVO (needs thirdparty/DPVO)"""

import time
import warnings
import json
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from utils.pose import CameraPose, CSVPoseLoader
from utils.alignment import align_trajectories, align_first_frame


# ===================================================================== #
#  Image directory capture (drop-in replacement for cv2.VideoCapture)   #
# ===================================================================== #

class ImageDirCapture:
    """cv2.VideoCapture-compatible wrapper that reads frames from an image directory.

    Supports the subset of cv2.VideoCapture API used by the VO wrappers:
    ``isOpened()``, ``get(prop)``, ``set(prop, value)``, ``read()``, ``release()``.

    Images are expected as ``{dir}/{NNNNNN}.{ext}`` where NNNNNN is the
    zero-padded frame index.  Common extensions: jpg, png, jpeg, bmp."""

    def __init__(self, image_dir: str):
        import cv2
        self._cv2 = cv2
        self._dir = Path(image_dir)
        if not self._dir.is_dir():
            self._files = {}
            return

        # Build mapping: frame_index -> filepath
        self._files = {}
        for p in sorted(self._dir.iterdir()):
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
                try:
                    idx = int(p.stem)
                    self._files[idx] = p
                except ValueError:
                    pass

        self._pos = 0  # current frame position
        self._w = 0
        self._h = 0

        # Read first image to get dimensions
        if self._files:
            first_key = min(self._files.keys())
            img = cv2.imread(str(self._files[first_key]))
            if img is not None:
                self._h, self._w = img.shape[:2]

    def isOpened(self) -> bool:
        return len(self._files) > 0

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        elif prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(max(self._files.keys()) + 1) if self._files else 0.0
        elif prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._pos)
        return 0.0

    def set(self, prop, value):
        import cv2
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._pos = int(value)
            return True
        return False

    def read(self):
        if self._pos in self._files:
            img = self._cv2.imread(str(self._files[self._pos]))
            self._pos += 1
            return (img is not None), img
        self._pos += 1
        return False, None

    def release(self):
        self._files = {}


def open_frame_source(path: str) -> "ImageDirCapture | cv2.VideoCapture":
    """Open a video file or image directory as a frame source.

    Returns either a cv2.VideoCapture or an ImageDirCapture depending on
    whether *path* points to a file or a directory."""
    import cv2
    p = Path(path)
    if p.is_dir():
        cap = ImageDirCapture(path)
    else:
        cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open frame source: {path}")
    return cap


# ===================================================================== #
#  Metric computation helpers                                           #
# ===================================================================== #

# Thresholds matching scripts/evaluate_foundation_comparison.py
POSITION_THRESHOLDS = [0.5, 1.0, 2.0, 5.0, 10.0]   # metres
ROTATION_THRESHOLDS = [1.0, 2.0, 5.0, 10.0]          # degrees
COMBINED_THRESHOLDS = [(1, 1), (2, 2), (5, 5), (10, 10)]  # (m, deg)


def _rotation_errors(R_pred: np.ndarray, R_gt: np.ndarray) -> np.ndarray:
    """Per-frame rotation error in degrees between (N, 3, 3) arrays."""
    errors = []
    for i in range(len(R_pred)):
        R_diff = R_pred[i] @ R_gt[i].T
        tr = np.clip(np.trace(R_diff), -1.0, 3.0)
        angle = np.arccos(np.clip((tr - 1) / 2, -1.0, 1.0))
        errors.append(np.degrees(angle))
    return np.array(errors)


def _metrics_from_alignment(
    alignment: dict,
    rotations_gt: Optional[np.ndarray],
    method_name: str,
    alignment_name: str,
) -> dict:
    """Build the standard metrics dict from a completed alignment result.

    Matches the format produced by scripts/evaluate_foundation_comparison.py."""
    pos_errors = alignment['position_errors']
    n = len(pos_errors)
    metrics = {
        'method': method_name,
        'alignment': alignment_name,
        'n_frames': n,
        'alignment_scale': float(alignment['s_align']),
        'pos_mean': float(np.mean(pos_errors)),
        'pos_median': float(np.median(pos_errors)),
        'pos_std': float(np.std(pos_errors)),
        'pos_max': float(np.max(pos_errors)),
        'pos_rmse': float(np.sqrt(np.mean(pos_errors ** 2))),
    }
    for t in POSITION_THRESHOLDS:
        metrics[f'recall_{t}m'] = float(np.mean(pos_errors < t)) * 100

    rot_aligned = alignment.get('rotations_aligned')
    if rot_aligned is not None and rotations_gt is not None:
        rot_errors = _rotation_errors(rot_aligned, rotations_gt)
        metrics['rot_mean'] = float(np.mean(rot_errors))
        metrics['rot_median'] = float(np.median(rot_errors))
        metrics['rot_std'] = float(np.std(rot_errors))
        metrics['rot_max'] = float(np.max(rot_errors))
        for t in ROTATION_THRESHOLDS:
            metrics[f'recall_{t}deg'] = float(np.mean(rot_errors < t)) * 100
        for pos_t, rot_t in COMBINED_THRESHOLDS:
            ok = (pos_errors < pos_t) & (rot_errors < rot_t)
            metrics[f'recall_{pos_t}m_{rot_t}deg'] = float(np.mean(ok)) * 100
    else:
        metrics['rot_mean'] = None
        metrics['rot_median'] = None

    return metrics


# ===================================================================== #
#  Base wrapper                                                         #
# ===================================================================== #


class VOBaselineWrapper(ABC):
    """Abstract VO/SLAM baseline wrapper.

    Subclasses implement ``_run_vo()`` which returns raw VO poses.
    The base class handles GT loading, alignment (all modes), and metric computation."""

    name: str = "vo_base"

    @abstractmethod
    def _run_vo(
        self,
        video_path: str,
        intrinsics: np.ndarray,
        frame_indices: List[int],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[float]]:
        """Run the VO system on the given video.

        Returns
        -------
        positions : (N, 3) -- camera centres in local VO frame.
        rotations : (N, 3, 3) or None -- camera-to-world rotations (optional).
        timings   : list[float] -- per-frame processing time in seconds."""
        ...

    def run_sequence(
        self,
        video_path: str,
        intrinsics_path: str,
        gt_poses_path: str,
        frame_indices: List[int],
        output_dir: Optional[str] = None,
    ) -> dict:
        """Run VO, evaluate with all alignment modes, return multi-mode metrics.

        Uses the same alignment protocol as the foundation model evaluation
        (scripts/evaluate_foundation_comparison.py):
        - first_frame:       anchor rotation from first GT frame, s=1 (metric drift)
        - first_frame_scale: anchor rotation + LS scale (non-metric drift)
        - ate_sim3:          global Sim(3) Umeyama alignment (non-metric shape)

        Returns
        -------
        dict with structure:
            {
                'method': str,
                'n_frames': int,
                'fps': float,
                'total_time': float,
                'first_frame': {pos_rmse, pos_median, rot_median, ...},
                'first_frame_scale': {pos_rmse, pos_median, rot_median, ...},
                'ate_sim3': {pos_rmse, pos_median, rot_median, ...},
            }"""
        # Load GT poses
        gt_reader = CSVPoseLoader(gt_poses_path)
        gt_by_fid = {p.frame_id: p for p in gt_reader.poses}

        # Load intrinsics
        intrinsics = self._load_intrinsics(intrinsics_path)

        # Run VO
        print(f"[{self.name}] Running on {len(frame_indices)} frames ...")
        self._processing_time = None  # reset; subclass may set this
        t0 = time.time()
        vo_positions, vo_rotations, timings = self._run_vo(
            video_path, intrinsics, frame_indices,
        )
        total_time = time.time() - t0
        # Use method-reported processing time if available.
        # This lets wrappers exclude preprocessing overhead (e.g. frame
        # extraction to disk for ORB-SLAM3) from the FPS calculation.
        processing_time = self._processing_time or total_time
        fps = len(frame_indices) / max(processing_time, 0.01)
        print(f"[{self.name}] Finished in {total_time:.1f}s ({fps:.1f} FPS)")

        # Gather GT data
        gt_positions = []
        gt_rotations = []
        valid_mask = []
        for fi in frame_indices:
            if fi in gt_by_fid:
                pose = gt_by_fid[fi]
                gt_positions.append([pose.x, pose.y, pose.z])
                gt_rotations.append(pose.rotation_matrix)  # C2W
                valid_mask.append(True)
            else:
                gt_positions.append([0.0, 0.0, 0.0])
                gt_rotations.append(np.eye(3))
                valid_mask.append(False)

        gt_positions = np.array(gt_positions)
        gt_rotations = np.array(gt_rotations)
        valid_mask = np.array(valid_mask)
        n_valid = int(valid_mask.sum())

        # --- Evaluate with all alignment modes ---
        result = {
            'method': self.name,
            'n_frames': len(frame_indices),
            'n_valid': n_valid,
            'fps': float(fps),
            'total_time': float(total_time),
        }

        if n_valid >= 3:
            vo_pos_v = vo_positions[valid_mask]
            vo_rot_v = vo_rotations[valid_mask] if vo_rotations is not None else None
            gt_pos_v = gt_positions[valid_mask]
            gt_rot_v = gt_rotations[valid_mask]

            # Mode 1: First-frame anchor, no scale (metric drift)
            a1 = align_first_frame(
                vo_pos_v, gt_pos_v,
                rotations_pred=vo_rot_v, rotations_gt=gt_rot_v,
                with_scale=False,
            )
            result['first_frame'] = _metrics_from_alignment(
                a1, gt_rot_v, self.name, 'first_frame')

            # Mode 2: First-frame anchor + LS scale (non-metric drift)
            a2 = align_first_frame(
                vo_pos_v, gt_pos_v,
                rotations_pred=vo_rot_v, rotations_gt=gt_rot_v,
                with_scale=True,
            )
            result['first_frame_scale'] = _metrics_from_alignment(
                a2, gt_rot_v, self.name, 'first_frame_scale')

            # Mode 3: Global Sim(3) Umeyama (non-metric shape)
            a3 = align_trajectories(
                vo_pos_v, gt_pos_v,
                rotations_pred=vo_rot_v,
                with_scale=True,
            )
            result['ate_sim3'] = _metrics_from_alignment(
                a3, gt_rot_v, self.name, 'ate_sim3')

            # Print summary for each mode
            for mode_key in ('first_frame', 'first_frame_scale', 'ate_sim3'):
                m = result[mode_key]
                rot_str = f", RE(med): {m['rot_median']:.2f}deg" if m.get('rot_median') is not None else ""
                print(f"  [{mode_key}] scale={m['alignment_scale']:.4f}, "
                      f"RMSE={m['pos_rmse']:.2f}m, TE(med)={m['pos_median']:.2f}m{rot_str}")
        else:
            warnings.warn(f"[{self.name}] Too few valid GT poses ({n_valid}) for alignment!")
            a1 = a2 = a3 = None

        # Save results
        if output_dir is not None:
            self._save_results(output_dir, result, vo_positions, vo_rotations,
                               gt_positions, gt_rotations, valid_mask, frame_indices,
                               alignments={'first_frame': a1,
                                           'first_frame_scale': a2,
                                           'ate_sim3': a3})

        return result

    def _save_results(self, output_dir, result, vo_positions, vo_rotations,
                      gt_positions, gt_rotations, valid_mask, frame_indices,
                      alignments=None):
        """Save predictions and multi-mode summary."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save raw VO predictions (same format as foundation models: predictions.npz)
        save_dict = {
            'positions': vo_positions,
            'frame_indices': np.array(frame_indices),
        }
        if vo_rotations is not None:
            save_dict['rotations'] = vo_rotations
        np.savez(out / 'predictions.npz', **save_dict)

        # Save GT for convenience
        np.savez(out / 'ground_truth.npz',
                 positions=gt_positions, rotations=gt_rotations,
                 valid_mask=valid_mask)

        # Save aligned trajectories for each alignment mode
        # These cover only the valid frames (where GT exists)
        if alignments:
            valid_frame_indices = np.array(frame_indices)[valid_mask]
            aligned_dict = {'valid_frame_indices': valid_frame_indices}
            for mode, aln in alignments.items():
                if aln is None:
                    continue
                aligned_dict[f'{mode}_positions'] = aln['positions_aligned']
                if 'rotations_aligned' in aln:
                    aligned_dict[f'{mode}_rotations'] = aln['rotations_aligned']
                # Also store the alignment transform for reproducibility
                aligned_dict[f'{mode}_R_align'] = aln['R_align']
                aligned_dict[f'{mode}_t_align'] = aln['t_align']
                aligned_dict[f'{mode}_s_align'] = np.array(aln['s_align'])
            np.savez(out / 'aligned_trajectories.npz', **aligned_dict)

        # Save summary JSON with all alignment modes
        with open(out / "summary.json", 'w') as f:
            json.dump(result, f, indent=2)

        print(f"[{self.name}] Saved results to {out}")

    @staticmethod
    def _load_intrinsics(path: str) -> np.ndarray:
        """Load 3x3 camera intrinsics from JSON or text file."""
        path = Path(path)
        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            return np.array([
                [data["fx"], 0, data["cx"]],
                [0, data["fy"], data["cy"]],
                [0, 0, 1],
            ], dtype=np.float64)
        else:
            return np.loadtxt(path).reshape(3, 3)


# ===================================================================== #
#  Homography decomposition helpers                                     #
# ===================================================================== #

def _best_homography_decomposition(
    H: np.ndarray,
    K: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Select the best (R, t) from homography decomposition.

    Uses ``cv2.decomposeHomographyMat`` which returns up to 4 candidate
    (R, t, n) solutions. We pick the one that:
      (a) puts the plane normal n roughly towards the camera (n_z > 0), and
      (b) maximises the number of points with positive depth in both views.

    Returns (R_rel, t_rel) in the same convention as ``recoverPose``:
        p_cam2 = R_rel @ p_cam1 + t_rel
    Or (None, None) if no valid decomposition is found."""
    import cv2

    n_solutions, Rs, ts, normals = cv2.decomposeHomographyMat(H, K)

    best_R = None
    best_t = None
    best_score = -1

    K_inv = np.linalg.inv(K)
    # Normalised coordinates
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])  # (N, 3)
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])
    rays1 = (K_inv @ pts1_h.T).T  # (N, 3)
    rays2 = (K_inv @ pts2_h.T).T

    for i in range(n_solutions):
        Rc = Rs[i]
        tc = ts[i].ravel()
        nc = normals[i].ravel()

        # Plane normal should face the camera (z > 0 in cam1 frame)
        if nc[2] < 0:
            nc = -nc

        # Cheirality check: compute depths via triangulation-free approach.
        # For each point, d1 * ray1 and d2 * ray2 = R @ d1*ray1 + t
        # d1 = (t x ray2) . (ray2 x R@ray1) / ||ray2 x R@ray1||^2
        score = 0
        for j in range(min(len(rays1), 200)):  # subsample for speed
            r1 = rays1[j]
            r2 = rays2[j]
            Rr1 = Rc @ r1
            cross_r2_Rr1 = np.cross(r2, Rr1)
            denom = np.dot(cross_r2_Rr1, cross_r2_Rr1)
            if denom < 1e-10:
                continue
            d1 = np.dot(np.cross(tc, r2), cross_r2_Rr1) / denom
            d2 = np.dot(np.cross(tc, Rr1), np.cross(r2, Rr1)) / denom
            if d1 > 0 and d2 > 0:
                score += 1

        if score > best_score:
            best_score = score
            best_R = Rc
            best_t = tc

    if best_score < 5:
        return None, None

    return best_R, best_t


# ===================================================================== #
#  Classical Monocular VO (Essential + Homography, always works)        #
# ===================================================================== #


class FivePointVO(VOBaselineWrapper):
    """Classical monocular VO with Essential matrix + Homography fallback.

    On each frame pair the pipeline:
    1. Extract SIFT features and match with FLANN + ratio test.
    2. Estimate Essential matrix (5-point RANSAC) and count recoverPose inliers.
    3. Estimate Homography (RANSAC) and count inliers.
    4. Use ORB-SLAM-style model selection: if the Homography inlier ratio
       exceeds a threshold (``h_ratio_thresh``), decompose H; otherwise use E.
       This handles the **planar scene degeneracy** common in aerial/UAV imagery
       where the Essential matrix decomposition fails because the scene is flat.
    5. Chain relative poses (unit translation scale) to build trajectory.

    Scale is unobservable in monocular VO. We fix relative scale = 1
    between all successive frames and rely on Sim(3) Umeyama alignment
    to recover the global scale from ground truth.

    No external dependencies beyond OpenCV and NumPy.

    Parameters
    ----------
    max_features : int
        Max SIFT keypoints per frame.
    match_ratio : float
        Lowe's ratio test threshold.
    ransac_thresh : float
        RANSAC inlier threshold in pixels.
    h_ratio_thresh : float
        If H_inliers / (H_inliers + E_inliers) > this threshold, prefer
        Homography decomposition. Default 0.40 (ORB-SLAM uses 0.45).
    target_width : int or None
        Resize frames to this width (preserving aspect ratio). None = original."""

    name = "five_point_vo"

    def __init__(
        self,
        max_features: int = 4000,
        match_ratio: float = 0.75,
        ransac_thresh: float = 1.0,
        h_ratio_thresh: float = 0.40,
        target_width: Optional[int] = None,
    ):
        self._max_features = max_features
        self._match_ratio = match_ratio
        self._ransac_thresh = ransac_thresh
        self._h_ratio_thresh = h_ratio_thresh
        self._target_width = target_width

    def _run_vo(
        self,
        video_path: str,
        intrinsics: np.ndarray,
        frame_indices: List[int],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[float]]:
        import cv2

        cap = open_frame_source(video_path)

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine resize
        if self._target_width is not None and self._target_width < orig_w:
            scale_factor = self._target_width / orig_w
            target_w = self._target_width
            target_h = int(orig_h * scale_factor)
        else:
            scale_factor = 1.0
            target_w = orig_w
            target_h = orig_h

        # Scale intrinsics
        K = intrinsics.copy()
        K[0, 0] *= scale_factor
        K[0, 2] *= scale_factor
        K[1, 1] *= scale_factor
        K[1, 2] *= scale_factor

        # Feature extractor and matcher
        sift = cv2.SIFT_create(nfeatures=self._max_features)
        FLANN_INDEX_KDTREE = 1
        flann = cv2.FlannBasedMatcher(
            dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
            dict(checks=50),
        )

        # Global pose: starts at identity (C2W convention)
        R_global = np.eye(3)
        t_global = np.zeros(3)

        positions = []
        rotations = []
        timings = []

        prev_gray = None
        prev_kp = None
        prev_des = None

        n_E_used = 0
        n_H_used = 0
        n_failed = 0

        for idx, fi in enumerate(frame_indices):
            t0 = time.time()

            # Read frame
            if fi >= total_frames:
                positions.append(positions[-1].copy() if positions else np.zeros(3))
                rotations.append(rotations[-1].copy() if rotations else np.eye(3))
                timings.append(0.0)
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                positions.append(positions[-1].copy() if positions else np.zeros(3))
                rotations.append(rotations[-1].copy() if rotations else np.eye(3))
                timings.append(0.0)
                continue

            if scale_factor != 1.0:
                frame = cv2.resize(frame, (target_w, target_h))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect features
            kp, des = sift.detectAndCompute(gray, None)

            if idx == 0 or prev_des is None or des is None or len(kp) < 10:
                # First frame or insufficient features: record identity pose
                positions.append(t_global.copy())
                rotations.append(R_global.copy())
                timings.append(time.time() - t0)
                prev_gray = gray
                prev_kp = kp
                prev_des = des
                continue

            if len(prev_kp) < 10:
                positions.append(positions[-1].copy())
                rotations.append(rotations[-1].copy())
                timings.append(time.time() - t0)
                prev_gray = gray
                prev_kp = kp
                prev_des = des
                continue

            # Match features
            try:
                raw_matches = flann.knnMatch(prev_des, des, k=2)
            except cv2.error:
                positions.append(positions[-1].copy())
                rotations.append(rotations[-1].copy())
                timings.append(time.time() - t0)
                prev_gray = gray
                prev_kp = kp
                prev_des = des
                continue

            # Lowe's ratio test
            good_matches = []
            for m_pair in raw_matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < self._match_ratio * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 10:
                positions.append(positions[-1].copy())
                rotations.append(rotations[-1].copy())
                timings.append(time.time() - t0)
                prev_gray = gray
                prev_kp = kp
                prev_des = des
                continue

            # Get matched point coordinates
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good_matches])
            pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])

            # --- Model selection (ORB-SLAM style) ---
            # Estimate Essential matrix
            E, mask_E = cv2.findEssentialMat(
                pts1, pts2, K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=self._ransac_thresh,
            )
            n_E_inliers_raw = int(mask_E.sum()) if mask_E is not None else 0

            # Estimate Homography
            H, mask_H = cv2.findHomography(
                pts1, pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=self._ransac_thresh,
            )
            n_H_inliers = int(mask_H.sum()) if mask_H is not None else 0

            # Try Essential matrix decomposition first
            R_rel = None
            t_rel = None
            used_model = None

            if E is not None and mask_E is not None:
                n_E_pose, R_E, t_E, _ = cv2.recoverPose(
                    E, pts1, pts2, K, mask=mask_E.copy(),
                )
                if n_E_pose >= 10:
                    R_rel = R_E
                    t_rel = t_E.ravel()
                    used_model = 'E'

            # Decide whether to prefer Homography based on inlier ratio
            denom = max(n_H_inliers + n_E_inliers_raw, 1)
            h_ratio = n_H_inliers / denom

            # Use Homography if: (a) E decomposition failed, or
            # (b) H inlier ratio exceeds threshold (planar scene)
            if R_rel is None or h_ratio > self._h_ratio_thresh:
                if H is not None:
                    R_H, t_H = _best_homography_decomposition(
                        H, K, pts1, pts2,
                    )
                    if R_H is not None:
                        R_rel = R_H
                        t_rel = t_H
                        # Normalise translation to unit length
                        tn = np.linalg.norm(t_rel)
                        if tn > 1e-8:
                            t_rel = t_rel / tn
                        used_model = 'H'

            if R_rel is None:
                # Both models failed
                positions.append(positions[-1].copy())
                rotations.append(rotations[-1].copy())
                timings.append(time.time() - t0)
                prev_gray = gray
                prev_kp = kp
                prev_des = des
                n_failed += 1
                continue

            if used_model == 'E':
                n_E_used += 1
            else:
                n_H_used += 1

            # Chain relative motion using C2W accumulation.
            # recoverPose / decomposeHomography give [R|t] such that
            # camera 2 = [R|t] relative to camera 1 = [I|0].
            # That is: p_cam2 = R @ p_cam1 + t.
            #
            # Camera center in cam1 frame: C2_in_cam1 = -R^T @ t
            # Camera center in world: C2 = R_c2w_old @ C2_in_cam1 + C_old
            delta_c = R_global @ (-R_rel.T @ t_rel)
            t_global = t_global + delta_c
            R_global = R_global @ R_rel.T

            positions.append(t_global.copy())
            rotations.append(R_global.copy())
            timings.append(time.time() - t0)

            prev_gray = gray
            prev_kp = kp
            prev_des = des

        cap.release()

        n_pairs = len(frame_indices) - 1
        print(f"[{self.name}] Model usage: E={n_E_used}, H={n_H_used}, "
              f"failed={n_failed} / {n_pairs} pairs")

        positions = np.array(positions)
        rotations = np.array(rotations)

        return positions, rotations, timings


# ===================================================================== #
#  DROID-SLAM Wrapper                                                   #
# ===================================================================== #


class DROIDSLAMWrapper(VOBaselineWrapper):
    """Wrapper for DROID-SLAM (Teed & Deng, NeurIPS 2021).

    DROID-SLAM is a deep visual SLAM system that maintains a dense factor
    graph of optical flow correlations. It selects keyframes adaptively,
    runs frontend/backend bundle adjustment on them, then uses a trajectory
    filler to interpolate poses for non-keyframe frames.

    Install:
        cd thirdparty
        git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git
        pip install lietorch torch-scatter

    Parameters
    ----------
    device : str -- 'cuda' or 'cpu'.
    weights : str or None -- path to droid.pth checkpoint.
    target_pixels : int -- target total pixel count for resizing (default 384*512).
        Frames are resized so that H*W ≈ target_pixels while keeping H,W divisible by 8."""

    name = "droid_slam"

    def __init__(
        self,
        device: str = "cuda",
        weights: Optional[str] = None,
        target_pixels: int = 384 * 512,
        buffer: int = 1024,
    ):
        self._device = device
        thirdparty = Path(__file__).resolve().parent.parent.parent / "thirdparty"
        self._weights = weights or str(thirdparty / "DROID-SLAM" / "droid.pth")
        self._target_pixels = target_pixels
        self._buffer = buffer

    def _run_vo(
        self,
        video_path: str,
        intrinsics: np.ndarray,
        frame_indices: List[int],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[float]]:
        import torch
        import cv2
        from scipy.spatial.transform import Rotation

        import sys
        droid_path = str(Path(__file__).resolve().parent.parent.parent / "thirdparty" / "DROID-SLAM")
        droid_slam_path = str(Path(droid_path) / "droid_slam")
        for p in [droid_path, droid_slam_path]:
            if p not in sys.path:
                sys.path.insert(0, p)

        try:
            from droid import Droid
        except ImportError:
            raise ImportError(
                "DROID-SLAM not installed. Install with:\n"
                "  cd thirdparty && git clone --recursive "
                "https://github.com/princeton-vl/DROID-SLAM.git\n"
                "  pip install lietorch torch-scatter"
            )

        cap = open_frame_source(video_path)

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Compute target resolution (matches demo.py logic)
        scale = np.sqrt(self._target_pixels / (orig_h * orig_w))
        target_h = int(orig_h * scale)
        target_w = int(orig_w * scale)
        # Make divisible by 8
        target_h = target_h - target_h % 8
        target_w = target_w - target_w % 8

        sx = target_w / orig_w
        sy = target_h / orig_h
        calib = torch.as_tensor([fx * sx, fy * sy, cx * sx, cy * sy],
                                dtype=torch.float32)

        from argparse import Namespace
        args = Namespace(
            image_size=[target_h, target_w],
            stereo=False,
            disable_vis=True,
            beta=0.3,
            filter_thresh=2.4,
            warmup=8,
            buffer=self._buffer,
            keyframe_thresh=3.5,
            frontend_thresh=16.0,
            frontend_window=25,
            frontend_radius=2,
            frontend_nms=1,
            backend_thresh=22.0,
            backend_radius=2,
            backend_nms=3,
            upsample=False,
            weights=self._weights,
        )

        droid = Droid(args)
        timings = []

        # We need to feed all frames to track() AND build an image_stream
        # for terminate() to fill non-keyframe poses.
        # Store frames for the trajectory filler.
        stored_frames = []

        for i, fi in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                timings.append(0.0)
                continue

            frame = cv2.resize(frame, (target_w, target_h))
            frame_tensor = torch.as_tensor(frame).permute(2, 0, 1)  # (3, H, W) uint8

            # Store for trajectory filler stream
            stored_frames.append((fi, frame_tensor, calib.clone()))

            t0 = time.time()
            droid.track(fi, frame_tensor[None], intrinsics=calib)
            timings.append(time.time() - t0)

        # Build image stream generator for trajectory filler (fills non-keyframe poses)
        def image_stream():
            for (tstamp, img_tensor, intr) in stored_frames:
                yield tstamp, img_tensor[None], intr

        traj = droid.terminate(stream=image_stream())  # (N, 7) [tx ty tz qx qy qz qw]
        cap.release()
        del droid
        torch.cuda.empty_cache()

        # Convert SE3 representation to positions + rotation matrices
        # DROID-SLAM returns camera_trajectory.inv().data => C2W poses as lietorch SE3
        # SE3 data: [tx, ty, tz, qx, qy, qz, qw]
        positions = []
        rot_matrices = []

        n_output = min(len(traj), len(frame_indices))
        for i in range(n_output):
            t_vec = traj[i, :3]
            q = traj[i, 3:]  # [qx, qy, qz, qw] -- scipy convention
            R_mat = Rotation.from_quat(q).as_matrix()  # C2W rotation
            positions.append(t_vec)
            rot_matrices.append(R_mat)

        # Pad if trajectory is shorter than frame_indices (shouldn't happen with traj_filler)
        while len(positions) < len(frame_indices):
            positions.append(positions[-1] if positions else np.zeros(3))
            rot_matrices.append(rot_matrices[-1] if rot_matrices else np.eye(3))
            timings.append(0.0)

        timings = timings[:len(frame_indices)]
        while len(timings) < len(frame_indices):
            timings.append(0.0)

        return np.array(positions), np.array(rot_matrices), timings


# ===================================================================== #
#  DPVO / DPV-SLAM Wrapper                                              #
# ===================================================================== #


class DPVOWrapper(VOBaselineWrapper):
    """Wrapper for DPVO (Teed et al., CVPR 2023) / DPV-SLAM (Lipson et al., ECCV 2024).

    DPVO is a deep patch visual odometry system that tracks sparse patches
    and maintains a factor graph of correlations for bundle adjustment.
    With loop closure enabled, it becomes DPV-SLAM.

    Install:
        cd thirdparty && git clone --recursive https://github.com/princeton-vl/DPVO.git
        cd DPVO && python setup.py build_ext --inplace
        pip install numba pypose
        # Download: wget https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip && unzip models.zip

    Parameters
    ----------
    device : str -- 'cuda' or 'cpu'.
    weights : str or None -- path to DPVO checkpoint (dpvo.pth).
    config : str or None -- path to DPVO config yaml.
    use_loop_closure : bool -- enable loop closure (DPV-SLAM mode).
    buffer_size : int -- DPVO internal buffer size (increase for longer sequences).
    target_pixels : int -- target total pixel count for image resizing (default 320*512)."""

    name = "dpvo"

    def __init__(
        self,
        device: str = "cuda",
        weights: Optional[str] = None,
        config: Optional[str] = None,
        use_loop_closure: bool = False,
        buffer_size: int = 4096,
        target_pixels: int = 320 * 512,
    ):
        self._device = device
        dpvo_dir = Path(__file__).resolve().parent.parent.parent / "thirdparty" / "DPVO"
        self._weights = weights or str(dpvo_dir / "dpvo.pth")
        self._config = config or str(dpvo_dir / "config" / "default.yaml")
        self._use_loop_closure = use_loop_closure
        self._buffer_size = buffer_size
        self._target_pixels = target_pixels
        if use_loop_closure:
            self.name = "dpv_slam"

    def _run_vo(
        self,
        video_path: str,
        intrinsics: np.ndarray,
        frame_indices: List[int],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[float]]:
        import torch
        import cv2
        from scipy.spatial.transform import Rotation

        import sys
        dpvo_path = str(Path(__file__).resolve().parent.parent.parent / "thirdparty" / "DPVO")
        if dpvo_path not in sys.path:
            sys.path.insert(0, dpvo_path)

        try:
            from dpvo.dpvo import DPVO
            from dpvo.config import cfg as dpvo_cfg
        except ImportError:
            raise ImportError(
                "DPVO not installed. Install with:\n"
                "  cd thirdparty && git clone --recursive "
                "https://github.com/princeton-vl/DPVO.git\n"
                "  cd DPVO && python setup.py build_ext --inplace\n"
                "  pip install numba pypose"
            )

        dpvo_cfg.merge_from_file(self._config)
        # Increase buffer for longer sequences
        dpvo_cfg.BUFFER_SIZE = self._buffer_size
        if self._use_loop_closure:
            dpvo_cfg.LOOP_CLOSURE = True

        cap = open_frame_source(video_path)

        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Resize to keep memory manageable (DPVO scales ~O(H*W) per keyframe).
        # Target ~320*512 ≈ 163840 px, matching DPVO evaluation benchmarks.
        target_pixels = self._target_pixels
        scale = np.sqrt(target_pixels / (orig_h * orig_w))
        if scale < 1.0:
            target_h = int(round(orig_h * scale / 8)) * 8
            target_w = int(round(orig_w * scale / 8)) * 8
        else:
            target_h, target_w = orig_h, orig_w
        sx, sy = target_w / orig_w, target_h / orig_h
        fx_s, fy_s = fx * sx, fy * sy
        cx_s, cy_s = cx * sx, cy * sy
        self._resize = (target_w, target_h) if scale < 1.0 else None

        # DPVO expects intrinsics as a CUDA tensor [fx, fy, cx, cy]
        calib = torch.as_tensor([fx_s, fy_s, cx_s, cy_s], dtype=torch.float32).cuda()

        slam = DPVO(dpvo_cfg, self._weights, ht=target_h, wd=target_w)
        timings = []

        for i, fi in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                timings.append(0.0)
                continue

            # DPVO expects (3, H, W) uint8->float CUDA tensor (BGR is fine, normalized internally)
            if self._resize is not None:
                frame = cv2.resize(frame, self._resize, interpolation=cv2.INTER_AREA)
            image = torch.from_numpy(frame).permute(2, 0, 1).cuda()

            t0 = time.time()
            try:
                # DPVO's update() doesn't use torch.no_grad() internally, causing
                # autograd graphs to accumulate and leak ~130 MB/frame. We wrap the
                # call in no_grad to prevent this.
                with torch.no_grad():
                    slam(fi, image, calib)
            except Exception as e:
                print(f"  Warning: DPVO frame {fi} failed: {e}")
            timings.append(time.time() - t0)

        cap.release()

        # terminate() returns (poses, tstamps)
        # poses: (N, 7) numpy array [tx, ty, tz, qx, qy, qz, qw] -- C2W (already inverted)
        with torch.no_grad():
            traj, tstamps = slam.terminate()
        del slam
        torch.cuda.empty_cache()

        # Build a timestamp -> index map for matching output poses to input frame_indices
        # tstamps from DPVO are the frame indices we fed in
        tstamp_to_pose = {}
        for i in range(len(tstamps)):
            t = int(tstamps[i])
            tstamp_to_pose[t] = i

        positions = []
        rot_matrices = []

        for fi in frame_indices:
            if fi in tstamp_to_pose:
                idx = tstamp_to_pose[fi]
                t_vec = traj[idx, :3]
                q = traj[idx, 3:]  # [qx, qy, qz, qw] -- scipy convention
                R_mat = Rotation.from_quat(q).as_matrix()
                positions.append(t_vec)
                rot_matrices.append(R_mat)
            else:
                # Frame was skipped by DPVO's motion filter -- use last known pose
                positions.append(positions[-1] if positions else np.zeros(3))
                rot_matrices.append(rot_matrices[-1] if rot_matrices else np.eye(3))

        timings = timings[:len(frame_indices)]
        while len(timings) < len(frame_indices):
            timings.append(0.0)

        return np.array(positions), np.array(rot_matrices), timings
