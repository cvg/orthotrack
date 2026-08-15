"""
Optical flow feature tracker for tracking 2D-3D correspondences between frames."""

import numpy as np
import cv2
from typing import Tuple


# Known ptlflow model names (checked at runtime against ptlflow.get_model_names())
_PTLFLOW_MODELS = None

def _is_ptlflow_model(name: str) -> bool:
    """Check if a flow_method name corresponds to a ptlflow model."""
    global _PTLFLOW_MODELS
    if name in ('lk', 'lk_gpu', 'waft'):
        return False
    if _PTLFLOW_MODELS is None:
        try:
            import ptlflow
            _PTLFLOW_MODELS = set(ptlflow.get_model_names())
        except Exception:
            _PTLFLOW_MODELS = set()
    return name in _PTLFLOW_MODELS


class FeatureTracker:
    """
    Tracks 2D-3D correspondences between frames using optical flow.
    Key insight: if we know 2D→3D from keyframe, track 2D points to get
    new 2D→3D correspondences without re-matching to orthophoto!

    Supports three backends:
    - 'lk': Lucas-Kanade optical flow. Automatically uses GPU (PyTorch) when
      CUDA is available, falls back to CPU (OpenCV) otherwise.
    - 'waft': WAFT learned optical flow (handles large motion ~100+px)
    - Any ptlflow model name (e.g. 'raft', 'flowformer', 'gma', 'sea_raft', ...)"""

    def __init__(self, max_features: int = 2000, flow_method: str = 'waft',
                 fb_threshold: float = 3.0,
                 filter_by_trackability: bool = False,
                 trackability_min_score: float = 0.001,
                 max_tracking_points: int = 0,
                 accumulate_points: bool = False,
                 accumulate_dedup_radius: float = 3.0):
        self.max_features = max_features
        self.flow_method = flow_method
        self.fb_threshold = fb_threshold
        self.filter_by_trackability = filter_by_trackability
        self.trackability_min_score = trackability_min_score
        self.max_tracking_points = max_tracking_points  # 0 = no cap
        self.accumulate_points = accumulate_points  # merge old tracked pts with new KF pts
        self.accumulate_dedup_radius = accumulate_dedup_radius  # px radius for dedup
        # 'lk_gpu' is accepted as an alias for 'lk'
        if flow_method == 'lk_gpu':
            flow_method = 'lk'
            self.flow_method = 'lk'

        self.use_ptlflow = _is_ptlflow_model(flow_method)

        # Auto-detect GPU availability for LK
        if flow_method == 'lk':
            import torch
            import os
            force_cpu = os.environ.get('FORCE_CPU_LK', '').lower() in ('1', 'true', 'yes')
            self.use_gpu_lk = torch.cuda.is_available() and not force_cpu
            # Adaptive max_level: set during first set_keyframe based on image size
            self._lk_max_level = 3  # base default, may be increased for large images
            self._lk_win_size = 21
            if self.use_gpu_lk:
                # GPU-accelerated Lucas-Kanade (lazy init)
                self.lk_params = None
                self.gpu_lk = None  # Will be initialized on first call
            else:
                # CPU Lucas-Kanade via OpenCV
                self.lk_params = dict(
                    winSize=(self._lk_win_size, self._lk_win_size),
                    maxLevel=self._lk_max_level,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                )
                self.gpu_lk = None
            self.waft = None
            self.ptlflow_model = None
        elif flow_method == 'waft':
            # WAFT dense optical flow (lazy init on first use)
            self.lk_params = None
            self.waft = None  # Will be initialized on first call
            self.ptlflow_model = None
        elif self.use_ptlflow:
            # ptlflow model (lazy init on first use)
            self.lk_params = None
            self.waft = None
            self.ptlflow_model = None  # Will be initialized on first call
            self.gpu_lk = None
        else:
            raise ValueError(f"Unknown flow_method: {flow_method}. Use 'lk', 'waft', or a ptlflow model name.")

        # Current state
        self.prev_image = None  # RGB image for WAFT, grayscale for LK
        self.tracked_pts_2d = None  # (N, 2) current 2D positions
        self.tracked_pts_3d = None  # (N, 3) corresponding 3D world positions
        self.tracked_confs = None  # (N,) match confidences from keyframe
        self.initial_num_pts = 0
        self.keyframe_id = -1
        # Motion-adaptive tracking stats (updated each frame)
        self.last_median_displacement = 0.0  # px, median fwd displacement
        self.last_survival_rate = 1.0  # fraction of points surviving tracking

        # Keyframe data for direct keyframe-to-current tracking (WAFT only)
        self.keyframe_image = None  # RGB keyframe image
        self.keyframe_pts_2d = None  # (N, 2) original keyframe 2D positions
        self.keyframe_pts_3d = None  # (N, 3) original keyframe 3D positions
        self.keyframe_confs = None  # (N,) original keyframe confidences

    def _get_gpu_lk(self):
        """Lazy-initialize GPU LK on first use.
        
        Auto-selects the best backend: OpenCV CUDA if available, else PyTorch.
        Uses adaptive max_level computed from image dimensions."""
        if self.gpu_lk is None:
            from orthotrack.optical_flow.gpu_lk import create_gpu_lk
            self.gpu_lk = create_gpu_lk(
                win_size=self._lk_win_size, max_level=self._lk_max_level,
                max_iter=10, eps=0.01
            )
        return self.gpu_lk

    def _adapt_lk_levels(self, image_shape: tuple):
        """Adapt LK pyramid max_level based on image size for the first keyframe.

        For images larger than ~1920px, more pyramid levels are needed to handle
        larger pixel motions. The search range at level L with window W is
        approximately (W/2) * 2^L pixels."""
        import math
        h, w = image_shape[:2]
        max_dim = max(h, w)
        # Compute required levels: target search range = max_dim * 0.10
        # (handle ~10% of image dimension per frame; some UAV sequences like
        # UAVD4L inTraj have inter-frame motion of ~189px at 1920px resolution)
        target_range = max(84.0, max_dim * 0.10)
        half_win = self._lk_win_size / 2.0
        needed_level = max(3, math.ceil(math.log2(target_range / half_win)))
        needed_level = min(needed_level, 7)  # cap at 7 to avoid excessive memory

        if needed_level != self._lk_max_level:
            old_level = self._lk_max_level
            self._lk_max_level = needed_level
            # Force re-init of GPU LK with new level 
            if self.gpu_lk is not None:
                self.gpu_lk = None
            # Update CPU LK params too
            if self.lk_params is not None:
                self.lk_params['maxLevel'] = needed_level
            old_range = int(half_win * (2 ** old_level))
            new_range = int(half_win * (2 ** needed_level))
            print(f"  [LK] Adapted max_level: {old_level}->{needed_level} "
                  f"for {w}x{h} images (search range {old_range}->{new_range}px)")

    def _effective_lk_level(self) -> int:
        """Reduce pyramid levels when inter-frame motion is small.

        The base _lk_max_level is set once per sequence by _adapt_lk_levels().
        This method further lowers it per-frame based on last_median_displacement
        to skip unnecessary coarse levels and save compute."""
        if self.last_median_displacement <= 0:
            return self._lk_max_level
        half_win = self._lk_win_size / 2.0
        # Need search_range >= 4 * median_displacement for safety margin
        import math
        needed = max(2, math.ceil(math.log2(
            max(1.0, self.last_median_displacement * 4.0) / half_win)))
        return min(needed, self._lk_max_level)

    def _get_waft(self):
        """Lazy-initialize WAFT model on first use."""
        if self.waft is None:
            from orthotrack.optical_flow.waft_flow import WAFTOpticalFlow
            self.waft = WAFTOpticalFlow()
        return self.waft

    def _get_ptlflow(self):
        """Lazy-initialize ptlflow model on first use."""
        if self.ptlflow_model is None:
            from orthotrack.optical_flow.ptlflow_flow import PTLFlowOpticalFlow
            self.ptlflow_model = PTLFlowOpticalFlow(model_name=self.flow_method)
        return self.ptlflow_model

    def set_keyframe(self, frame_id: int, image: np.ndarray,
                     pts_2d: np.ndarray, pts_3d: np.ndarray,
                     confs: np.ndarray = None):
        """
        Set a new keyframe with known 2D-3D correspondences.

        If accumulate_points is enabled, merges surviving tracked points
        from the previous keyframe with the new keyframe's matches
        (after spatial deduplication).

        Args:
            frame_id: Keyframe ID
            image: Keyframe image (RGB)
            pts_2d: (N, 2) 2D image coordinates
            pts_3d: (N, 3) 3D world coordinates (UTM)
            confs: (N,) match confidences (optional)"""
        # --- Accumulate: merge old tracked points with new matches ---
        if (self.accumulate_points
                and self.tracked_pts_2d is not None
                and len(self.tracked_pts_2d) > 0):
            old_2d = self.tracked_pts_2d.reshape(-1, 2)
            old_3d = self.tracked_pts_3d
            old_confs = self.tracked_confs

            # Deduplicate: remove old points that are spatially close to new ones
            if len(pts_2d) > 0 and len(old_2d) > 0:
                # For each old point, find min distance to any new point
                # Use efficient broadcasting for moderate point counts
                diffs = old_2d[:, None, :] - pts_2d[None, :, :]  # (M, N, 2)
                dists = np.linalg.norm(diffs, axis=2).min(axis=1)  # (M,)
                keep_old = dists > self.accumulate_dedup_radius
            else:
                keep_old = np.ones(len(old_2d), dtype=bool)

            n_old_kept = keep_old.sum()
            if n_old_kept > 0:
                # Concatenate: new points first (higher priority), then old survivors
                pts_2d = np.concatenate([pts_2d, old_2d[keep_old]], axis=0)
                pts_3d = np.concatenate([pts_3d, old_3d[keep_old]], axis=0)
                if confs is not None:
                    # Slightly reduce confidence of carried-over points
                    old_kept_confs = old_confs[keep_old] * 0.8
                    confs = np.concatenate([confs, old_kept_confs], axis=0)

                print(f"  [Accumulate] Merged {n_old_kept} old tracked pts + {len(pts_2d) - n_old_kept} new → {len(pts_2d)} total")

        self.keyframe_id = frame_id
        if self.flow_method == 'lk':
            # Adapt pyramid levels for image resolution (first keyframe only)
            if self.keyframe_id == frame_id and not hasattr(self, '_lk_adapted'):
                self._adapt_lk_levels(image.shape)
                self._lk_adapted = True
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if self.use_gpu_lk:
                import torch
                self.prev_image = torch.from_numpy(gray.astype(np.float32)).cuda()
            else:
                self.prev_image = gray
        else:
            gray = None
            self.prev_image = image.copy()

        # --- Shi-Tomasi trackability filter (LK / LK-GPU only) ---
        if (self.filter_by_trackability
                and self.flow_method == 'lk'
                and gray is not None):
            from utils.image import filter_by_shi_tomasi
            keep = filter_by_shi_tomasi(
                pts_2d, gray, min_score=self.trackability_min_score)
            n_before = len(pts_2d)
            pts_2d = pts_2d[keep]
            pts_3d = pts_3d[keep]
            if confs is not None:
                confs = confs[keep]
            n_after = len(pts_2d)
            if n_before > n_after:
                print(f"  [Shi-Tomasi] Filtered {n_before - n_after}/{n_before} "
                      f"low-trackability points (kept {n_after})")

        # --- Cap to max_tracking_points (top by confidence) ---
        if self.max_tracking_points > 0 and len(pts_2d) > self.max_tracking_points:
            n_before = len(pts_2d)
            if confs is not None:
                top_idx = np.argsort(confs)[::-1][:self.max_tracking_points]
            else:
                # Random subsample if no confidences
                top_idx = np.random.choice(len(pts_2d), self.max_tracking_points, replace=False)
            pts_2d = pts_2d[top_idx]
            pts_3d = pts_3d[top_idx]
            if confs is not None:
                confs = confs[top_idx]
            print(f"  [MaxPts] Capped tracking points: {n_before} -> {self.max_tracking_points}")

        self.tracked_pts_2d = pts_2d.astype(np.float32).reshape(-1, 1, 2)
        self.tracked_pts_3d = pts_3d.copy()
        self.tracked_confs = confs.copy() if confs is not None else np.ones(len(pts_2d), dtype=np.float32)
        self.initial_num_pts = len(pts_2d)

        # Store keyframe data for direct keyframe→current tracking (WAFT / ptlflow)
        if self.flow_method == 'waft' or self.use_ptlflow:
            self.keyframe_image = image.copy()
            self.keyframe_pts_2d = pts_2d.astype(np.float32).copy()
            self.keyframe_pts_3d = pts_3d.copy()
            self.keyframe_confs = confs.copy() if confs is not None else np.ones(len(pts_2d), dtype=np.float32)

    def track_to_frame(self, image: np.ndarray,
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Track features to new frame using frame-to-frame optical flow.

        Args:
            image: Current frame (H, W, 3) uint8 RGB.

        Returns:
            pts_2d: (M, 2) tracked 2D points in current frame
            pts_3d: (M, 3) corresponding 3D points
            confs: (M,) match confidences from keyframe
            num_tracked: Number of successfully tracked points"""
        if self.tracked_pts_2d is None or len(self.tracked_pts_2d) < 10:
            return None, None, None, 0

        if self.flow_method == 'lk' and self.use_gpu_lk:
            return self._track_lk_gpu(image)
        elif self.flow_method == 'waft':
            return self._track_waft(image)
        elif self.use_ptlflow:
            return self._track_ptlflow(image)
        else:
            return self._track_lk(image)

    def _track_lk(self, image: np.ndarray,
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Track using Lucas-Kanade optical flow."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Adaptive pyramid levels based on inter-frame motion
        eff_level = self._effective_lk_level()
        lk_params = dict(self.lk_params, maxLevel=eff_level)

        # Track with optical flow (frame-to-frame)
        new_pts, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_image, gray, self.tracked_pts_2d, None, **lk_params
        )

        # Backward check for robustness
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_image, new_pts, None, **lk_params
        )

        # Check forward-backward consistency
        orig_pts = self.tracked_pts_2d.reshape(-1, 2)
        fb_error = np.linalg.norm(
            orig_pts - back_pts.reshape(-1, 2),
            axis=1
        )

        # Keep points with good tracking
        status = status.flatten()
        back_status = back_status.flatten()
        bidir_ok = (status == 1) & (back_status == 1)

        # Estimate actual inter-frame motion from converged forward tracks
        n_bidir = bidir_ok.sum()
        if n_bidir > 10:
            fwd_disp = np.linalg.norm(
                new_pts.reshape(-1, 2)[bidir_ok] - orig_pts[bidir_ok], axis=1)
            median_disp = float(np.median(fwd_disp))
        else:
            median_disp = 0.0
        self.last_median_displacement = median_disp

        good_mask = bidir_ok & (fb_error < self.fb_threshold)

        # Motion-adaptive FB cap
        half_win = self._lk_win_size / 2.0
        search_range = half_win * (2 ** self._lk_max_level)
        motion_fb_cap = max(3.0, min(median_disp * 0.15, search_range * 0.03))

        # Adaptive FB: if too few points pass, relax threshold using p25
        base_count = good_mask.sum()
        if base_count < self.initial_num_pts * 0.15 and n_bidir > 50:
            fb_errs_ok = fb_error[bidir_ok]
            p25 = float(np.percentile(fb_errs_ok, 25))
            adaptive_th = min(max(p25 * 2, self.fb_threshold), motion_fb_cap)
            if adaptive_th > self.fb_threshold:
                good_mask = bidir_ok & (fb_error < adaptive_th)
                new_count = good_mask.sum()
                if new_count > base_count:
                    print(f"  [LK] Adaptive FB: {self.fb_threshold:.1f}->{adaptive_th:.1f}px "
                          f"(p25={p25:.1f}, motion={median_disp:.0f}px, cap={motion_fb_cap:.1f}), "
                          f"pts {base_count}->{new_count}")

        # Update tracked points
        good_new = new_pts[good_mask].reshape(-1, 2)
        good_3d = self.tracked_pts_3d[good_mask]
        good_confs = self.tracked_confs[good_mask]

        # Track survival rate for downstream adaptive thresholds
        self.last_survival_rate = len(good_new) / max(1, self.initial_num_pts)

        # Update state for next frame
        self.prev_image = gray
        self.tracked_pts_2d = good_new.reshape(-1, 1, 2).astype(np.float32)
        self.tracked_pts_3d = good_3d
        self.tracked_confs = good_confs

        return good_new, good_3d, good_confs, len(good_new)

    def _track_lk_gpu(self, image: np.ndarray,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Track using GPU-accelerated Lucas-Kanade via PyTorch grid_sample."""
        import torch

        # Convert current frame to grayscale GPU tensor
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_t = torch.from_numpy(gray.astype(np.float32)).cuda()

        # Convert tracked points to (N, 2) GPU tensor
        pts_t = torch.from_numpy(
            self.tracked_pts_2d.reshape(-1, 2).astype(np.float32)
        ).cuda()

        # Adaptive pyramid levels based on inter-frame motion
        eff_level = self._effective_lk_level()
        gpu_lk = self._get_gpu_lk()
        if hasattr(gpu_lk, '_lk') and hasattr(gpu_lk._lk, 'setMaxLevel'):
            gpu_lk._lk.setMaxLevel(eff_level)
        elif hasattr(gpu_lk, 'max_level'):
            gpu_lk.max_level = eff_level

        # Run GPU LK with forward-backward check using base threshold
        gpu_lk = self._get_gpu_lk()
        new_pts_t, status_t, fb_err_t = gpu_lk.calc_with_fb_check(
            self.prev_image, gray_t, pts_t, self.fb_threshold,
        )

        # Estimate actual inter-frame motion from converged forward tracks
        finite_mask = torch.isfinite(fb_err_t)  # bidir-converged points
        n_finite = finite_mask.sum().item()
        if n_finite > 10:
            fwd_disp = torch.norm(new_pts_t[finite_mask] - pts_t[finite_mask], dim=1)
            median_disp = float(torch.median(fwd_disp).item())
        else:
            median_disp = 0.0
        self.last_median_displacement = median_disp

        # Motion-adaptive FB cap: scale with actual displacement so
        # dense video (small motion) stays tight while sparse video
        # (large motion) allows proportionally larger FB errors.
        # Cap at search_range * 0.03 to avoid accepting garbage tracks.
        half_win = self._lk_win_size / 2.0
        search_range = half_win * (2 ** self._lk_max_level)
        motion_fb_cap = max(3.0, min(median_disp * 0.15, search_range * 0.03))

        # Adaptive FB: if too few points pass the base threshold, relax it
        # using the 25th percentile of FB errors, capped by motion-adaptive limit
        base_count = status_t.sum().item()
        if base_count < self.initial_num_pts * 0.15 and n_finite > 50:
            fb_errs_ok = fb_err_t[finite_mask]
            p25 = float(torch.quantile(fb_errs_ok, 0.25).item())
            adaptive_th = min(max(p25 * 2, self.fb_threshold), motion_fb_cap)
            if adaptive_th > self.fb_threshold:
                status_t = finite_mask & (fb_err_t < adaptive_th)
                new_count = status_t.sum().item()
                if new_count > base_count:
                    print(f"  [LK] Adaptive FB: {self.fb_threshold:.1f}->{adaptive_th:.1f}px "
                          f"(p25={p25:.1f}, motion={median_disp:.0f}px, cap={motion_fb_cap:.1f}), "
                          f"pts {base_count}->{new_count}")

        # Transfer results back to CPU
        new_pts = new_pts_t.cpu().numpy()
        good_mask = status_t.cpu().numpy()

        # Update tracked points
        good_new = new_pts[good_mask].reshape(-1, 2)
        good_3d = self.tracked_pts_3d[good_mask]
        good_confs = self.tracked_confs[good_mask]

        # Track survival rate for downstream adaptive thresholds
        self.last_survival_rate = len(good_new) / max(1, self.initial_num_pts)

        # Update state for next frame
        self.prev_image = gray_t
        self.tracked_pts_2d = good_new.reshape(-1, 1, 2).astype(np.float32)
        self.tracked_pts_3d = good_3d
        self.tracked_confs = good_confs

        return good_new, good_3d, good_confs, len(good_new)

    def _track_waft(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Track using WAFT dense optical flow — keyframe to current frame directly.

        Unlike LK which chains frame-to-frame, WAFT handles large motions so
        we always compute flow from the keyframe to the current frame. This
        eliminates drift accumulation entirely.

        Forward-only flow (no FB check) for speed — PnP RANSAC on the tracked
        frame already filters outliers robustly."""
        waft = self._get_waft()

        # Forward-only flow from keyframe → current (no backward pass needed)
        new_pts = waft.track_points(
            self.keyframe_image, image, self.keyframe_pts_2d
        )

        # Bounds check: keep only points inside image
        H, W = image.shape[:2]
        in_bounds = (
            (new_pts[:, 0] >= 0) & (new_pts[:, 0] < W) &
            (new_pts[:, 1] >= 0) & (new_pts[:, 1] < H)
        )
        good_pts = new_pts[in_bounds]
        good_3d = self.keyframe_pts_3d[in_bounds]
        good_confs = self.keyframe_confs[in_bounds]

        # Update tracked state (used for point count / distribution checks)
        self.tracked_pts_2d = good_pts.reshape(-1, 1, 2).astype(np.float32)
        self.tracked_pts_3d = good_3d
        self.tracked_confs = good_confs

        return good_pts, good_3d, good_confs, len(good_pts)


    def _track_ptlflow(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Track using ptlflow dense optical flow — keyframe to current frame directly.

        Same strategy as WAFT: always flow from keyframe → current to avoid drift."""
        model = self._get_ptlflow()

        # Forward-only flow from keyframe → current
        new_pts = model.track_points(
            self.keyframe_image, image, self.keyframe_pts_2d
        )

        # Bounds check: keep only points inside image
        H, W = image.shape[:2]
        in_bounds = (
            (new_pts[:, 0] >= 0) & (new_pts[:, 0] < W) &
            (new_pts[:, 1] >= 0) & (new_pts[:, 1] < H)
        )
        good_pts = new_pts[in_bounds]
        good_3d = self.keyframe_pts_3d[in_bounds]
        good_confs = self.keyframe_confs[in_bounds]

        # Update tracked state
        self.tracked_pts_2d = good_pts.reshape(-1, 1, 2).astype(np.float32)
        self.tracked_pts_3d = good_3d
        self.tracked_confs = good_confs

        return good_pts, good_3d, good_confs, len(good_pts)

    def offload_waft(self):
        """Move WAFT/ptlflow to CPU to free GPU memory for keyframe matching."""
        if self.waft is not None:
            self.waft.offload_to_cpu()
        if self.ptlflow_model is not None:
            self.ptlflow_model.offload_to_cpu()

    def reload_waft(self):
        """Move WAFT/ptlflow back to GPU after keyframe matching."""
        if self.waft is not None:
            self.waft.reload_to_gpu()
        if self.ptlflow_model is not None:
            self.ptlflow_model.reload_to_gpu()
