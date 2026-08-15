"""
TrackingPipeline — orchestrates keyframe localisation, optical-flow tracking,
and result collection.

Heavy logic is delegated to:
  - orthotrack.localization   (matching, PnP, DSM lifting)
  - orthotrack.crop_strategy  (DOP crop computation)
  - orthotrack.visualization  (per-frame / summary plots)
  - orthotrack.evaluation     (error maths, printing)
  - orthotrack.feature_tracker(FeatureTracker)"""

# ── Stage-level debug flag ────────────────────────────────────────────────────
# Set to True to save per-stage visualizations and (optionally) stop after each
# stage for step-by-step inspection of the first-frame localization pipeline.
# Each stage has a commented-out sys.exit() that you can uncomment to stop there.
PIPELINE_DEBUG_STAGES = False

# Cap the longest side of input UAV frames (pixels). Matching and flow run at this
# resolution (or smaller). Set to 0 in TrackingPipeline to disable downscaling.
DEFAULT_MAX_IMAGE_DIM = 1920

import numpy as np
import cv2
import time
import json
import random
import threading
from collections import deque
import torch
from concurrent.futures import ThreadPoolExecutor

try:
    from decord import VideoReader, cpu as decord_cpu
    _DECORD_AVAILABLE = True
except ImportError:
    _DECORD_AVAILABLE = False
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from functools import partial

from utils.geo import GeoTIFFHandler, MultiTileGeoTIFFHandler, SequenceGeoHandler
from utils.pose import PoseLoader, CameraPose, CameraIntrinsics, CSVPoseLoader, rotation_to_quat

from orthotrack.types import FrameResult

from orthotrack.feature_tracker import FeatureTracker
from orthotrack.matchers.base_matcher import BaseMatcher
from orthotrack.matchers import create_matcher

from orthotrack import localization as loc
from orthotrack import crop_strategy as crop
from orthotrack import visualization as vis
from orthotrack import evaluation as evl
from orthotrack import geo_localizer as geo_loc
from orthotrack.sensor_prior import SensorPrior
from orthotrack import baseline_modes
from orthotrack.exceptions import (
    FirstFrameLocalizationError,
    InvalidGeometryError,
    InsufficientConfidentMatchesError,
    IntrinsicsRequiredError,
    KeyframeLocalizationError,
    VisibleCropError,
)
from utils.image import downsample_image


class _NullGTPose:
    """Sentinel used in place of a real CameraPose when no GT is available.

    All coordinate/quaternion fields are None.  NumPy and FrameResult both
    accept None for optional GT fields, so no individual None-checks are
    needed elsewhere when this sentinel is returned from get_pose()."""
    x = y = z = None
    qw = qx = qy = qz = None
    rotation_matrix = None


class TrackingPipeline:
    """
    Main tracking pipeline with:
    - GPS only on first frame
    - Optical flow tracking of 2D-3D correspondences
    - Keyframe-based re-localisation
    - View-based DOP cropping using previous pose"""

    def __init__(
        self,
        dop_path: str = None,
        dsm_path: str = None,
        gt_json_path: str = None,
        footage_dir: str = None,
        output_dir: str = "results",
        sequence_dir: str = None,
        fine_matcher_setting: str = "fast",
        fps: float = 24.0,
        keyframe_min_points: int = 100,
        keyframe_max_interval: int = 30,
        num_crop_candidates: int = 5,
        save_keyframe_vis: bool = False,
        save_tracking_vis: bool = False,
        save_vis: bool = False,                    # combined: enables keyframe + tracking + stage debug vis
        vis_interval: int = 1,                     # save tracking vis every N frames (1=every frame)
        flow_method: str = "lk",
        num_matches: int = 3000,
        grace_ramp_frames: int = 15,
        min_kf_interval: int = 5,
        # --- Tunable hyperparameters (sensitivity study) ---
        point_drop_ratio: float = 0.40,
        confidence_threshold: float = 0.4,
        confidence_fallback: float = 0.1,
        confidence_min_count: int = 50,
        fb_threshold: float = 1.0,
        pnp_reproj_threshold: float = 7.0,
        single_crop_min_inliers_ratio: float = 0.30,  # fraction of num_matches required as PnP inliers
        spatial_collapse_frac: float = None,
        spatial_collapse_px: float = 30.0,
        reproj_abs_threshold: float = 2.0,
        growth_decay_frames: int = 100,
        min_growth_margin: float = 0.35,
        # --- Optional pre-built matcher (avoids reloading model) ---
        fine_matcher: 'BaseMatcher' = None,
        fine_matcher_name: str = None,
        coarse_matcher_setting: str = "turbo",
        dop_year: 'Union[str, int]' = 'last',
        fig_ext: str = "png",
        max_keyframes: int = None,
        tracking_mode: str = "default",
        filter_by_trackability: bool = False,
        max_tracking_points: int = 0,
        accumulate_points: bool = False,
        # --- Sensor prior (simulated GPS + IMU) ---
        use_prior: bool = False,
        prior_gps_sigma: float = 3.0,
        prior_gps_vertical_sigma: float = 5.0,
        prior_imu_sigma: float = 1.0,
        prior_imu_yaw_sigma: float = 4.0,
        prior_seed: int = 42,
        # --- Image resizing ---
        max_image_dim: int = DEFAULT_MAX_IMAGE_DIM,

        # --- Initial position prior (UTM) for first frame ---
        initial_position: 'Optional[np.ndarray]' = None,
        # --- LOD mesh for debug visualization overlay ---
        lod_obj_dir: str = None,
        # --- Force FoV self-calibration even if intrinsics.json is present ---
        force_calibration: bool = False,
        # --- Explicit intrinsics file (overrides auto-detection) ---
        intrinsics_path: str = None,
        # --- DSM degradation (rebuttal sensitivity sweep) ---
        dsm_scale: float = 1.0,
        dsm_sigma_z: float = 0.0,
        dsm_noise_seed: int = 0,
        # --- DOP degradation (rebuttal sensitivity sweep) ---
        dop_scale: float = 1.0,
    ):
        self.output_dir = Path(output_dir)
        self.flow_method = flow_method
        # Combined vis flag: save_vis enables all visualization outputs
        save_keyframe_vis = save_keyframe_vis or save_vis
        save_tracking_vis = save_tracking_vis or save_vis
        # Use os.makedirs with retry to handle NFS race conditions
        import os as _os
        for _attempt in range(5):
            try:
                _os.makedirs(str(self.output_dir), exist_ok=True)
                break
            except OSError:
                import time as _t; _t.sleep(0.5)
        if save_keyframe_vis:
            for _attempt in range(5):
                try:
                    _os.makedirs(str(self.output_dir / "keyframes"), exist_ok=True)
                    break
                except OSError:
                    import time as _t; _t.sleep(0.5)

        self.fps = fps
        self.video_fps = fps  # actual video FPS (may be read from file)
        self.dt = 1.0 / fps
        self.keyframe_min_points = keyframe_min_points
        self.keyframe_max_interval = keyframe_max_interval
        self.num_crop_candidates = num_crop_candidates
        self.num_matches = num_matches

        # Store tunable hyperparameters
        self.point_drop_ratio = point_drop_ratio
        self.confidence_threshold = confidence_threshold
        self.confidence_fallback = confidence_fallback
        self.confidence_min_count = confidence_min_count
        self.fb_threshold = fb_threshold
        self.pnp_reproj_threshold = pnp_reproj_threshold
        self.single_crop_min_inliers_ratio = single_crop_min_inliers_ratio
        self.spatial_collapse_frac = spatial_collapse_frac
        self.spatial_collapse_px = spatial_collapse_px
        self.growth_decay_frames = growth_decay_frames
        self.min_growth_margin = min_growth_margin
        self.max_tracking_points = max_tracking_points


        # -- initial position prior (UTM) for first frame ----------------
        self.initial_position = initial_position

        # -- image resizing (auto-downscale large images) ---------------
        self.max_image_dim = max_image_dim
        self._image_scale: float = 1.0  # set after first image load
        self._image_resize_initialized = False
        self._intrinsics_calibrated = False  # set True after self-calibration

        # -- geo -------------------------------------------------------
        self.dsm_scale = float(dsm_scale)
        self.dsm_sigma_z = float(dsm_sigma_z)
        self.dsm_noise_seed = int(dsm_noise_seed)
        self.dop_scale = float(dop_scale)
        self._init_geo(sequence_dir, dop_path, dsm_path, dop_year)

        # -- camera intrinsics (from intrinsics.json / meta.json) --------
        self._init_intrinsics(sequence_dir, footage_dir, intrinsics_path, force_calibration)

        # -- poses (evaluation only) -----------------------------------
        if gt_json_path is not None:
            print("Loading ground truth poses...")
            if gt_json_path.endswith(".csv"):
                self.gt_reader = CSVPoseLoader(gt_json_path)
            else:
                self.gt_reader = PoseLoader(gt_json_path)
        else:
            self.gt_reader = None
            print("No ground truth poses provided — error metrics will not be computed.")

        # -- footage ---------------------------------------------------
        self._init_footage(footage_dir)

        # -- fine matcher -----------------------------------------------
        if fine_matcher is not None:
            self.fine_matcher = fine_matcher
        elif fine_matcher_name is not None:
            print(f"Initialising fine matcher: {fine_matcher_name}")
            self.fine_matcher = create_matcher(fine_matcher_name)
        else:
            print(f"Initialising fine matcher ({fine_matcher_setting})...")
            self.fine_matcher = create_matcher(fine_matcher_setting)

        # -- coarse matcher (fast model for ROI detection / grid search) --
        _effective_fine_setting = (
            fine_matcher_name if fine_matcher_name is not None else
            (fine_matcher_setting if fine_matcher is None else None)
        )
        if coarse_matcher_setting == _effective_fine_setting:
            # Same setting as fine matcher -- reuse the same instance
            self.coarse_matcher = self.fine_matcher
            print(f"  Coarse matcher: reusing fine matcher ({coarse_matcher_setting})")
        else:
            print(f"  Initialising coarse matcher ({coarse_matcher_setting})...")
            self.coarse_matcher = create_matcher(coarse_matcher_setting)

        # -- LOD mesh (optional, for debug stage overlay) ---------------
        self._init_lod(lod_obj_dir)

        # -- components ------------------------------------------------
        self.tracker = FeatureTracker(max_features=2000, flow_method=self.flow_method,
                                      fb_threshold=self.fb_threshold,
                                      filter_by_trackability=filter_by_trackability,
                                      max_tracking_points=self.max_tracking_points,
                                      accumulate_points=accumulate_points)

        # -- state -----------------------------------------------------
        self.last_keyframe_id = -1
        self.frames_since_keyframe = 0

        self.reproj_error_history: List[float] = []
        self.keyframe_reproj_threshold = 2.0  # growth factor: trigger when reproj grows 100%
        self.keyframe_baseline_reproj: float = 0.0  # reproj right after keyframe
        self.reproj_abs_threshold: float = reproj_abs_threshold  # absolute reproj threshold for keyframe trigger
        self.reproj_abs_default: float = reproj_abs_threshold  # default value to reset to
        self.consecutive_reproj_keyframes: int = 0  # count of consecutive reproj-triggered keyframes
        self.grace_ramp_frames: int = grace_ramp_frames  # frames over which reproj threshold grace decays
        self.min_kf_interval: int = min_kf_interval  # minimum frames between consecutive KFs (warmup cooldown)

        self.prev_position: Optional[np.ndarray] = None
        self.prev_R_c2w: Optional[np.ndarray] = None

        self.last_keyframe_position: Optional[np.ndarray] = None
        self.last_keyframe_time: Optional[int] = None
        self.last_keyframe_crop: Optional[tuple] = None  # (cx, cy, size) of last keyframe DOP crop
        self.last_keyframe_R_c2w: Optional[np.ndarray] = None  # R_c2w from last keyframe (from direct DOP matching)
        self.pnp_failed_prev = False

        self.save_keyframe_vis = save_keyframe_vis
        self.save_tracking_vis = save_tracking_vis
        self.vis_interval = max(1, int(vis_interval))
        self.fig_ext = fig_ext
        self.max_keyframes = max_keyframes

        # Tracking mode: "default", "localize_every_frame", "dsm_tracking_only"
        self.tracking_mode = tracking_mode

        # Trajectory tracking for visualization
        self._trajectory_positions: List[Tuple[float, float]] = []
        self._trajectory_kf_flags: List[bool] = []
        self._current_processing_fps: Optional[float] = None
        self._total_processing_time: float = 0.0
        self._total_processed_frames: int = 0
        # Persistent keyframe DOP points (for showing lost points in tracked frames)
        self._keyframe_all_dop_pts: Optional[np.ndarray] = None
        self._keyframe_all_dop_confs: Optional[np.ndarray] = None
        # _step_history tracks recent frame-to-frame displacements for the jump guard.
        self._step_history: deque = deque(maxlen=15)  # recent frame-to-frame displacements (m)

        # Guard against expanding-search death spiral: after N consecutive
        # failures, skip the expensive expanding search and fall through to
        # prediction immediately.
        self._consecutive_expanding_failures: int = 0
        self._max_consecutive_expanding_failures: int = 3
        # Re-localization: trigger full grid search if many consecutive
        # frames fail after the first frame (catches bad first-frame init).
        self._consecutive_frame_failures: int = 0
        self._reloc_failure_threshold: int = 5  # trigger re-loc after this many
        self._reloc_attempts: int = 0
        self._max_reloc_attempts: int = 3  # max re-localizations per sequence
        # Results snapshot and threshold history for time-series visualization
        self._vis_results_snapshot: List = []
        self._vis_threshold_history: List[Tuple[int, float, float]] = []  # (frame_id, abs_threshold, rel_threshold)

        # --- Sensor prior (simulated GPS + IMU) ---
        self.use_prior = use_prior
        self.sensor_prior: Optional[SensorPrior] = None
        if self.use_prior:
            if self.gt_reader is None:
                raise ValueError("--use_prior requires --gt_poses to be provided")
            self.sensor_prior = SensorPrior.from_gt_reader(
                self.gt_reader,
                gps_horizontal_sigma=prior_gps_sigma,
                gps_vertical_sigma=prior_gps_vertical_sigma,
                imu_roll_sigma=prior_imu_sigma,
                imu_pitch_sigma=prior_imu_sigma,
                imu_yaw_sigma=prior_imu_yaw_sigma,
                seed=prior_seed,
            )
            print(f"  Sensor prior enabled: GPS σ_h={prior_gps_sigma}m, σ_v={prior_gps_vertical_sigma}m, "
                  f"IMU σ_rp={prior_imu_sigma}°, σ_yaw={prior_imu_yaw_sigma}°")

        # Build LocalizationConfig for the coarse/fine pipeline
        self._loc_cfg = geo_loc.LocalizationConfig(
            geo_handler=self.geo_handler,
            coarse_matcher=self.coarse_matcher,
            fine_matcher=self.fine_matcher,
            intrinsics=self.intrinsics,
            num_matches=self.num_matches,
            confidence_threshold=self.confidence_threshold,
            confidence_min_count=self.confidence_min_count,
            pnp_reproj_threshold=self.pnp_reproj_threshold,
            use_prior=self.use_prior,
            sensor_prior=self.sensor_prior,
        )
        # Debug visualizer for stage-by-stage inspection (only used when
        # debug_stages=True is passed to _localize_full_pipeline)
        from orthotrack.visualization import StageDebugVisualizer
        self._stage_vis = StageDebugVisualizer(
            self.output_dir / "keyframes",
            self.geo_handler,
            self.intrinsics,
            lod=self.lod,
        )
        if self._reference_fov is not None:
            self._stage_vis.set_gt_fov(self._reference_fov)

    # ------------------------------------------------------------------ #
    #  Init sub-routines (called from __init__)                           #
    # ------------------------------------------------------------------ #

    def _init_geo(self, sequence_dir, dop_path, dsm_path, dop_year):
        if sequence_dir:
            print(f"Loading sequence data from {sequence_dir}...")
            self.geo_handler = SequenceGeoHandler(
                sequence_dir, dop_year=dop_year,
                dsm_scale=self.dsm_scale,
                dsm_sigma_z=self.dsm_sigma_z,
                dsm_noise_seed=self.dsm_noise_seed,
                dop_scale=self.dop_scale,
            )
        else:
            print("Loading geospatial data...")
            if not dop_path or not dsm_path:
                raise ValueError("Both dop_path and dsm_path are required if sequence_dir is not provided.")
            dop_paths = list(dop_path)
            dsm_paths = list(dsm_path)
            if len(dop_paths) == 1 and len(dsm_paths) == 1:
                self.geo_handler = GeoTIFFHandler(dop_paths[0], dsm_paths[0])
                max_side = max(getattr(self.geo_handler, "dop_width", 0) or 0,
                               getattr(self.geo_handler, "dop_height", 0) or 0)
                if max_side and max_side <= 6000:
                    self.geo_handler.preload(is_dsm=False)
                    self.geo_handler.preload(is_dsm=True)
            else:
                self.geo_handler = MultiTileGeoTIFFHandler(dop_paths, dsm_paths)
                print(f"  Multi-tile: {len(self.geo_handler._dop_tiles)} DOP, "
                      f"{len(self.geo_handler._dsm_tiles)} DSM tiles")

    def _init_intrinsics(self, sequence_dir, footage_dir, intrinsics_path, force_calibration):
        if intrinsics_path is not None:
            _ip = Path(intrinsics_path)
            intrinsics_source = str(_ip.parent if _ip.suffix == '.json' else _ip)
        else:
            intrinsics_source = sequence_dir
        if not intrinsics_source and footage_dir:
            footage_p = Path(footage_dir)
            parent = footage_p.parent if footage_p.is_dir() else footage_p.parent
            if (parent / "intrinsics.json").exists() or (parent / "meta.json").exists():
                intrinsics_source = str(parent)
        if intrinsics_source:
            self.intrinsics = CameraIntrinsics.from_meta(intrinsics_source)
            if self.intrinsics.fx is not None or self.intrinsics.fov_vertical > 0:
                self._intrinsics_calibrated = True
            print(f"  Camera intrinsics: fov={self.intrinsics.fov_vertical:.1f}\u00b0, "
                  f"fx={self.intrinsics.fx}, cx={self.intrinsics.cx}")
        else:
            self.intrinsics = CameraIntrinsics()

        # Capture distortion + native K from intrinsics.json BEFORE
        # ``--force_calibration`` clears fx/fy/cx/cy. We use these to undistort
        # query frames at load time even when self-calibration takes over for
        # the focal length used downstream.
        self._undistort_dist = getattr(self.intrinsics, 'dist_coef', None)
        if self._undistort_dist is not None and self.intrinsics.fx is not None:
            self._undistort_native_K = np.array([
                [float(self.intrinsics.fx), 0.0, float(self.intrinsics.cx if self.intrinsics.cx is not None else self.intrinsics.width / 2)],
                [0.0, float(self.intrinsics.fy), float(self.intrinsics.cy if self.intrinsics.cy is not None else self.intrinsics.height / 2)],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            self._undistort_native_size = (int(self.intrinsics.height), int(self.intrinsics.width))
        else:
            self._undistort_native_K = None
            self._undistort_native_size = None
            self._undistort_dist = None

        self._reference_fov: Optional[float] = (
            self.intrinsics.fov_vertical
            if self._intrinsics_calibrated and self.intrinsics.fov_vertical > 0
            else None
        )

        _has_intrinsics = (
            self.intrinsics.fov_vertical > 0 or self.intrinsics.fx is not None
        )
        if not intrinsics_source and not force_calibration:
            raise IntrinsicsRequiredError(
                "No intrinsics source (intrinsics.json / meta.json next to sequence or "
                "footage). Provide them or pass --force-calibration."
            )
        if intrinsics_source and not _has_intrinsics and not force_calibration:
            raise IntrinsicsRequiredError(
                f"Intrinsics in {intrinsics_source} have no usable fov_vertical or fx/fy. "
                "Fix the file or use --force-calibration."
            )
        if not _has_intrinsics and force_calibration:
            self.intrinsics.fov_vertical = 60.0
            print("  No usable intrinsics \u2014 seed FoV=60\u00b0 for self-calibration (--force-calibration).")

        if force_calibration and self._intrinsics_calibrated:
            print(f"  [force_calibration] Ignoring loaded intrinsics "
                  f"(fov={self.intrinsics.fov_vertical:.1f}\u00b0, fx={self.intrinsics.fx}) \u2014 "
                  f"coarse stage will self-calibrate.")
            self.intrinsics.fx = None
            self.intrinsics.fy = None
            self.intrinsics.fov_vertical = 60.0
            self._intrinsics_calibrated = False

    def _init_footage(self, footage_dir):
        self.footage_dir = Path(footage_dir)
        self._load_image_lock = threading.Lock()
        if self.footage_dir.is_file() and self.footage_dir.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            self.is_video = True
            if _DECORD_AVAILABLE:
                self._vr = VideoReader(str(self.footage_dir), ctx=decord_cpu(0))
                total_frames = len(self._vr)
                self.num_frames = total_frames
                vid_fps = self._vr.get_avg_fps()
                if vid_fps > 0:
                    self.video_fps = vid_fps
                self.video_cap = None
                self._next_video_frame = None
                print(f"Found video with {total_frames} frames ({self.video_fps:.1f} fps) [decord]")
            else:
                self._vr = None
                self.video_cap = cv2.VideoCapture(str(self.footage_dir))
                total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.num_frames = total_frames
                self._next_video_frame = 0
                vid_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                if vid_fps > 0:
                    self.video_fps = vid_fps
                print(f"Found video with {total_frames} frames ({self.video_fps:.1f} fps) [cv2]")
            self.image_files = None
        else:
            self.is_video = False
            self.video_cap = None
            self.image_files = sorted(self.footage_dir.glob("*.jpeg"))
            if not self.image_files:
                self.image_files = sorted(self.footage_dir.glob("*.jpg"))
            if not self.image_files:
                self.image_files = sorted(self.footage_dir.glob("*.png"))
            self.num_frames = len(self.image_files)
            print(f"Found {len(self.image_files)} images")

    def _init_lod(self, lod_obj_dir):
        self.lod = None
        if lod_obj_dir is not None:
            try:
                from utils.lod import LoD, load_lod_from_gml, merge_obj_files
                paths = [Path(p) for p in lod_obj_dir]
                gml_paths = [p for p in paths if p.suffix.lower() == '.gml']
                obj_paths = [p for p in paths if p.suffix.lower() in {'.obj', '.ply'}]
                npz_paths = [p for p in paths if p.suffix.lower() == '.npz']
                if npz_paths:
                    if len(npz_paths) == 1:
                        self.lod = LoD.from_npz(str(npz_paths[0]))
                    else:
                        print("  LOD mesh: merging multiple .npz files is not currently supported")
                elif gml_paths:
                    self.lod = load_lod_from_gml(gml_paths, self.output_dir)
                elif obj_paths:
                    if len(obj_paths) == 1:
                        self.lod = LoD(str(obj_paths[0]))
                    else:
                        self.lod = merge_obj_files(obj_paths, self.output_dir)
                else:
                    print(f"  LOD mesh: no supported files (.obj, .ply, .gml, .npz) in provided list")
                if self.lod is not None:
                    print(f"  LOD mesh loaded: {len(self.lod.vertices)} vertices, "
                          f"{len(self.lod.faces)} faces")
            except Exception as _lod_e:
                print(f"  LOD mesh load failed: {_lod_e}")

    # ------------------------------------------------------------------ #
    #  I/O helpers                                                        #
    # ------------------------------------------------------------------ #



    def load_image(self, frame_id: int) -> np.ndarray:
        with self._load_image_lock:
            return self._load_image_unlocked(frame_id)

    def _load_image_unlocked(self, frame_id: int) -> np.ndarray:
        if self.is_video:
            if self._vr is not None:
                # decord: returns RGB uint8 (H, W, 3) directly — no cvtColor needed
                img = self._vr[frame_id].asnumpy()
            else:
                # cv2 fallback: avoid seek when reading sequentially
                if frame_id != self._next_video_frame:
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = self.video_cap.read()
                self._next_video_frame = frame_id + 1
                if not ret:
                    raise RuntimeError(f"Failed to read frame {frame_id}")
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raw = cv2.imread(str(self.image_files[frame_id]))
            img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        # Align declared intrinsics (e.g. COLMAP at 1920×1080) to the actual
        # frame size (demo videos are often already downscaled). Without this,
        # max_image_dim scaling is applied on top of the wrong native size and
        # PnP uses an inflated K.
        h, w = img.shape[:2]
        if not self._image_resize_initialized:
            iw = int(self.intrinsics.width or 0)
            ih = int(self.intrinsics.height or 0)
            if iw > 0 and ih > 0 and (iw != w or ih != h):
                sx = w / float(iw)
                sy = h / float(ih)
                print(
                    f"  Rescaling intrinsics to frame size: {iw}x{ih} -> {w}x{h} "
                    f"(sx={sx:.4f}, sy={sy:.4f})"
                )
                if self.intrinsics.fx is not None:
                    self.intrinsics.fx *= sx
                if self.intrinsics.fy is not None:
                    self.intrinsics.fy *= sy
                elif self.intrinsics.fx is not None:
                    self.intrinsics.fy = float(self.intrinsics.fx)
                if self.intrinsics.cx is not None:
                    self.intrinsics.cx *= sx
                if self.intrinsics.cy is not None:
                    self.intrinsics.cy *= sy
                self.intrinsics.width = w
                self.intrinsics.height = h

        # Auto-downscale large images to max_image_dim
        if self.max_image_dim > 0:
            h, w = img.shape[:2]
            if not self._image_resize_initialized:
                max_dim = max(h, w)
                if max_dim > self.max_image_dim:
                    self._image_scale = self.max_image_dim / max_dim
                    new_w = int(w * self._image_scale)
                    new_h = int(h * self._image_scale)
                    print(f"  Auto-resizing images: {w}x{h} -> {new_w}x{new_h} "
                          f"(scale={self._image_scale:.3f}, max_dim={self.max_image_dim})")
                    # Scale stored intrinsics to match new resolution
                    if self.intrinsics.fx is not None:
                        self.intrinsics.fx *= self._image_scale
                        self.intrinsics.fy *= self._image_scale
                        self.intrinsics.cx *= self._image_scale
                        self.intrinsics.cy *= self._image_scale
                    self.intrinsics.width = new_w
                    self.intrinsics.height = new_h
                else:
                    # No resize needed — still record image dimensions
                    if self.intrinsics.width == 0:
                        self.intrinsics.width = w
                    if self.intrinsics.height == 0:
                        self.intrinsics.height = h
                self._image_resize_initialized = True
            if self._image_scale < 1.0:
                new_w = int(w * self._image_scale)
                new_h = int(h * self._image_scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        elif not self._image_resize_initialized:
            if self.intrinsics.width == 0:
                self.intrinsics.width = w
            if self.intrinsics.height == 0:
                self.intrinsics.height = h
            self._image_resize_initialized = True

        # Undistort using prior distortion coefficients (e.g. COLMAP k1) so the
        # query frames the matcher sees match the pinhole model used by PnP.
        # Uses the *native* K captured at intrinsics-load time (before any
        # ``--force_calibration`` reset) and rescales it to the current frame.
        if self._undistort_dist is not None and self._undistort_native_K is not None:
            h2, w2 = img.shape[:2]
            nh, nw = self._undistort_native_size
            sx = w2 / float(nw) if nw > 0 else 1.0
            sy = h2 / float(nh) if nh > 0 else 1.0
            K = self._undistort_native_K.copy()
            K[0, 0] *= sx; K[0, 2] *= sx
            K[1, 1] *= sy; K[1, 2] *= sy
            try:
                img = cv2.undistort(img, K, self._undistort_dist.astype(np.float64))
            except Exception as _ud_e:  # noqa: BLE001
                print(f"  [warn] cv2.undistort failed ({_ud_e}); using raw frame")

        return img

    # ------------------------------------------------------------------ #
    #  Visualisation wrappers (bound to output_dir)                       #
    # ------------------------------------------------------------------ #

    def _save_keyframe_vis(self, *args, **kwargs):
        # Inject trajectory and processing FPS
        kwargs.setdefault('trajectory_positions', list(self._trajectory_positions))
        kwargs.setdefault('trajectory_keyframe_flags', list(self._trajectory_kf_flags))
        kwargs.setdefault('processing_fps', self._current_processing_fps)
        # Set results history and threshold history for time-series panels
        vis.save_keyframe_visualization._results_history = self._vis_results_snapshot
        vis.save_keyframe_visualization._threshold_history = self._vis_threshold_history
        vis.save_keyframe_visualization._keyframe_min_points = self.keyframe_min_points
        vis.save_keyframe_visualization(*args, output_dir=self.output_dir,
                                        geo_handler=self.geo_handler, fig_ext=self.fig_ext, **kwargs)

    def _save_keyframe_vis_from_points(
        self, frame_id, image, pts_2d, pts_3d, est_position, gt_position,
        num_inliers, crop_center, crop_size, accepted=True, confidences=None,
        crop_specs=None, frame_type="keyframe", reproj_error=None,
        keyframe_id=None, initial_num_pts=None, R_c2w=None,
    ):
        """Generate keyframe visualization from 2D/3D points by cropping the DOP."""
        # Smart DOP crop: tightly fit around the actual 3D world points + estimated
        # position, with a 30% margin (like the fine/coarse stage debug figures).
        if len(pts_3d) > 0 and est_position is not None:
            all_x = list(pts_3d[:, 0]) + [float(est_position[0])]
            all_y = list(pts_3d[:, 1]) + [float(est_position[1])]
            extent = max(max(all_x) - min(all_x), max(all_y) - min(all_y))
            dop_cx = (max(all_x) + min(all_x)) / 2
            dop_cy = (max(all_y) + min(all_y)) / 2
            dop_size_smart = max(extent * 1.3, 80.0)
        elif est_position is not None:
            dop_cx, dop_cy, dop_size_smart = float(est_position[0]), float(est_position[1]), 200.0
        else:
            dop_cx, dop_cy, dop_size_smart = crop_center[0], crop_center[1], crop_size
        dop_tile = self.geo_handler.crop_dop(dop_cx, dop_cy, dop_size_smart)
        if dop_tile is None:
            return
        if len(pts_2d) == 0:
            kpts_dop = np.zeros((0, 2))
            inlier_mask = np.zeros(0, dtype=bool)
        else:
            kpts_dop_x, kpts_dop_y = dop_tile.utm_to_pixel_batch(
                pts_3d[:, 0], pts_3d[:, 1]
            )
            kpts_dop = np.column_stack([kpts_dop_x, kpts_dop_y])
            inlier_mask = np.ones(len(pts_2d), dtype=bool)
        # crop_specs for Panel 2 DOP overview: show the re-localization search area
        # = centred at current est_position (best prior for next keyframe) with search crop_size
        if crop_specs is None:
            cx2 = float(est_position[0]) if est_position is not None else crop_center[0]
            cy2 = float(est_position[1]) if est_position is not None else crop_center[1]
            crop_specs = [(cx2, cy2, crop_size)]
        # For tracked frames: project stored keyframe UTM points into current smart tile
        kf_dop_pts = None
        kf_dop_confs = None
        if frame_type == "tracked" and self._keyframe_all_dop_pts is not None:
            kfx, kfy = dop_tile.utm_to_pixel_batch(
                self._keyframe_all_dop_pts[:, 0], self._keyframe_all_dop_pts[:, 1]
            )
            kf_dop_pts = np.column_stack([kfx, kfy])
            kf_dop_confs = self._keyframe_all_dop_confs
        # LoD edge overlay on UAV image (requires known rotation)
        lod_overlay = None
        if R_c2w is not None and est_position is not None:
            H_img, W_img = image.shape[:2]
            lod_overlay = self._stage_vis.render_lod_overlay(R_c2w, est_position, (H_img, W_img))
        # Footprint polygon for Panel 2 (actual ground footprint from camera pose)
        footprint_polygon = None
        if R_c2w is not None and est_position is not None and len(pts_3d) > 0:
            H_img, W_img = image.shape[:2]
            ground_z = float(np.median(pts_3d[:, 2]))
            footprint_polygon = vis.compute_footprint_polygon(
                est_position, R_c2w, (H_img, W_img), ground_z, self.intrinsics
            )
        self._save_keyframe_vis(
            frame_id, image, dop_tile, pts_2d, kpts_dop, inlier_mask,
            est_position=est_position, gt_position=gt_position,
            accepted=accepted, num_inliers=num_inliers,
            confidences=confidences, crop_specs=crop_specs,
            frame_type=frame_type, reproj_error=reproj_error,
            keyframe_id=keyframe_id, initial_num_pts=initial_num_pts,
            keyframe_dop_points=kf_dop_pts, keyframe_dop_confs=kf_dop_confs,
            lod_overlay=lod_overlay,
            footprint_polygon=footprint_polygon,
        )

    def _save_multicrop_vis(self, *args, **kwargs):
        vis.save_multicrop_visualization(*args, output_dir=self.output_dir, fig_ext=self.fig_ext, **kwargs)

    def _update_trajectory(self, position: np.ndarray, is_keyframe: bool, processing_time: float,
                           result: 'FrameResult' = None):
        """Track trajectory and processing FPS for visualization."""
        self._trajectory_positions.append((float(position[0]), float(position[1])))
        self._trajectory_kf_flags.append(is_keyframe)
        self._total_processing_time += processing_time
        self._total_processed_frames += 1
        if self._total_processing_time > 0:
            self._current_processing_fps = self._total_processed_frames / self._total_processing_time
        if result is not None:
            self._vis_results_snapshot.append(result)
            # Compute adaptive thresholds for this frame (Eq. 3)
            grace_margin = 2.0
            if self.grace_ramp_frames > 0:
                proximity_grace = grace_margin * max(0.0, 1.0 - self.frames_since_keyframe / self.grace_ramp_frames)
            else:
                proximity_grace = 0.0
            abs_thr = self.reproj_abs_threshold + proximity_grace
            # Relative threshold (Eq. 4)
            base = self.keyframe_baseline_reproj
            if base > 0 and self.frames_since_keyframe >= self.grace_ramp_frames:
                extra = self.frames_since_keyframe - self.grace_ramp_frames
                decay = min(1.0, extra / self.growth_decay_frames)
                ini = self.keyframe_reproj_threshold - 1.0
                margin = ini * (1.0 - decay) + self.min_growth_margin * decay
                rel_thr = base * (1.0 + margin)
            else:
                rel_thr = 999.0  # not active yet
            self._vis_threshold_history.append((result.frame_id, abs_thr, rel_thr))

    def _store_keyframe_dop_points(self, pts_3d: np.ndarray, confs: np.ndarray):
        """Store keyframe 3D points (UTM) for persistent DOP overlay in tracked frames."""
        if len(pts_3d) == 0:
            self._keyframe_all_dop_pts = None
            self._keyframe_all_dop_confs = None
            return
        self._keyframe_all_dop_pts = pts_3d.copy()
        self._keyframe_all_dop_confs = confs.copy() if confs is not None else None

    # ------------------------------------------------------------------ #
    #  FrameResult helpers                                                 #
    # ------------------------------------------------------------------ #

    def _make_result(self, frame_id, gt_pose, position, R_c2w, *,
                     is_keyframe, method, start_time,
                     num_tracked_points=0, num_inliers=0,
                     tracked_points_threshold=0,
                     reproj_error=None, kf_reason="",
                     baseline_reproj=0.0):
        """Build a successful FrameResult with errors and quaternion."""
        errors = evl.compute_errors(position, gt_pose, R_c2w)
        quat = rotation_to_quat(R_c2w)
        return FrameResult(
            frame_id=frame_id, is_keyframe=is_keyframe, success=True,
            est_x=position[0], est_y=position[1], est_z=position[2],
            est_qw=quat[3] if quat is not None else None,
            est_qx=quat[0] if quat is not None else None,
            est_qy=quat[1] if quat is not None else None,
            est_qz=quat[2] if quat is not None else None,
            gt_x=gt_pose.x, gt_y=gt_pose.y, gt_z=gt_pose.z,
            gt_qw=gt_pose.qw, gt_qx=gt_pose.qx,
            gt_qy=gt_pose.qy, gt_qz=gt_pose.qz,
            position_error=errors['position_error'],
            horizontal_error=errors['horizontal_error'],
            vertical_error=errors['vertical_error'],
            rotation_error=errors['rotation_error'],
            num_tracked_points=num_tracked_points,
            num_inliers=num_inliers,
            tracked_points_threshold=tracked_points_threshold,
            method=method,
            reproj_error=reproj_error,
            processing_time=time.time() - start_time,
            kf_reason=kf_reason,
            baseline_reproj=baseline_reproj,
        )

    @staticmethod
    def _make_failed_result(frame_id, gt_pose, *, is_keyframe, start_time,
                            num_tracked_points=0, tracked_points_threshold=0,
                            reproj_error=None, kf_reason="",
                            baseline_reproj=0.0):
        """Build a failed FrameResult."""
        return FrameResult(
            frame_id=frame_id, is_keyframe=is_keyframe, success=False,
            gt_x=gt_pose.x, gt_y=gt_pose.y, gt_z=gt_pose.z,
            gt_qw=gt_pose.qw, gt_qx=gt_pose.qx,
            gt_qy=gt_pose.qy, gt_qz=gt_pose.qz,
            num_tracked_points=num_tracked_points,
            tracked_points_threshold=tracked_points_threshold,
            method="failed",
            reproj_error=reproj_error,
            processing_time=time.time() - start_time,
            kf_reason=kf_reason,
            baseline_reproj=baseline_reproj,
        )

    def _localize_full_pipeline(
        self, frame_id: int, image: np.ndarray, h: int, w: int,
        verbose: bool = False,
        tracked_prior=None,
        tracked_pts_3d=None,
        calibrate: bool = False,

    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int,
               np.ndarray, np.ndarray, np.ndarray,
               Optional[Tuple[float, float, float]],
               Optional[list]]:
        """Run ``geo_localizer.localize_full_pipeline`` (coarse + fine for first KF; fine-only with prior).

        Returns (position, R_c2w, num_inliers, pts_2d, pts_3d, confs, crop_spec, crop_vis_data)."""
        *result, intrinsics_updated = geo_loc.localize_full_pipeline(
            frame_id, image, h, w, self._loc_cfg,
            tracked_prior=tracked_prior,
            tracked_pts_3d=tracked_pts_3d,
            calibrate=calibrate,
            debug_vis=self._stage_vis if self.save_keyframe_vis else None,
            prev_R_c2w=self.prev_R_c2w,
            save_crop_vis=self.save_keyframe_vis,
            verbose=verbose,
        )
        if intrinsics_updated:
            self._intrinsics_calibrated = True
        return tuple(result)

    # ------------------------------------------------------------------ #
    #  Per-frame processing methods (called from run_sequence)            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _gt_pos(gt_pose) -> 'Optional[np.ndarray]':
        """Extract GT position as array (or None if unavailable)."""
        if gt_pose is not None and gt_pose.x is not None:
            return np.array([gt_pose.x, gt_pose.y, gt_pose.z])
        return None

    def _process_first_frame(self, frame_id: int, image: np.ndarray,
                             gt_pose, h: int, w: int,
                             start_time: float, current_threshold: int,
                             verbose: bool) -> FrameResult:
        """Localize the first frame: coarse + fine estimation.

        On success, sets the keyframe and returns a FrameResult.
        On failure, raises FirstFrameLocalizationError."""
        if verbose:
            print(f"\n[Frame {frame_id}] First frame")

        if self._stage_vis is not None and gt_pose is not None:
            self._stage_vis.set_gt_pose(self._gt_pos(gt_pose), gt_pose.rotation_matrix)

        _need_calibrate = not self._intrinsics_calibrated and (
            self.intrinsics.fx is None or self.intrinsics.fov_vertical <= 0)
        position, est_rotation, num_inliers, \
            tracking_pts_2d, tracking_pts_3d, tracking_confs, \
            crop_spec, crop_vis_data = \
            self._localize_full_pipeline(frame_id, image, h, w, verbose=verbose,
                                         tracked_prior=self.initial_position,
                                         calibrate=_need_calibrate)

        crop_cx, crop_cy, crop_size = crop_spec if crop_spec is not None else (0.0, 0.0, 0.0)

        if position is None:
            print(f"  ERROR: First frame localization failed!")
            if self.save_keyframe_vis:
                vis_2d = tracking_pts_2d if tracking_pts_2d is not None and len(tracking_pts_2d) > 0 else np.zeros((0, 2), dtype=np.float32)
                vis_3d = tracking_pts_3d if tracking_pts_3d is not None and len(tracking_pts_3d) > 0 else np.zeros((0, 3), dtype=np.float32)
                vis_cf = tracking_confs if tracking_confs is not None and len(tracking_confs) > 0 else np.zeros(0, dtype=np.float32)
                self._save_keyframe_vis_from_points(
                    frame_id, image, vis_2d, vis_3d,
                    None, self._gt_pos(gt_pose),
                    0, (crop_cx, crop_cy), crop_size,
                    accepted=False, confidences=vis_cf if len(vis_2d) > 0 else None,
                    R_c2w=est_rotation,
                )
                if crop_vis_data:
                    self._save_multicrop_vis(
                        frame_id, image, crop_vis_data,
                        gt_position=self._gt_pos(gt_pose),
                    )
            raise FirstFrameLocalizationError(
                f"Frame {frame_id}: first frame localization failed — no pose estimated."
            )

        # Success — initialize tracking state
        self._tracking_established = True
        self.tracker.set_keyframe(frame_id, image, tracking_pts_2d, tracking_pts_3d,
                                  tracking_confs)
        self.last_keyframe_id = frame_id
        self.frames_since_keyframe = 0
        self.last_keyframe_position = position.copy()
        self.last_keyframe_time = frame_id
        self.last_keyframe_crop = (crop_cx, crop_cy, crop_size)
        self.last_keyframe_R_c2w = est_rotation
        self.keyframe_baseline_reproj = 0.0
        self.reproj_error_history.clear()
        self.prev_position = position
        self.prev_R_c2w = est_rotation

        result = self._make_result(
            frame_id, gt_pose, position, est_rotation,
            is_keyframe=True, method="keyframe", start_time=start_time,
            num_tracked_points=len(tracking_pts_2d),
            num_inliers=num_inliers, tracked_points_threshold=current_threshold,
        )
        self._update_trajectory(position, is_keyframe=True,
                                processing_time=time.time() - start_time, result=result)
        self._store_keyframe_dop_points(tracking_pts_3d, tracking_confs)

        if self.save_keyframe_vis:
            self._save_keyframe_vis_from_points(
                frame_id, image, tracking_pts_2d, tracking_pts_3d,
                position, self._gt_pos(gt_pose),
                num_inliers, (crop_cx, crop_cy), crop_size,
                confidences=tracking_confs,
                R_c2w=est_rotation,
            )
            if crop_vis_data:
                self._save_multicrop_vis(
                    frame_id, image, crop_vis_data,
                    est_position=position,
                    gt_position=self._gt_pos(gt_pose),
                    total_inliers=num_inliers,
                )
        return result

    def _create_new_keyframe(self, frame_id: int, image: np.ndarray,
                             gt_pose, h: int, w: int,
                             start_time: float, current_threshold: int,
                             num_tracked: int, pts_3d: np.ndarray,
                             last_reproj: float, kf_reason: str,
                             verbose: bool) -> FrameResult:
        """Create a new keyframe: prior pose → visible crop → match → PnP.

        On success, sets the keyframe and returns a FrameResult.
        On failure, raises KeyframeLocalizationError."""
        self.tracker.offload_waft()
        if verbose:
            print(f"  Creating new keyframe (tracked={num_tracked}, "
                  f"since_kf={self.frames_since_keyframe})")

        # Compute visible-crop from prior pose + rotation
        crop_R_c2w = self.last_keyframe_R_c2w if self.last_keyframe_R_c2w is not None else self.prev_R_c2w
        position = None
        num_inliers = 0
        est_rotation = None
        tracking_pts_2d = np.zeros((0, 2))
        tracking_pts_3d = np.zeros((0, 3))
        tracking_confs = np.zeros(0, dtype=np.float32)
        crop_cx, crop_cy, crop_size = 0.0, 0.0, 0.0

        # For max_interval-forced keyframes, skip the visible-crop stage.
        # After 100+ frames of LK tracking, the prior can be several metres off,
        # making the visible-crop unreliable. Go straight to the full pipeline.
        force_full_pipeline = kf_reason.startswith("max_interval")
        # Always use the tracked prior in the full pipeline (prior-aware fine crop).
        # Even for max_interval KFs the drift is typically small enough that the
        # prior-seeded fine search is far cheaper and equally accurate compared to
        # a global coarse tile search.
        prior_is_stale = False

        if not force_full_pipeline and self.prev_position is not None and crop_R_c2w is not None:
            try:
                crop_cx, crop_cy, crop_size = crop.compute_visible_dop_crop(
                    self.prev_position, crop_R_c2w, (h, w), self.intrinsics.fov_vertical,
                    self.geo_handler, verbose=verbose,
                    K=self.intrinsics.K,
                )
            except (VisibleCropError, InvalidGeometryError):
                pass
            else:
                # Match and lift to 3D
                init_2d, init_3d, init_cf = loc.match_and_lift(
                    image, (crop_cx, crop_cy), crop_size,
                    self.geo_handler, self.fine_matcher,
                    num_matches=self.num_matches, verbose=verbose,
                )
                # Filter by confidence and solve PnP (with adaptive fallback for low-confidence scenes)
                conf_mask = init_cf >= self.confidence_threshold
                if conf_mask.sum() < self.confidence_min_count:
                    conf_mask = init_cf >= self.confidence_fallback
                    if verbose:
                        print(f"  Low confidence scene: relaxed threshold to {self.confidence_fallback} ({conf_mask.sum()} pts)")
                if conf_mask.sum() >= self.confidence_min_count:
                    pnp_2d_in = init_2d[conf_mask]
                    pnp_3d_in = init_3d[conf_mask]
                    pnp_cf_in = init_cf[conf_mask]

                    if len(pnp_2d_in) >= 30:
                        position, pnp_2d, pnp_3d, num_inliers, est_rotation, pnp_idx = \
                            loc.localize_from_correspondences(
                                pnp_2d_in, pnp_3d_in, (h, w),
                                self.intrinsics.fov_vertical, verbose=verbose,
                                K=self.intrinsics.K,
                            )
                        if position is not None and num_inliers >= 50:
                            # Sanity check 1: reject visible-crop result if it jumped too far
                            # from the tracked prior (catches false matches during rapid
                            # altitude change — the crop was computed at wrong scale).
                            if self.prev_position is not None:
                                kf_jump_3d = float(np.linalg.norm(position - self.prev_position))
                                if kf_jump_3d > 50.0:
                                    if verbose:
                                        print(f"  Visible-crop KF rejected: 3D jump={kf_jump_3d:.1f}m > 50m "
                                              f"— falling back to full pipeline")
                                    position = None

                            # Sanity check 2: reject if too few inliers — poor visible-crop
                            # localization (e.g. stale prior after long drift) should fall
                            # back to a global full-pipeline search.
                            if position is not None and num_inliers < 300:
                                if verbose:
                                    print(f"  Visible-crop KF rejected: only {num_inliers} inliers < 300 "
                                          f"— falling back to full pipeline (prior-based fine search)")
                                position = None
                                force_full_pipeline = True  # skip visible-crop; prior is still valid

                        if position is not None:
                            # Use the tight PnP inliers directly for tracking.
                            # The original code never expanded beyond PnP inliers, and
                            # expanding with a loose 3× threshold (21px) adds ~60% noisy
                            # points that degrade LK tracking quality and cause RANSAC
                            # instability (inlier rate drops from ~95% to ~38%, causing
                            # ±5m frame-to-frame jitter).  Using strict inliers keeps
                            # the inlier rate near 95% matching original behavior.
                            tracking_pts_2d = pnp_2d
                            tracking_pts_3d = pnp_3d
                            tracking_confs = pnp_cf_in[pnp_idx] if len(pnp_idx) == len(pnp_2d) else pnp_cf_in[:len(pnp_2d)]

        # Fallback: full pipeline if visible-crop PnP failed
        if position is None:
            if verbose:
                if prior_is_stale:
                    print(f"  Max-interval KF: skipping visible-crop, using full pipeline (prior-aware fine search) for KF {frame_id}")
                elif force_full_pipeline:
                    print(f"  Visible-crop KF rejected — falling back to full pipeline (using tracked prior) for KF {frame_id}")
                else:
                    print(f"  Visible-crop failed — falling back to full pipeline for KF {frame_id}")
            try:
                # For max_interval KFs the prior may be metres off — force a global coarse search.
                # For other fallbacks (weak visible-crop result), the prior is still valid;
                # pass it so _localize_full_pipeline skips the expensive tile search.
                effective_prior = None if prior_is_stale else self.prev_position
                fb_pos, fb_rot, fb_inl, fb_2d, fb_3d, fb_cf, fb_crop, _ = \
                    self._localize_full_pipeline(frame_id, image, h, w, verbose=verbose,
                                                 tracked_prior=effective_prior,
                                                 tracked_pts_3d=pts_3d)
                if fb_pos is not None and fb_inl >= 30:
                    position = fb_pos
                    est_rotation = fb_rot
                    num_inliers = fb_inl
                    tracking_pts_2d = fb_2d
                    tracking_pts_3d = fb_3d
                    tracking_confs = fb_cf
                    if fb_crop is not None:
                        crop_cx, crop_cy, crop_size = fb_crop
            except Exception as _e:
                if verbose:
                    print(f"  Full pipeline also failed: {_e}")
                position = None

        if position is not None and num_inliers >= 30:
            # Track consecutive reproj-triggered keyframes for adaptive relaxation.
            reproj_triggered = (self.reproj_abs_threshold > 0
                                and last_reproj > self.reproj_abs_threshold)
            if reproj_triggered:
                self.consecutive_reproj_keyframes += 1
                if self.consecutive_reproj_keyframes >= 2:
                    self.reproj_abs_threshold = min(
                        self.reproj_abs_default + self.consecutive_reproj_keyframes * 0.5,
                        4.0,
                    )
                    if verbose:
                        print(f"  Reproj threshold relaxed to {self.reproj_abs_threshold:.1f}px "
                              f"({self.consecutive_reproj_keyframes} consecutive reproj keyframes)")
            else:
                self.consecutive_reproj_keyframes = 0

            self.tracker.set_keyframe(frame_id, image, tracking_pts_2d, tracking_pts_3d,
                                      tracking_confs)
            self.last_keyframe_id = frame_id
            self.frames_since_keyframe = 0
            self._consecutive_expanding_failures = 0
            self.last_keyframe_position = position.copy()
            self.last_keyframe_time = frame_id
            self.last_keyframe_crop = (crop_cx, crop_cy, crop_size)
            self.last_keyframe_R_c2w = est_rotation
            self.keyframe_baseline_reproj = 0.0
            self.reproj_error_history.clear()
            self._tracking_established = True
            self.prev_R_c2w = est_rotation

            result = self._make_result(
                frame_id, gt_pose, position, est_rotation,
                is_keyframe=True, method="keyframe", start_time=start_time,
                num_tracked_points=len(tracking_pts_2d),
                num_inliers=num_inliers, tracked_points_threshold=current_threshold,
                reproj_error=last_reproj if last_reproj > 0 else None,
                kf_reason=kf_reason,
                baseline_reproj=self.keyframe_baseline_reproj,
            )
            self._update_trajectory(position, is_keyframe=True,
                                    processing_time=time.time() - start_time, result=result)
            self._store_keyframe_dop_points(tracking_pts_3d, tracking_confs)

            if self.save_keyframe_vis:
                self._save_keyframe_vis_from_points(
                    frame_id, image, tracking_pts_2d, tracking_pts_3d,
                    position, self._gt_pos(gt_pose),
                    num_inliers, (crop_cx, crop_cy), crop_size,
                    confidences=tracking_confs,
                    R_c2w=est_rotation,
                )
            return result

        # Failed
        raise KeyframeLocalizationError(
            f"Frame {frame_id}: keyframe localization failed "
            f"(num_inliers={num_inliers}, num_tracked={num_tracked})."
        )

    def _process_tracked_frame(self, frame_id: int, image: np.ndarray,
                               gt_pose, h: int, w: int,
                               start_time: float, current_threshold: int,
                               num_tracked: int, pts_2d: np.ndarray,
                               pts_3d: np.ndarray, pts_confs: np.ndarray,
                               predicted_pos: 'Optional[np.ndarray]',
                               need_keyframe: bool, kf_reason: str,
                               last_reproj: float,
                               verbose: bool) -> FrameResult:
        """Estimate pose from tracked correspondences (non-keyframe path).

        Calls estimate_pose_from_2d3d_corrspondences and wraps result into FrameResult.
        If PnP fails or reproj is too bad, returns a failed result."""
        position, num_inliers, reproj_error, tracked_R_c2w_frame = \
            loc.estimate_pose_from_2d3d_corrspondences(
                pts_2d, pts_3d, (h, w), self.intrinsics.fov_vertical,
                verbose=verbose, reproj_threshold=self.pnp_reproj_threshold,
                K=self.intrinsics.K,
            )

        if reproj_error is not None:
            self.reproj_error_history.append(reproj_error)
            if len(self.reproj_error_history) > 20:
                self.reproj_error_history.pop(0)

        if position is not None and reproj_error < 10.0:
            # Jump guard: mirrors original's tracked_max_jump_factor=10.0 / tracked_min_jump_threshold=20.0.
            # Reject tracked pose if it jumps anomalously far from the previous position.
            # This catches LK optical-flow drift failures where tracked features "teleport".
            if self.prev_position is not None and len(self._step_history) >= 3:
                jump = float(np.linalg.norm(position[:2] - self.prev_position[:2]))
                expected = float(np.median(list(self._step_history)))
                if expected <= 0:
                    expected = 2.0
                max_jump = max(20.0, expected * 10.0)
                if jump > max_jump:
                    if verbose:
                        print(f"  Tracked pose jump={jump:.1f}m > {max_jump:.1f}m — rejected, forcing KF")
                    self.pnp_failed_prev = True
                    return self._make_failed_result(
                        frame_id, gt_pose, is_keyframe=False, start_time=start_time,
                        num_tracked_points=num_tracked, tracked_points_threshold=current_threshold,
                        reproj_error=reproj_error, baseline_reproj=self.keyframe_baseline_reproj,
                    )

            if verbose:
                print(f"  Tracked pose: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), "
                      f"reproj={reproj_error:.1f}px")

            result = self._make_result(
                frame_id, gt_pose, position, tracked_R_c2w_frame,
                is_keyframe=False, method="tracked", start_time=start_time,
                num_tracked_points=num_tracked, num_inliers=num_inliers,
                tracked_points_threshold=current_threshold,
                reproj_error=reproj_error,
                baseline_reproj=self.keyframe_baseline_reproj,
            )
            self._update_trajectory(position, is_keyframe=False,
                                    processing_time=time.time() - start_time, result=result)

            if self.save_tracking_vis and (frame_id % self.vis_interval == 0):
                kf_crop = self.last_keyframe_crop
                if kf_crop is not None:
                    self._save_keyframe_vis_from_points(
                        frame_id, image, pts_2d, pts_3d,
                        position, self._gt_pos(gt_pose),
                        num_inliers, (kf_crop[0], kf_crop[1]), kf_crop[2],
                        confidences=pts_confs,
                        frame_type="tracked",
                        reproj_error=reproj_error,
                        keyframe_id=self.last_keyframe_id,
                        initial_num_pts=self.tracker.initial_num_pts,
                        R_c2w=tracked_R_c2w_frame,
                    )

            if tracked_R_c2w_frame is not None:
                self.prev_R_c2w = tracked_R_c2w_frame
            return result

        # PnP failed or reproj too bad
        self.pnp_failed_prev = True
        if verbose:
            if need_keyframe:
                print("  Tracked PnP failed + keyframe needed, marking frame as failed")
            elif reproj_error is not None and reproj_error >= 10.0:
                print(f"  Tracking quality bad (reproj={reproj_error:.1f}px), marking frame as failed")
            else:
                print("  Tracking PnP failed, marking frame as failed")
        return self._make_failed_result(
            frame_id, gt_pose, is_keyframe=False, start_time=start_time,
            num_tracked_points=num_tracked, tracked_points_threshold=current_threshold,
            reproj_error=reproj_error if reproj_error is not None else (last_reproj if last_reproj > 0 else None),
            kf_reason=kf_reason if need_keyframe else None,
            baseline_reproj=self.keyframe_baseline_reproj,
        )

    # ------------------------------------------------------------------ #
    #  Primary tracking loop                                              #
    # ------------------------------------------------------------------ #

    def run_sequence(self, frame_indices: List[int],
                     verbose: bool = False, reverse: bool = False) -> List[FrameResult]:
        """Run tracking on *frame_indices* (forward or backward)."""

        # Print hyperparameters
        print("\n" + "=" * 60)
        print("OrthoTrack Tracking Pipeline")
        print("=" * 60)
        print(f"Total frames: {len(frame_indices)}")
        print(f"Processing frames {frame_indices[0]} to {frame_indices[-1]} "
              f"(every {frame_indices[1] - frame_indices[0] if len(frame_indices) > 1 else 1} frame)")
        print(f"Tracking {len(frame_indices)} frames")
        print(f"Keyframe settings: min_points={self.keyframe_min_points}")
        print(f"Reprojection threshold: {self.reproj_abs_threshold}px")
        print("=" * 60)
        print(f"\nHyperparameters:")
        print(f"  Fine matcher:          {type(self.fine_matcher).__name__}")
        print(f"  Coarse matcher:        {type(self.coarse_matcher).__name__}")
        print(f"  Flow method:           {self.flow_method}")
        print(f"  Num matches:           {self.num_matches}")
        print(f"  Confidence threshold:  {self.confidence_threshold}")
        print(f"  Confidence min count:  {self.confidence_min_count}")
        print(f"  PnP reproj threshold:  {self.pnp_reproj_threshold}px")
        print(f"  Reproj abs threshold:  {self.reproj_abs_threshold}px")
        print(f"  Point drop ratio:      {self.point_drop_ratio}")
        print(f"  FB threshold:          {self.fb_threshold}")
        print(f"  Single-crop min inlier ratio: {self.single_crop_min_inliers_ratio}")
        if self.use_prior:
            sp = self.sensor_prior
            print(f"  Sensor prior:          ON (GPS σ_h={sp.gps_horizontal_sigma:.1f}m, σ_v={sp.gps_vertical_sigma:.1f}m, "
                  f"IMU σ_rp={sp.imu_roll_sigma:.1f}°, σ_yaw={sp.imu_yaw_sigma:.1f}°)")
        else:
            print(f"  Sensor prior:          OFF")
        print(f"  Tracking mode:         {self.tracking_mode}")
        print(f"  Accumulate points:     {self.tracker.accumulate_points}")
        print(f"  Save keyframe vis:     {self.save_keyframe_vis}")
        print("=" * 60)
        # Dispatch to alternate tracking modes
        if self.tracking_mode == "localize_every_frame":
            return baseline_modes.run_localize_every_frame(self, frame_indices, verbose=verbose)
        elif self.tracking_mode == "dsm_tracking_only":
            return baseline_modes.run_dsm_tracking_only(self, frame_indices, verbose=verbose)

        # Fix seeds for reproducibility (RoMaV2 sampling + PnP RANSAC)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        cv2.setRNGSeed(42)
        random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        results: List[FrameResult] = []
        # Reset visualization state
        self._trajectory_positions.clear()
        self._trajectory_kf_flags.clear()
        self._vis_results_snapshot.clear()
        self._vis_threshold_history.clear()
        self._total_processing_time = 0.0
        self._total_processed_frames = 0
        self._current_processing_fps = None

        self._tracking_established = False  # True once we have a reliable first position
        self._consecutive_expanding_failures = 0
        self._consecutive_frame_failures = 0
        self._reloc_attempts = 0
        desc = "Backward" if reverse else "Tracking"
        pbar = tqdm(frame_indices, desc=desc)
        _prefetch_pool = ThreadPoolExecutor(max_workers=1)
        _prefetch_future = None

        # Rolling stats for postfix display (last 30 frames)
        _ROLL = 30
        _tracking_times: list[float] = []   # pure tracking (no vis)
        _vis_times: list[float] = []        # vis overhead per frame

        def _update_pbar(r, frame_wall_start: float):
            """Update tqdm description + postfix after each frame."""
            _type = "KF" if r.is_keyframe else "TR"
            if r.success and r.position_error is not None:
                pbar.set_description(f"{desc} [{_type} err={r.position_error:.1f}m]")
            elif r.success:
                pbar.set_description(f"{desc} [{_type} ok]")
            else:
                pbar.set_description(f"{desc} [{_type} FAIL]")

            # Compute vis overhead: wall time − pure tracking time
            wall = time.time() - frame_wall_start
            t_track = r.processing_time  # set before vis in _update_trajectory
            t_vis = max(0.0, wall - t_track)
            _tracking_times.append(t_track)
            _vis_times.append(t_vis)
            if len(_tracking_times) > _ROLL:
                _tracking_times.pop(0)
                _vis_times.pop(0)
            avg_track = sum(_tracking_times) / len(_tracking_times)
            avg_vis = sum(_vis_times) / len(_vis_times)
            fps = 1.0 / avg_track if avg_track > 0 else 0.0
            postfix: dict = {"fps": f"{fps:.1f}"}
            if avg_vis > 0.05:  # only show vis overhead when it matters
                postfix["vis"] = f"{avg_vis:.1f}s"
            pbar.set_postfix(postfix)

        for i, frame_id in enumerate(pbar):
            start_time = time.time()
            # Use prefetched frame if available, otherwise load synchronously
            if _prefetch_future is not None:
                image = _prefetch_future.result()
                _prefetch_future = None
            else:
                image = self.load_image(frame_id)
            # Prefetch next frame asynchronously
            if i + 1 < len(frame_indices):
                _prefetch_future = _prefetch_pool.submit(self.load_image, frame_indices[i + 1])
            gt_pose = self.gt_reader.get_pose(frame_id) if self.gt_reader is not None else _NullGTPose()
            h, w = image.shape[:2]

            current_threshold = (
                int(self.tracker.initial_num_pts * 0.25) if self.tracker.initial_num_pts > 0
                else self.keyframe_min_points
            )

            # ============================================================
            #  FIRST FRAME: coarse + fine localization
            # ============================================================
            if i == 0:
                result = self._process_first_frame(
                    frame_id, image, gt_pose, h, w, start_time, current_threshold, verbose)
                results.append(result)
                _update_pbar(result, start_time)
                continue

            # ============================================================
            #  SUBSEQUENT FRAMES
            # ============================================================
            predicted_pos = self.prev_position  # always set after first frame
            self.frames_since_keyframe += 1

            # ============================================================
            #  OPTICAL FLOW TRACKING (standard path)
            # ============================================================

            pts_2d, pts_3d, pts_confs, num_tracked = self.tracker.track_to_frame(
                image)

            # Use latest reprojection error as quality signal
            last_reproj = self.reproj_error_history[-1] if self.reproj_error_history else 0.0

            if verbose:
                reproj_str = f", reproj={last_reproj:.1f}px" if last_reproj > 0 else ""
                print(f"\n[Frame {frame_id}] Tracked {num_tracked} points from KF {self.last_keyframe_id}{reproj_str}")

            # ── Adaptive abs reproj threshold ─────────────────────────────────────
            # Record baseline reproj from the first tracked frame after each KF
            # and update the abs threshold to track per-sequence quality.
            #
            # Strategy: threshold = max(default, min(baseline + HEADROOM, MAX_SCALE * default))
            #   • HEADROOM (0.5 px) ensures the threshold sits above the normal
            #     per-sequence reproj range, not just equal to the baseline.
            #     Sequences that permanently operate at 3–5 px (self-calibrated
            #     intrinsics) would otherwise trigger KFs on every minor fluctuation.
            #   • MAX_SCALE (2.5×) caps the threshold so a very poor initial
            #     localization cannot suppress KF creation indefinitely.
            #   • On every new KF the threshold re-adapts, so quality improvements
            #     (e.g. re-entering a well-textured area) lower it again automatically.
            #   • The existing reset-to-default when reproj < 1.5 px is preserved
            #     for sequences where tracking quality genuinely recovers mid-run.
            if self.keyframe_baseline_reproj == 0.0 and last_reproj > 0:
                self.keyframe_baseline_reproj = last_reproj

            # Adaptive reproj threshold: reset to default when reproj is good
            if last_reproj > 0 and last_reproj < 1.5 and self.reproj_abs_threshold > self.reproj_abs_default:
                self.reproj_abs_threshold = self.reproj_abs_default
                self.consecutive_reproj_keyframes = 0
                if verbose:
                    print(f"  Reproj threshold reset to {self.reproj_abs_threshold:.1f}px (reproj recovered)")

            need_keyframe, kf_reason = loc.should_create_keyframe(
                num_tracked, self.frames_since_keyframe,
                self.keyframe_min_points,
                initial_num_pts=self.tracker.initial_num_pts,
                pts_2d=pts_2d,
                reproj_error=last_reproj,
                baseline_reproj=self.keyframe_baseline_reproj,
                reproj_growth_factor=self.keyframe_reproj_threshold,
                reproj_abs_threshold=self.reproj_abs_threshold,
                grace_ramp_frames=self.grace_ramp_frames,
                point_drop_ratio=self.point_drop_ratio,
                spatial_collapse_frac=self.spatial_collapse_frac,
                spatial_collapse_px=self.spatial_collapse_px,
                image_shape=(h, w),
                growth_decay_frames=self.growth_decay_frames,
                min_growth_margin=self.min_growth_margin,
                keyframe_max_interval=self.keyframe_max_interval,
                min_kf_interval=self.min_kf_interval,
            )

            if need_keyframe:
                print(f"  [Frame {frame_id}] KF trigger: {kf_reason} "
                      f"(tracked={num_tracked}, since_kf={self.frames_since_keyframe})")

            if self.pnp_failed_prev:
                self.pnp_failed_prev = False

            # --- Decide: keyframe or tracked frame ---
            if need_keyframe:
                try:
                    result = self._create_new_keyframe(
                        frame_id, image, gt_pose, h, w, start_time, current_threshold,
                        num_tracked, pts_3d, last_reproj, kf_reason, verbose)
                except KeyframeLocalizationError as _kfe:
                    print(f"  [warn] Keyframe localization failed at frame {frame_id}: {_kfe}")
                    print(f"  [warn] Using last tracked pose as fallback and continuing.")
                    # Build a failed result with the last known position so tracking can continue.
                    result = FrameResult(
                        frame_id=frame_id,
                        success=False,
                        is_keyframe=True,
                        est_x=float(self.prev_position[0]) if self.prev_position is not None else None,
                        est_y=float(self.prev_position[1]) if self.prev_position is not None else None,
                        est_z=float(self.prev_position[2]) if self.prev_position is not None else None,
                        num_inliers=0,
                        processing_time=time.time() - start_time,
                    )
            else:
                result = self._process_tracked_frame(
                    frame_id, image, gt_pose, h, w, start_time, current_threshold,
                    num_tracked, pts_2d, pts_3d, pts_confs,
                    predicted_pos, need_keyframe, kf_reason, last_reproj, verbose)
            results.append(result)
            _update_pbar(result, start_time)
            if results:
                last_res = results[-1]
                if last_res.success and last_res.est_x is not None:
                    new_pos = np.array([last_res.est_x, last_res.est_y, last_res.est_z])
                    # Update step history for the jump guard in _process_tracked_frame.
                    if last_res.is_keyframe:
                        self._step_history.clear()
                    elif self.prev_position is not None:
                        step = float(np.linalg.norm(new_pos - self.prev_position))
                        self._step_history.append(step)

                    self.prev_position = new_pos
                    self._consecutive_frame_failures = 0
                else:
                    self._consecutive_frame_failures += 1

            # ---- Re-localization: use last pose as prior + fine estimation ----
            if (self._consecutive_frame_failures >= self._reloc_failure_threshold
                    and self._reloc_attempts < self._max_reloc_attempts
                    and self._tracking_established):
                self._reloc_attempts += 1
                if verbose:
                    print(f"\n  RE-LOCALIZATION triggered "
                          f"({self._consecutive_frame_failures} consecutive failures, "
                          f"attempt {self._reloc_attempts}/{self._max_reloc_attempts})")

                reloc_pos = None
                reloc_rot = None
                reloc_inliers = 0
                reloc_pts_2d = np.zeros((0, 2))
                reloc_pts_3d = np.zeros((0, 3))
                reloc_confs = np.zeros(0, dtype=np.float32)

                try:
                    reloc_pos, reloc_rot, reloc_inliers, reloc_pts_2d, reloc_pts_3d, reloc_confs, _, _ = \
                        self._localize_full_pipeline(
                            frame_id, image, h, w, verbose=verbose,
                            tracked_prior=self.prev_position,
                        )
                except (FirstFrameLocalizationError, InsufficientConfidentMatchesError) as e:
                    if verbose:
                        print(f"  Re-localization error: {e}")

                if reloc_pos is not None and reloc_inliers >= 30:
                    if verbose:
                        print(f"  RE-LOCALIZATION SUCCESS: "
                              f"pos=({reloc_pos[0]:.1f}, {reloc_pos[1]:.1f}, {reloc_pos[2]:.1f}), "
                              f"{reloc_inliers} inliers")
                    self.prev_position = reloc_pos
                    self.prev_R_c2w = reloc_rot
                    self.last_keyframe_position = reloc_pos.copy()
                    self.last_keyframe_time = frame_id
                    self._consecutive_frame_failures = 0
                    self._consecutive_expanding_failures = 0
                    self.avg_displacement = 0.0

                    self.tracker.set_keyframe(frame_id, image,
                                              reloc_pts_2d, reloc_pts_3d,
                                              reloc_confs)
                    self.last_keyframe_id = frame_id
                    self.frames_since_keyframe = 0
                    self._tracking_established = True

                    if results:
                        results[-1] = self._make_result(
                            frame_id, gt_pose, reloc_pos, reloc_rot,
                            is_keyframe=True, method="re-localized", start_time=start_time,
                            num_tracked_points=len(reloc_pts_2d),
                            num_inliers=reloc_inliers,
                        )
                else:
                    if verbose:
                        print(f"  Re-localization failed")

            # Early stop if max_keyframes reached
            if self.max_keyframes is not None:
                n_kf = sum(1 for r in results if r.is_keyframe)
                if n_kf >= self.max_keyframes:
                    if verbose:
                        print(f"\n  Reached max_keyframes={self.max_keyframes}, stopping early.")
                    break

        _prefetch_pool.shutdown(wait=False)
        return results

    # ------------------------------------------------------------------ #
    #  Delegation helpers                                                  #
    # ------------------------------------------------------------------ #

    def print_summary(self, results: List[FrameResult]):
        evl.print_summary(results)

    def plot_results(self, results: List[FrameResult], filename: str = "tracking_results.png"):
        vis.plot_results(results, filename, keyframe_min_points=self.keyframe_min_points, fig_ext=self.fig_ext)
