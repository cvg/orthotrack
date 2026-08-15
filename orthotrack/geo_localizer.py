"""Geo-localization for keyframes: coarse (first frame only) then fine.

**First keyframe** (no ``tracked_prior`` and no sensor prior)

1. **Stage 1 — coarse**

   a. Match against the full DOP (``full_dop_roi_detection``). If a region /
      correspondences are available, continue to (c). Otherwise go to (b).

   b. Exhaustive tile search. If a tile yields usable correspondences, go to
      (c). Otherwise raise `FirstFrameLocalizationError` (first frame cannot
      be localized).

   c. Estimate a rough camera pose from the correspondences (PnP where
      possible; optional intrinsics calibration when ``calibrate=True``).
      Derive the fine-stage **footprint centre** and **visible crop size**:
      when pose + DSM projection are reliable, use ``compute_visible_dop_crop``;
      otherwise use the horizontal spread of 3D DOP points from the matches.

2. **Stage 2 — fine**

   a. Build three DOP crops at ``0.7×``, ``1.0×``, and ``1.4×`` the base
      visible size, centred on the footprint centre from stage 1(c).

   b. Match with ``fine_matcher`` (batched; sequential on OOM or empty batch).

   c. Merge correspondences, optionally fine-calibrate intrinsics, then PnP.

**Later keyframes** (``tracked_prior`` set)

Stage 1 is skipped. Stage 2 uses the prior position and ``prev_R_c2w`` to
compute footprint centre and visible size on the DSM, then runs the same
fine matching and PnP as above.

The public entry point is ``localize_full_pipeline()``; configuration is
``LocalizationConfig``."""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from orthotrack import localization as loc
from orthotrack import tile_search
from orthotrack import crop_strategy as crop
from orthotrack.exceptions import (
    FirstFrameLocalizationError,
    InsufficientConfidentMatchesError,
    InvalidGeometryError,
    VisibleCropError,
)

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------


@dataclass
class LocalizationConfig:
    """References into the live pipeline (handlers, matchers, intrinsics)."""

    geo_handler: Any
    coarse_matcher: Any
    fine_matcher: Any
    intrinsics: Any
    num_matches: int = 3000
    confidence_threshold: float = 0.4
    confidence_min_count: int = 50
    pnp_reproj_threshold: float = 7.0
    use_prior: bool = False
    sensor_prior: Any = None


# Relative sizes for the three fine-stage DOP crops (stage 2a).
FINE_CROP_SCALES: Tuple[float, float, float] = (0.7, 1.0, 1.4)


# ---------------------------------------------------------------------------
#  Small helpers
# ---------------------------------------------------------------------------


def _fail() -> tuple:
    return (
        None,
        None,
        0,
        np.zeros((0, 2)),
        np.zeros((0, 3)),
        np.zeros(0, dtype=np.float32),
        None,
        None,
        False,
    )


def _matcher_label(m) -> Optional[str]:
    n = getattr(m, "name", None)
    if n:
        return str(n)
    cn = type(m).__name__
    return cn[:-7] if cn.endswith("Matcher") else cn


def _sequential_match(image, dop_images, matcher, num_matches) -> list:
    results = []
    for dop_img in dop_images:
        if dop_img is None:
            results.append(None)
            continue
        try:
            results.append(matcher.match(image, dop_img, num_matches=num_matches))
        except Exception:
            results.append(None)
    return results


def _update_intrinsics(intrinsics, calib: dict, verbose: bool = False, label: str = "") -> None:
    intrinsics.fx = calib["fx"]
    intrinsics.fy = calib["fy"]
    intrinsics.cx = calib["cx"]
    intrinsics.cy = calib["cy"]
    intrinsics.fov_vertical = calib["fov_vertical"]
    if verbose:
        print(
            f"  Intrinsics updated ({label}): FoV={calib['fov_vertical']:.1f}°, "
            f"fx={calib['fx']:.1f}, fy={calib['fy']:.1f}, "
            f"cx={calib['cx']:.1f}, cy={calib['cy']:.1f}"
        )


def _footprint_size_from_correspondences_xy(
    pts_3d: np.ndarray,
    margin: float = 1.15,
) -> Tuple[float, float, float]:
    """Footprint centre (UTM XY) and a single crop diameter (m) from matched ground points."""
    if len(pts_3d) < 10:
        raise FirstFrameLocalizationError(
            "Fewer than 10 3D correspondences; cannot estimate footprint from geometry."
        )
    cx = float(np.median(pts_3d[:, 0]))
    cy = float(np.median(pts_3d[:, 1]))
    ex = float(np.percentile(pts_3d[:, 0], 95) - np.percentile(pts_3d[:, 0], 5))
    ey = float(np.percentile(pts_3d[:, 1], 95) - np.percentile(pts_3d[:, 1], 5))
    size = float(max(ex, ey) * margin)
    return cx, cy, size


def _fine_crop_anchor_from_pose_or_geometry(
    cfg: LocalizationConfig,
    rough_position: np.ndarray,
    rough_R_c2w: Optional[np.ndarray],
    roi_3d: np.ndarray,
    h: int,
    w: int,
    *,
    prefer_pose_projection: bool,
    verbose: bool,
) -> Tuple[float, float, float]:
    """Footprint centre + base visible crop size (m) for stage 2a."""
    if prefer_pose_projection and rough_R_c2w is not None:
        try:
            cx, cy, sz = crop.compute_visible_dop_crop(
                rough_position,
                rough_R_c2w,
                (h, w),
                cfg.intrinsics.fov_vertical,
                cfg.geo_handler,
                verbose=verbose,
                K=cfg.intrinsics.K,
            )
            return float(cx), float(cy), float(sz)
        except (VisibleCropError, InvalidGeometryError):
            if verbose:
                print("  Visible crop from pose failed — using correspondence geometry.")
    return _footprint_size_from_correspondences_xy(roi_3d)


# ---------------------------------------------------------------------------
#  Stage 1 — first keyframe only (no prior pose)
# ---------------------------------------------------------------------------


@dataclass
class _CoarseStageState:
    """Everything produced by coarse localization before the fine stage."""

    rough_position: Optional[np.ndarray]
    rough_R_c2w: Optional[np.ndarray]
    roi_2d: np.ndarray
    roi_3d: np.ndarray
    roi_cf: np.ndarray
    fd_info: Optional[Dict[str, Any]]
    fd_match_ok: bool
    fd_position: Optional[np.ndarray]   # fd PnP result (may be None if PnP failed)
    fd_R_c2w_raw: Optional[np.ndarray]  # fd rotation (may be None)
    fd_inl_raw: int                     # fd PnP inlier count (0 if PnP failed)
    fd_med_conf: float                  # median confidence of fd matches
    fd_inl_idx: np.ndarray              # indices into fd_info["pts_2d"] that are PnP inliers
    ex_inl: int
    tile_kpts_q: Optional[np.ndarray]
    tile_kpts_dop: Optional[np.ndarray]
    tile_confs: Optional[np.ndarray]
    tile_crop: Optional[Tuple[float, float, float]]
    tile_accepted: bool
    tile_inl_idx: np.ndarray            # PnP inlier indices into tile_kpts_q
    tile_position: Optional[np.ndarray] # tile PnP camera position (may be None)
    tile_R_c2w: Optional[np.ndarray]    # tile PnP rotation (may be None)
    tile_score_map: Any  # list of (cx, cy, n_conf, n_raw) for all evaluated tiles
    calib_has_pts: bool
    coarse_calibrated: bool
    s3_inl: int
    calib_fov_vis: Optional[float]
    calib_candidates_vis: Any
    localization_strong: bool  # True when rough_position/R_c2w are from a successful PnP (fd, tile, or calib)
    intrinsics_updated: bool
    localization_failed: bool = False  # True when both fd and tile failed; caller will raise


def _run_coarse_stage(
    cfg: LocalizationConfig,
    image: np.ndarray,
    h: int,
    w: int,
    *,
    calibrate: bool,
    verbose: bool,
) -> _CoarseStageState:
    """Stage 1a → 1b → 1c for the first localized keyframe."""

    fd_info = loc.full_dop_roi_detection(
        image,
        cfg.geo_handler,
        cfg.coarse_matcher,
        num_matches=cfg.num_matches,
        fov_vertical=cfg.intrinsics.fov_vertical,
        verbose=verbose,
    )

    fd_match_ok = False
    rough_position: Optional[np.ndarray] = None
    rough_R_c2w: Optional[np.ndarray] = None
    roi_2d = np.zeros((0, 2))
    roi_3d = np.zeros((0, 3))
    roi_cf = np.zeros(0, dtype=np.float32)

    tile_kpts_q = tile_kpts_dop = tile_confs = None
    tile_crop = None
    tile_accepted = False
    tile_inl_idx: np.ndarray = np.zeros(0, dtype=np.int64)
    tile_position: Optional[np.ndarray] = None
    tile_R_c2w: Optional[np.ndarray] = None
    tile_score_map: list = []
    calib_has_pts = False
    ex_inl = 0
    calib: Optional[dict] = None

    fd_position: Optional[np.ndarray] = None
    fd_R_c2w_raw: Optional[np.ndarray] = None
    fd_inl_raw: int = 0
    fd_med_conf: float = 0.0
    fd_inl_idx: np.ndarray = None  # type: ignore[assignment]

    # --- 1a: full DOP -------------------------------------------------------
    if verbose:
        print("\n  ── Stage 1a: full-DOP match ──")

    fd_has_region = fd_info is not None and len(fd_info.get("pts_2d", [])) >= 20

    if fd_has_region:
        fd_2d = fd_info["pts_2d"].astype(np.float64)
        fd_3d = fd_info["pts_3d"].astype(np.float64)
        fd_cf = fd_info.get("confs", np.ones(len(fd_2d), dtype=np.float32)).astype(np.float32)

        if len(fd_2d) >= 30:
            fd_pos, _, _, fd_inl, fd_rot, _fd_inl_idx = loc.localize_from_correspondences(
                fd_2d,
                fd_3d,
                (h, w),
                cfg.intrinsics.fov_vertical,
                verbose=False,
                K=cfg.intrinsics.K,
            )
            fd_med_conf = float(np.median(fd_cf))
            # Always store fd PnP result for visualization, regardless of acceptance
            if fd_pos is not None:
                fd_position = fd_pos
                fd_R_c2w_raw = fd_rot
                fd_inl_raw = int(fd_inl)
                fd_inl_idx = _fd_inl_idx
            if fd_pos is not None and fd_med_conf > 0.4:
                rough_position = fd_pos
                rough_R_c2w = fd_rot
                roi_2d = fd_2d
                roi_3d = fd_3d
                roi_cf = fd_cf
                fd_match_ok = True
                # When calibration mode is active, tag fd correspondences as
                # calibration candidates so Stage 1c can run the FoV sweep.
                calib_has_pts = calibrate
                if verbose:
                    print(
                        f"  Full-DOP: strong PnP — {fd_inl} inliers, "
                        f"med_conf={fd_med_conf:.2f}, "
                        f"pos=({fd_pos[0]:.0f}, {fd_pos[1]:.0f}, {fd_pos[2]:.0f})"
                    )
            elif verbose:
                reason = (f"low conf (med={fd_med_conf:.2f})"
                          if (fd_pos is not None and fd_med_conf <= 0.4)
                          else f"PnP weak ({fd_inl} inliers)")
                print(f"  Full-DOP: {reason} — falling back to tile search")
        elif verbose:
            print(
                f"  Full-DOP: too few 3D correspondences ({len(fd_2d)}) — skipping"
            )

        if not fd_match_ok:
            roi_2d = fd_2d
            roi_3d = fd_3d
            roi_cf = fd_cf
    elif verbose:
        print("  Full-DOP: no region — tile search (stage 1b)")

    # --- 1b: tile search if full-DOP did not succeed (PnP failed OR low conf) ----
    if not fd_match_ok:
        if verbose:
            print("\n  ── Stage 1b: tile search (coarse-to-fine) ──")

        ex_pos, ex_2d, ex_3d, ex_cf, ex_inl, ex_rot, ex_crop, _ex_inl_mask, ex_tile_scores = (
            tile_search.coarse_to_fine_tile_localization(
                image,
                cfg.geo_handler,
                cfg.coarse_matcher,
                image_size=(h, w),
                fov_vertical=cfg.intrinsics.fov_vertical,
                num_matches=cfg.num_matches,
                confidence_threshold=cfg.confidence_threshold,
                min_inliers=30,
                K=cfg.intrinsics.K,
                verbose=verbose,
                return_best_on_pnp_fail=calibrate,
            )
        )

        tile_kpts_q = ex_2d
        tile_kpts_dop = ex_3d
        tile_confs = ex_cf
        tile_crop = ex_crop
        tile_accepted = ex_pos is not None and ex_crop is not None
        tile_inl_idx = _ex_inl_mask  # boolean mask or index array from tile PnP
        tile_position = ex_pos       # PnP camera position (None if search failed)
        tile_R_c2w = ex_rot          # PnP rotation (None if search failed)
        tile_score_map = ex_tile_scores if ex_tile_scores is not None else []
        # Enable calibration sweep whenever we have 2D-3D correspondences,
        # regardless of whether tile PnP succeeded or failed — the sweep
        # must run to set the correct FoV for the fine stage.
        calib_has_pts = (
            calibrate
            and ex_crop is not None
            and ex_2d is not None
            and len(ex_2d) >= 30
        )

        if ex_pos is None and ex_crop is None:
            if verbose:
                print("  Tile search: no valid tile — cannot localize first frame.")
            # Return partial state so the caller can still generate the debug visualization
            # before re-raising. localization_failed=True tells the caller to raise.
            return _CoarseStageState(
                rough_position=None,
                rough_R_c2w=None,
                roi_2d=np.zeros((0, 2)),
                roi_3d=np.zeros((0, 3)),
                roi_cf=np.zeros(0, dtype=np.float32),
                fd_info=fd_info,
                fd_match_ok=False,
                fd_position=fd_position,
                fd_R_c2w_raw=fd_R_c2w_raw,
                fd_inl_raw=fd_inl_raw,
                fd_med_conf=fd_med_conf,
                fd_inl_idx=fd_inl_idx if fd_inl_idx is not None else np.zeros(0, dtype=np.int64),
                ex_inl=0,
                tile_kpts_q=tile_kpts_q,
                tile_kpts_dop=tile_kpts_dop,
                tile_confs=tile_confs,
                tile_crop=None,
                tile_accepted=False,
                tile_inl_idx=np.zeros(0, dtype=np.int64),
                tile_position=None,
                tile_R_c2w=None,
                tile_score_map=tile_score_map,
                calib_has_pts=False,
                coarse_calibrated=False,
                s3_inl=0,
                calib_fov_vis=None,
                calib_candidates_vis=None,
                localization_strong=False,
                intrinsics_updated=False,
                localization_failed=True,
            )

        rough_position = ex_pos
        rough_R_c2w = ex_rot
        roi_2d = ex_2d if ex_2d is not None else np.zeros((0, 2))
        roi_3d = ex_3d if ex_3d is not None else np.zeros((0, 3))
        roi_cf = ex_cf if ex_cf is not None else np.zeros(0, dtype=np.float32)

        if tile_accepted and verbose:
            n_full = len(ex_2d) if ex_2d is not None else 0
            print(
                f"  Tile search: OK — {ex_inl} PnP inliers, "
                f"{n_full} correspondences, centre=({ex_crop[0]:.0f}, {ex_crop[1]:.0f})"
            )
        elif calib_has_pts and verbose:
            print(
                f"  Tile search: PnP failed — using best tile "
                f"({ex_crop[0]:.0f}, {ex_crop[1]:.0f}) for calibration ({len(roi_2d)} pts)"
            )

    # --- 1c: optional calibration + rough PnP -----------------------------
    if verbose:
        print("\n  ── Stage 1c: coarse pose + correspondences ──")

    calib_fov_vis = None
    calib_candidates_vis = None
    s3_inl = 0
    coarse_calibrated = False

    run_coarse_calib = calibrate and calib_has_pts and len(roi_2d) >= 30
    if run_coarse_calib:
        calib = loc.calibrate_intrinsics(
            roi_2d,
            roi_3d,
            (h, w),
            verbose=verbose,
            prefer_reproj=True,
        )
        if calib is not None:
            calib_fov_vis = calib.get("fov_vertical")
            calib_candidates_vis = calib.get("candidates")
            _update_intrinsics(cfg.intrinsics, calib, verbose=verbose, label="coarse calib")
            coarse_calibrated = True
            rvec_c = calib.get("rvec")
            tvec_c = calib.get("tvec")
            centroid_c = calib.get("centroid")
            if rvec_c is not None and tvec_c is not None and centroid_c is not None:
                import cv2 as _cv2

                R_c, _ = _cv2.Rodrigues(rvec_c)
                calib_pos = -R_c.T @ tvec_c.flatten() + centroid_c
                rough_position = calib_pos
                rough_R_c2w = R_c.T
                if verbose:
                    print(
                        f"  Coarse calib pose: ({calib_pos[0]:.0f}, {calib_pos[1]:.0f}, {calib_pos[2]:.0f}), "
                        f"{calib.get('num_inliers', 0)} inliers"
                    )

    # Report inliers from whichever stage provided the final pose — no merged PnP.
    # Stage 1a and 1b each run their own PnP on their own correspondences; merging
    # would just add noise from the stage that failed.
    if coarse_calibrated and calib is not None:
        s3_inl = int(calib.get("num_inliers", 0))
    elif tile_accepted:
        s3_inl = int(ex_inl)
    elif fd_match_ok:
        s3_inl = int(fd_inl_raw)

    if verbose and rough_position is not None:
        rp = rough_position
        src = "calib" if coarse_calibrated else ("tile" if tile_accepted else "fd")
        print(f"  Coarse pose ({src}): ({rp[0]:.0f}, {rp[1]:.0f}, {rp[2]:.0f}), {s3_inl} inliers")

    # At this point localization_failed=True was already returned if both fd and tile
    # failed, so rough_position/rough_R_c2w are guaranteed non-None here.
    # The pose may come from fd PnP, tile PnP, or calibration sweep — all are
    # considered "strong" for the purpose of projecting the visible footprint.
    localization_strong = rough_position is not None and rough_R_c2w is not None

    return _CoarseStageState(
        rough_position=rough_position,
        rough_R_c2w=rough_R_c2w,
        roi_2d=roi_2d,
        roi_3d=roi_3d,
        roi_cf=roi_cf,
        fd_info=fd_info,
        fd_match_ok=fd_match_ok,
        fd_position=fd_position,
        fd_R_c2w_raw=fd_R_c2w_raw,
        fd_inl_raw=fd_inl_raw,
        fd_med_conf=fd_med_conf,
        fd_inl_idx=fd_inl_idx if fd_inl_idx is not None else np.zeros(0, dtype=np.int64),
        ex_inl=int(ex_inl),
        tile_kpts_q=tile_kpts_q,
        tile_kpts_dop=tile_kpts_dop,
        tile_confs=tile_confs,
        tile_crop=tile_crop,
        tile_accepted=tile_accepted,
        tile_inl_idx=tile_inl_idx if tile_inl_idx is not None else np.zeros(0, dtype=np.int64),
        tile_position=tile_position,
        tile_R_c2w=tile_R_c2w,
        tile_score_map=tile_score_map,
        calib_has_pts=calib_has_pts,
        coarse_calibrated=coarse_calibrated,
        s3_inl=int(s3_inl),
        calib_fov_vis=calib_fov_vis,
        calib_candidates_vis=calib_candidates_vis,
        localization_strong=localization_strong,
        intrinsics_updated=coarse_calibrated,
    )


# ---------------------------------------------------------------------------
#  Stage 2 — fine (all keyframes once a rough pose exists)
# ---------------------------------------------------------------------------


def _run_fine_stage(
    cfg: LocalizationConfig,
    image: np.ndarray,
    h: int,
    w: int,
    *,
    anchor_cx: float,
    anchor_cy: float,
    base_crop_m: float,
    rough_position: np.ndarray,
    calibrate: bool,
    calib_has_pts: bool,
    coarse_calibrated: bool,
    frame_id: int,
    debug_vis: Any,
    save_crop_vis: bool,
    verbose: bool,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[Tuple[float, float, float]],
    Optional[list],
    bool,
]:
    """Stage 2a–2c: three crops, merge, optional fine calib, PnP."""

    intrinsics_updated = False
    crop_specs = [
        (anchor_cx, anchor_cy, base_crop_m * FINE_CROP_SCALES[0]),
        (anchor_cx, anchor_cy, base_crop_m * FINE_CROP_SCALES[1]),
        (anchor_cx, anchor_cy, base_crop_m * FINE_CROP_SCALES[2]),
    ]
    if verbose:
        sizes = [f"{sz:.0f}m" for _, _, sz in crop_specs]
        print(f"\n  ── Stage 2a: fine crops ({', '.join(sizes)}) ──")

    crop_tiles: List[Any] = []
    dop_images: List[Any] = []
    for cx, cy, sz in crop_specs:
        tile = cfg.geo_handler.crop_dop(cx, cy, sz)
        crop_tiles.append(tile)
        dop_images.append(tile.data if tile is not None else None)

    valid_dop = [d for d in dop_images if d is not None]
    valid_idx = [i for i, d in enumerate(dop_images) if d is not None]
    crop_vis_data: Optional[list] = [] if save_crop_vis else None

    if not valid_dop:
        if verbose:
            print("  Stage 2b: no valid DOP tiles")
        return _fail()

    if verbose:
        print("\n  ── Stage 2b: fine matcher ──")

    try:
        match_results_valid = cfg.fine_matcher.match_batch(
            image, valid_dop, num_matches_per_crop=cfg.num_matches
        )
    except Exception:
        if verbose:
            print("    Batch OOM — sequential matching")
        torch.cuda.empty_cache()
        match_results_valid = _sequential_match(image, valid_dop, cfg.fine_matcher, cfg.num_matches)

    if all(mr is None or len(mr.kpts_query) < 10 for mr in match_results_valid):
        if verbose:
            print("    Empty batch — sequential matching")
        torch.cuda.empty_cache()
        match_results_valid = _sequential_match(image, valid_dop, cfg.fine_matcher, cfg.num_matches)

    match_results_all: List[Any] = [None] * len(crop_specs)
    for pos, idx in enumerate(valid_idx):
        match_results_all[idx] = match_results_valid[pos]

    all_pts_2d: List[np.ndarray] = []
    all_pts_3d: List[np.ndarray] = []
    all_confs: List[np.ndarray] = []
    per_crop_pts2d: List[Optional[np.ndarray]] = [None] * len(crop_specs)
    per_crop_pts3d: List[Optional[np.ndarray]] = [None] * len(crop_specs)

    for i_crop, (mr, tile, (cx, cy, sz)) in enumerate(zip(match_results_all, crop_tiles, crop_specs)):
        if mr is None or tile is None or len(mr.kpts_query) < 10:
            continue
        kpts_q = mr.kpts_query
        kpts_d = mr.kpts_dop
        conf = mr.confidences
        utm_xs, utm_ys = tile.pixel_to_utm_batch(kpts_d[:, 0], kpts_d[:, 1])
        zs = loc.sample_full_dsm_batch(cfg.geo_handler, utm_xs, utm_ys)
        valid = loc.compute_valid_mask(tile, kpts_d, zs)
        if valid.any():
            _c2d = kpts_q[valid].astype(np.float64)
            _c3d = np.column_stack([utm_xs[valid], utm_ys[valid], zs[valid]]).astype(np.float64)
            all_pts_2d.append(_c2d)
            all_pts_3d.append(_c3d)
            all_confs.append(conf[valid].astype(np.float32))
            per_crop_pts2d[i_crop] = _c2d
            per_crop_pts3d[i_crop] = _c3d
            if verbose:
                print(f"    Crop {i_crop + 1}: {int(valid.sum())} matches")
        if crop_vis_data is not None:
            crop_vis_data.append(
                {
                    "dop_tile": tile,
                    "kpts_query": kpts_q,
                    "kpts_dop": kpts_d,
                    "confidences": conf,
                    "name": f"Crop {i_crop + 1} ({sz:.0f}m)",
                }
            )

    if not all_pts_2d:
        if verbose:
            print("  Stage 2b: no lifted matches")
        return _fail()

    if verbose:
        print("\n  ── Stage 2c: merge + PnP ──")

    merged_2d = np.vstack(all_pts_2d)
    merged_3d = np.vstack(all_pts_3d)
    merged_cf = np.concatenate(all_confs)
    _, uniq = np.unique(np.round(merged_2d).astype(int), axis=0, return_index=True)
    merged_2d, merged_3d, merged_cf = merged_2d[uniq], merged_3d[uniq], merged_cf[uniq]

    conf_mask = merged_cf >= cfg.confidence_threshold
    if conf_mask.sum() < cfg.confidence_min_count:
        raise InsufficientConfidentMatchesError(
            f"fine merge: {int(conf_mask.sum())} pts >= confidence_threshold "
            f"(need {cfg.confidence_min_count})."
        )
    merged_2d = merged_2d[conf_mask]
    merged_3d = merged_3d[conf_mask]
    merged_cf = merged_cf[conf_mask]

    if verbose:
        print(f"  Merged: {len(merged_2d)} pts from {len(all_pts_2d)} crops")

    should_fine_calibrate = (
        calibrate
        and calib_has_pts
        and len(merged_2d) >= 50
        # If the coarse stage already produced a trusted FoV calibration on
        # the full ROI (typically 100+ inliers), the fine recalibration must
        # not be allowed to override it from a much sparser merged set:
        # with only ~150–500 fine points (typical when the scene has tall
        # narrow buildings, e.g. ETH MainBuilding) the reproj-first selector
        # picks a slightly different FoV at the edge of the sweep range,
        # and the resulting fx is incompatible with the actual camera, which
        # makes Fine PnP fail at the very next step.
        and not coarse_calibrated
    )
    if should_fine_calibrate:
        fine_calib = loc.calibrate_intrinsics(
            merged_2d,
            merged_3d,
            (h, w),
            fov_candidates=None,
            verbose=verbose,
            prefer_reproj=True,
        )
        if fine_calib is not None:
            _update_intrinsics(cfg.intrinsics, fine_calib, verbose=verbose, label="fine recalib")
            intrinsics_updated = True

    if verbose:
        z_min = float(merged_3d[:, 2].min()) if len(merged_3d) else 0.0
        z_max = float(merged_3d[:, 2].max()) if len(merged_3d) else 0.0
        print(
            f"  Fine PnP input: {len(merged_2d)} pts, Z=[{z_min:.0f}, {z_max:.0f}], "
            f"FoV={cfg.intrinsics.fov_vertical:.1f}°"
        )

    position, pnp_2d, pnp_3d, num_inliers, est_rotation, pnp_idx = loc.localize_from_correspondences(
        merged_2d,
        merged_3d,
        (h, w),
        cfg.intrinsics.fov_vertical,
        reproj_threshold=cfg.pnp_reproj_threshold,
        verbose=verbose,
        K=cfg.intrinsics.K,
    )

    if position is None:
        if verbose:
            print("  Fine PnP: failed")
        return _fail()

    if verbose:
        print(
            f"  Fine PnP: ({position[0]:.0f}, {position[1]:.0f}, {position[2]:.0f}), "
            f"{num_inliers} inliers"
        )

    pnp_cf_out = merged_cf[pnp_idx] if len(pnp_idx) == len(pnp_2d) else merged_cf[:len(pnp_2d)]
    pts_2d_out, pts_3d_out, confs_out = pnp_2d, pnp_3d, pnp_cf_out

    if debug_vis is not None:
        debug_vis.fine_stage(
            frame_id,
            image,
            crop_specs=crop_specs,
            crop_tiles=crop_tiles,
            match_results=match_results_all,
            pts_2d=pts_2d_out,
            pts_3d=pts_3d_out,
            confs=confs_out,
            position=position,
            R_c2w=est_rotation,
            num_inliers=num_inliers,
            recalib_fov=cfg.intrinsics.fov_vertical if should_fine_calibrate else None,
            recalib_was_run=should_fine_calibrate,
            fine_matcher_name=_matcher_label(cfg.fine_matcher),
            per_crop_pts2d=per_crop_pts2d,
            per_crop_pts3d=per_crop_pts3d,
            reproj_threshold=cfg.pnp_reproj_threshold,
        )

    # Use the actual fine-stage anchor as the output crop spec — this was
    # computed from DSM projection (or correspondence geometry) and is already
    # geometry-correct, unlike a naive altitude*tan(fov) estimate.
    crop_spec = (anchor_cx, anchor_cy, base_crop_m)

    return (
        position,
        est_rotation,
        num_inliers,
        pts_2d_out,
        pts_3d_out,
        confs_out,
        crop_spec,
        crop_vis_data,
        intrinsics_updated,
    )


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


def localize_full_pipeline(
    frame_id: int,
    image: np.ndarray,
    h: int,
    w: int,
    cfg: LocalizationConfig,
    tracked_prior: Optional[np.ndarray] = None,
    tracked_pts_3d: Optional[np.ndarray] = None,
    calibrate: bool = False,
    debug_vis=None,
    prev_R_c2w: Optional[np.ndarray] = None,
    save_crop_vis: bool = False,
    verbose: bool = False,
) -> tuple:
    """
    Run geo-localization.

    Returns
    -------
    position, R_c2w, num_inliers, pts_2d, pts_3d, confs, crop_spec,
    crop_vis_data, intrinsics_updated"""
    intrinsics_updated = False
    rough_position: Optional[np.ndarray] = None
    rough_R_c2w: Optional[np.ndarray] = None
    roi_2d = np.zeros((0, 2))
    roi_3d = np.zeros((0, 3))
    roi_cf = np.zeros(0, dtype=np.float32)
    fd_info: Optional[Dict] = None
    fd_match_ok = False
    s3_inl = 0
    tile_kpts_q = tile_kpts_dop = tile_confs = None
    tile_crop = None
    tile_accepted = False
    tile_score_map: list = []
    calib_has_pts = False
    coarse_calibrated = False
    calib_fov_vis = None
    calib_candidates_vis = None
    localization_strong = False

    # --- Prior: skip stage 1 -------------------------------------------------
    if cfg.use_prior and cfg.sensor_prior is not None:
        prior_pos = cfg.sensor_prior.get_position(frame_id)
        prior_R_c2w = cfg.sensor_prior.get_rotation_matrix(frame_id)
        if verbose:
            print(
                f"  Prior: pos=({float(prior_pos[0]):.0f}, {float(prior_pos[1]):.0f}, "
                f"{float(prior_pos[2]):.0f})"
            )
        rough_position = np.asarray(prior_pos, dtype=np.float64)
        rough_R_c2w = prior_R_c2w
        localization_strong = True

    elif tracked_prior is not None:
        if verbose:
            print(
                f"  Tracked prior: pos=({float(tracked_prior[0]):.0f}, "
                f"{float(tracked_prior[1]):.0f}, {float(tracked_prior[2]):.0f})"
            )
        rough_position = np.asarray(tracked_prior, dtype=np.float64)
        rough_R_c2w = prev_R_c2w
        localization_strong = True
        # Use tracked 3D points as correspondence fallback for anchor computation
        if tracked_pts_3d is not None and len(tracked_pts_3d) >= 10:
            roi_3d = tracked_pts_3d

    else:
        st = _run_coarse_stage(cfg, image, h, w, calibrate=calibrate, verbose=verbose)
        rough_position = st.rough_position
        rough_R_c2w = st.rough_R_c2w
        roi_2d, roi_3d, roi_cf = st.roi_2d, st.roi_3d, st.roi_cf
        fd_info = st.fd_info
        fd_match_ok = st.fd_match_ok
        s3_inl = st.s3_inl
        tile_kpts_q, tile_kpts_dop, tile_confs = st.tile_kpts_q, st.tile_kpts_dop, st.tile_confs
        tile_crop, tile_accepted = st.tile_crop, st.tile_accepted
        tile_score_map = st.tile_score_map
        calib_has_pts = st.calib_has_pts
        coarse_calibrated = st.coarse_calibrated
        calib_fov_vis = st.calib_fov_vis
        calib_candidates_vis = st.calib_candidates_vis
        localization_strong = st.localization_strong
        intrinsics_updated = st.intrinsics_updated

        if debug_vis is not None:
            debug_vis.coarse_stage(
                frame_id,
                image,
                all_kpts_q=fd_info.get("kpts_query") if fd_info else None,
                all_kpts_dop=fd_info.get("kpts_dop") if fd_info else None,
                all_confs=fd_info.get("confidences") if fd_info else None,
                utm_x=fd_info.get("utm_x") if fd_info else None,
                utm_y=fd_info.get("utm_y") if fd_info else None,
                fd_accepted=fd_match_ok,
                fd_2d=fd_info.get("pts_2d") if fd_info else None,
                fd_3d=fd_info.get("pts_3d") if fd_info else None,
                fd_cf=fd_info.get("confs") if fd_info else None,
                fd_position=st.fd_position,
                fd_R_c2w=st.fd_R_c2w_raw,
                fd_inl=st.fd_inl_raw,
                fd_med_conf=st.fd_med_conf,
                fd_inl_idx=st.fd_inl_idx,
                tile_kpts_q=tile_kpts_q,
                tile_kpts_dop=tile_kpts_dop,
                tile_confs=tile_confs,
                tile_crop=tile_crop,
                tile_accepted=tile_accepted,
                tile_inl_idx=st.tile_inl_idx,
                tile_position=st.tile_position,
                tile_R_c2w=st.tile_R_c2w,
                tile_score_map=tile_score_map,
                confidence_threshold=cfg.confidence_threshold,
                confidence_min_count=cfg.confidence_min_count,
                rough_position=rough_position,
                rough_R_c2w=rough_R_c2w,
                roi_inl=s3_inl,
                calib_fov=calib_fov_vis,
                roi_2d=roi_2d,
                roi_3d=roi_3d,
                roi_cf=roi_cf,
                coarse_matcher_name=_matcher_label(cfg.coarse_matcher),
            )

        if st.localization_failed:
            raise FirstFrameLocalizationError(
                "Exhaustive tile search found no usable tile; first frame cannot be localized."
            )

    if rough_position is None or rough_R_c2w is None:
        if verbose:
            print("  No rough pose — cannot run stage 2")
        return _fail()

    used_prior_pose = (cfg.use_prior and cfg.sensor_prior is not None) or (tracked_prior is not None)
    if not used_prior_pose and len(roi_3d) < 10:
        raise FirstFrameLocalizationError(
            "Coarse stage produced fewer than 10 3D points; cannot anchor fine crops."
        )

    if verbose:
        print("\n  ── Footprint anchor for stage 2 ──")

    # Use the same strategy for both first-frame and intermediate keyframes:
    # try pose-based footprint, fallback to correspondence XY positions.
    anchor_cx, anchor_cy, base_m = _fine_crop_anchor_from_pose_or_geometry(
        cfg,
        rough_position,
        rough_R_c2w,
        roi_3d,
        h,
        w,
        prefer_pose_projection=(rough_R_c2w is not None),
        verbose=verbose,
    )

    fine_ret = _run_fine_stage(
        cfg,
        image,
        h,
        w,
        anchor_cx=anchor_cx,
        anchor_cy=anchor_cy,
        base_crop_m=base_m,
        rough_position=rough_position,
        calibrate=calibrate,
        calib_has_pts=calib_has_pts,
        coarse_calibrated=coarse_calibrated,
        frame_id=frame_id,
        debug_vis=debug_vis,
        save_crop_vis=save_crop_vis,
        verbose=verbose,
    )

    pos, rot, inl, p2, p3, cf, crop_spec, vis, fine_intrinsics_updated = fine_ret
    intrinsics_updated = intrinsics_updated or fine_intrinsics_updated

    # Fallback: when the fine stage fails but the coarse stage already
    # produced a usable pose (typical of low-altitude / oblique sequences
    # such as ETH MainBuilding where the fine matcher OOMs or only lifts a
    # handful of valid 3D points after DSM masking), fall back to the coarse
    # pose instead of failing the whole sequence.
    #
    # Accept either a self-calibrated coarse pose OR known/bundled intrinsics
    # (demo COLMAP / FoV json). Require a modest inlier count — tile-search
    # poses on ETH often land in the 40–80 inlier range.
    has_known_intrinsics = (
        cfg.intrinsics is not None
        and (
            getattr(cfg.intrinsics, "fx", None) is not None
            or float(getattr(cfg.intrinsics, "fov_vertical", 0.0) or 0.0) > 0.0
        )
    )
    if (
        pos is None
        and not used_prior_pose
        and (coarse_calibrated or has_known_intrinsics)
        and rough_position is not None
        and rough_R_c2w is not None
        and int(s3_inl) >= 40
        and len(roi_2d) >= 30
    ):
        if verbose:
            print(
                f"  Fine stage failed — falling back to coarse pose "
                f"({int(s3_inl)} inliers, FoV={cfg.intrinsics.fov_vertical:.1f}°, "
                f"calibrated={coarse_calibrated})."
            )
        pos = rough_position
        rot = rough_R_c2w
        inl = int(s3_inl)
        p2 = roi_2d
        p3 = roi_3d
        cf = roi_cf
        crop_spec = (float(anchor_cx), float(anchor_cy), float(base_m))

    return (pos, rot, inl, p2, p3, cf, crop_spec, vis, intrinsics_updated)
