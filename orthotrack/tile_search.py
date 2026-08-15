"""
Tile-based global search over orthophoto DOP for initial localization.

Two strategies:
  - ``coarse_to_fine_tile_localization``: Multi-level pyramid (2^n grids)
  - ``exhaustive_tile_localization``: Fixed-size tiles with overlap"""

import numpy as np
import torch
import time as _time
from typing import Tuple, Optional
from tqdm import tqdm as _tqdm

from utils.geo import GeoTIFFHandler, SequenceGeoHandler
from orthotrack.matchers.base_matcher import BaseMatcher
from orthotrack.crop_strategy import get_intrinsics


def _get_loc_helpers():
    """Late import to avoid circular dependency with localization.py."""
    from orthotrack import localization as _loc
    return (
        _loc.sample_full_dsm_batch,
        _loc.compute_valid_mask,
        _loc.get_full_dop_image,
        _loc.localize_from_correspondences,
        _loc._mean_reproj_error,
    )
def coarse_to_fine_tile_localization(
    image: np.ndarray,
    geo_handler,
    matcher: BaseMatcher,
    image_size: Tuple[int, int],
    fov_vertical: float,
    num_matches: int = 3000,
    confidence_threshold: float = 0.4,
    min_inliers: int = 30,
    K: np.ndarray = None,
    verbose: bool = False,
    batch_size: int = 8,
    return_best_on_pnp_fail: bool = False,
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray, int, Optional[np.ndarray], Optional[Tuple[float, float, float]], np.ndarray, list]:
    """Coarse-to-fine tile localization using a pyramid of tile sizes.

    Divides the DOP into 2^n × 2^n grids for n = 1, 2, ..., down to the
    DOP's native pixel resolution. At each level all tiles are
    matched in batch; a tile is *accepted* when ``median_conf >= confidence_threshold``
    AND PnP succeeds with at least ``min_inliers`` inliers.  The search stops
    as soon as any tile is accepted at the current level.

    Returns the same tuple as ``exhaustive_tile_localization``."""
    (sample_full_dsm_batch, compute_valid_mask, get_full_dop_image,
     localize_from_correspondences, _mean_reproj_error) = _get_loc_helpers()

    fail = (None, np.zeros((0, 2)), np.zeros((0, 3)),
            np.zeros(0, dtype=np.float32), 0, None, None,
            np.zeros(0, dtype=bool), [])

    if isinstance(geo_handler, SequenceGeoHandler):
        _b = geo_handler.dop_bounds  # list [min_x, min_y, max_x, max_y]
        min_x, min_y, max_x, max_y = float(_b[0]), float(_b[1]), float(_b[2]), float(_b[3])
    elif isinstance(geo_handler, GeoTIFFHandler):
        _b = geo_handler.dop_bounds  # rasterio BoundingBox
        min_x, min_y, max_x, max_y = float(_b.left), float(_b.bottom), float(_b.right), float(_b.top)
    else:
        return fail

    dop_w = max_x - min_x
    dop_h = max_y - min_y

    # Build levels: n=1 → 2×2, n=2 → 4×4, ... until tile size reaches
    # approximately one DOP pixel in the smaller map dimension.
    dop_gsd = float(getattr(geo_handler, "dop_resolution", 0.0) or 0.0)
    if dop_gsd <= 0.0:
        dop_img = get_full_dop_image(geo_handler)
        if dop_img is not None:
            h_px, w_px = dop_img.shape[:2]
            dop_gsd = min(dop_w / max(w_px, 1), dop_h / max(h_px, 1))

    levels = []
    n = 1
    while True:
        grid = 2 ** n
        tile_w = dop_w / grid
        tile_h = dop_h / grid
        tile_size = min(tile_w, tile_h)
        if dop_gsd > 0.0 and tile_size < dop_gsd:
            break
        levels.append((n, grid, tile_w, tile_h))
        n += 1
    # Ensure at least one level
    if not levels:
        grid = 2
        levels.append((1, grid, dop_w / grid, dop_h / grid))

    if verbose:
        print(f"  Coarse-to-fine tile search: {len(levels)} levels, "
              f"DOP={dop_w:.0f}×{dop_h:.0f}m")

    all_tile_scores: list = []
    t0 = _time.time()
    prev_best_center = None  # best-scoring tile center from previous level

    for level_n, grid, tile_w, tile_h in levels:
        # Generate tile centres with 50% overlap (stride = tile_size / 2).
        # For a grid of `grid` non-overlapping tiles we get 2*grid-1 centres per
        # dimension, so e.g. a 2×2 base grid → 3×3 = 9 overlapping tiles.
        stride_x = tile_w / 2.0
        stride_y = tile_h / 2.0
        n_x = 2 * grid - 1
        n_y = 2 * grid - 1
        xs = min_x + tile_w / 2.0 + np.arange(n_x) * stride_x
        ys = min_y + tile_h / 2.0 + np.arange(n_y) * stride_y

        # Clamp centres so the tile stays inside the DOP
        xs = xs[(xs - tile_w / 2 >= min_x - 1e-3) & (xs + tile_w / 2 <= max_x + 1e-3)]
        ys = ys[(ys - tile_h / 2 >= min_y - 1e-3) & (ys + tile_h / 2 <= max_y + 1e-3)]
        tile_size = min(tile_w, tile_h)  # square tile (use smaller dimension to stay within DOP)

        tiles_this_level = [(float(cx), float(cy)) for cx in xs for cy in ys]
        # Sort tiles by proximity to previous level's best center (most promising first)
        if prev_best_center is not None:
            _bcx, _bcy = prev_best_center
            tiles_this_level.sort(key=lambda t: (t[0] - _bcx) ** 2 + (t[1] - _bcy) ** 2)
        if verbose:
            print(f"  Level {level_n}: {grid}×{grid} base grid (50% overlap), "
                  f"tile_size={tile_w:.0f}×{tile_h:.0f}m  ({len(tiles_this_level)} tiles, "
                  f"{len(xs)}×{len(ys)} centres)")

        # Pre-crop DOP for this level
        tile_data = []
        for cx, cy in tiles_this_level:
            dop_tile = geo_handler.crop_dop(cx, cy, max(tile_w, tile_h))
            if dop_tile is not None:
                tile_data.append((cx, cy, dop_tile))

        level_candidates = []
        _level_start_idx = len(all_tile_scores)

        pbar = _tqdm(total=len(tile_data),
                     desc=f"Level {level_n} ({grid}×{grid})", disable=not verbose)
        for batch_start in range(0, len(tile_data), batch_size):
            batch = tile_data[batch_start:batch_start + batch_size]
            dop_images = [td[2].data for td in batch]

            try:
                match_results = matcher.match_batch(
                    image, dop_images, num_matches_per_crop=num_matches)
            except Exception:
                if verbose:
                    print(f"    OOM at batch {batch_start}, falling back to sequential")
                torch.cuda.empty_cache()
                match_results = []
                for dop_img in dop_images:
                    try:
                        match_results.append(
                            matcher.match(image, dop_img, num_matches=num_matches))
                    except Exception:
                        match_results.append(None)

            _ts_crop = max(tile_w, tile_h)
            batch_candidates = []
            for (cx, cy, dop_tile), match_result in zip(batch, match_results):
                pbar.update(1)
                if match_result is None:
                    all_tile_scores.append((float(cx), float(cy), 0.0, 0, float(tile_w), float(tile_h), None))
                    continue

                kpts_q = match_result.kpts_query
                kpts_d = match_result.kpts_dop
                conf = match_result.confidences

                if len(kpts_q) == 0:
                    all_tile_scores.append((float(cx), float(cy), 0.0, 0, float(tile_w), float(tile_h), None))
                    continue

                utm_xs, utm_ys = dop_tile.pixel_to_utm_batch(kpts_d[:, 0], kpts_d[:, 1])
                zs = sample_full_dsm_batch(geo_handler, utm_xs, utm_ys)
                valid = compute_valid_mask(dop_tile, kpts_d, zs)

                if not valid.any():
                    all_tile_scores.append((float(cx), float(cy), 0.0, 0, float(tile_w), float(tile_h), None))
                    continue

                pts_2d = kpts_q[valid].astype(np.float64)
                pts_3d = np.column_stack([utm_xs[valid], utm_ys[valid], zs[valid]]).astype(np.float64)
                confs_v = conf[valid].astype(np.float32)

                med_conf = float(np.median(confs_v))
                n_raw = int(valid.sum())
                # Store subsampled UTM correspondences for visualization (max 500 pts)
                _n_vis = min(n_raw, 500)
                _vis_idx = (np.random.choice(n_raw, _n_vis, replace=False)
                            if n_raw > _n_vis else np.arange(n_raw))
                _kpts_vis = np.column_stack([utm_xs[valid][_vis_idx],
                                             utm_ys[valid][_vis_idx],
                                             confs_v[_vis_idx].astype(np.float64)])
                # 7-tuple: (cx, cy, med_conf, n_raw, tile_w, tile_h, kpts_vis)
                all_tile_scores.append((float(cx), float(cy), med_conf, n_raw,
                                        float(tile_w), float(tile_h), _kpts_vis))

                max_conf = float(np.max(confs_v))
                # Admit tile for PnP if median OR max confidence exceeds threshold.
                if med_conf < confidence_threshold and max_conf < confidence_threshold:
                    continue

                batch_candidates.append({
                    'pts_2d': pts_2d,
                    'pts_3d': pts_3d,
                    'confs': confs_v,
                    'med_conf': med_conf,
                    'max_conf': max_conf,
                    'n_raw': n_raw,
                    'crop': (cx, cy, max(tile_w, tile_h)),
                })

            # Accumulate; PnP attempts are deferred until ALL batches in this
            # level have been scored, so every tile in the base grid (e.g. all
            # 9 tiles for a 2x2 level with 50% overlap) is guaranteed to be
            # matched and recorded in ``all_tile_scores`` before any PnP
            # early-return can fire.
            level_candidates.extend(batch_candidates)

        pbar.close()

        # ── Per-level PnP attempt: run on all candidates accumulated above ──
        level_candidates.sort(key=lambda c: (c['med_conf'], c['max_conf']), reverse=True)
        for cand in list(level_candidates):
            cx, cy, cs = cand['crop']
            pos_margin = max(tile_w, tile_h)

            pnp_runs = []
            for _ in range(10):
                pos_r, p2d_r, p3d_r, ni_r, rot_r, idx_r = \
                    localize_from_correspondences(
                        cand['pts_2d'], cand['pts_3d'], image_size, fov_vertical,
                        verbose=False, K=K, pnp_iterations=5000,
                    )
                if pos_r is not None and ni_r > 0:
                    if (abs(pos_r[0] - cx) > pos_margin or
                            abs(pos_r[1] - cy) > pos_margin):
                        continue
                    K_used = K if K is not None else get_intrinsics(image_size, fov_vertical)
                    reproj = _mean_reproj_error(
                        cand['pts_2d'], cand['pts_3d'], rot_r, pos_r, K_used, idx_r)
                    pnp_runs.append((ni_r, reproj, pos_r, p2d_r, p3d_r, rot_r, idx_r))

            if not pnp_runs:
                if verbose:
                    print(f"      Tile ({cx:.0f},{cy:.0f}) PnP failed")
                continue

            pnp_runs.sort(key=lambda r: (-r[0], r[1]))
            n_inl, _re, position, pnp_2d, pnp_3d, est_rotation, pnp_idx = pnp_runs[0]

            if n_inl < min_inliers:
                if verbose:
                    print(f"      Tile ({cx:.0f},{cy:.0f}) too few inliers: {n_inl}")
                continue

            if verbose:
                print(f"    ACCEPTED tile ({cx:.0f},{cy:.0f}) at level {level_n}: "
                      f"med_conf={cand['med_conf']:.3f}, max_conf={cand['max_conf']:.3f}, inliers={n_inl}")

            inl_mask = np.zeros(len(cand['pts_2d']), dtype=bool)
            if len(pnp_idx) == n_inl:
                inl_mask[pnp_idx] = True
            else:
                inl_mask[:n_inl] = True

            elapsed = _time.time() - t0
            if verbose:
                print(f"  Coarse-to-fine search done in {elapsed:.1f}s")

            return (position, cand['pts_2d'], cand['pts_3d'],
                    cand['confs'], n_inl, est_rotation,
                    cand['crop'], inl_mask, all_tile_scores)

        # Track best tile center for next level's ordering
        _level_entries = all_tile_scores[_level_start_idx:]
        if _level_entries:
            _best_this = max(_level_entries, key=lambda s: s[2])
            prev_best_center = (float(_best_this[0]), float(_best_this[1]))
            if verbose:
                print(f"    Best tile at level {level_n}: "
                      f"({prev_best_center[0]:.0f}, {prev_best_center[1]:.0f}) "
                      f"score={float(_best_this[2]):.3f}")

        if not level_candidates:
            if verbose:
                print(f"    No candidates at level {level_n}")
            continue

        if verbose:
            best_cand = max(level_candidates, key=lambda c: (c['med_conf'], c['max_conf']))
            print(f"    {len(level_candidates)} candidates (all batches exhausted PnP), "
                  f"best med_conf={best_cand['med_conf']:.3f}  max_conf={best_cand['max_conf']:.3f}")

    elapsed = _time.time() - t0
    if verbose:
        print(f"  Coarse-to-fine search: no tile accepted in {elapsed:.1f}s")

    # If calibration mode: return the best candidate seen across all levels
    # (highest median confidence) so the caller can run a FoV sweep.
    if return_best_on_pnp_fail and all_tile_scores:
        # Find the tile with the highest med_conf across all levels
        best_ts = max(all_tile_scores, key=lambda s: s[2])
        best_cx, best_cy, best_mc = best_ts[0], best_ts[1], best_ts[2]
        # Try to find the corresponding candidate data from the last level
        # We need to re-match this one tile for the caller to have pts_2d/pts_3d
        # Find which level_w/level_h this tile belongs to by checking against levels
        # For simplicity, use the finest level crop size
        _, last_grid, last_tw, last_th = levels[-1]
        dop_tile = geo_handler.crop_dop(best_cx, best_cy, max(last_tw, last_th))
        if dop_tile is not None:
            try:
                mr = matcher.match(image, dop_tile.data, num_matches=num_matches)
                if mr is not None and len(mr.kpts_query) > 0:
                    utm_xs, utm_ys = dop_tile.pixel_to_utm_batch(mr.kpts_dop[:, 0], mr.kpts_dop[:, 1])
                    zs = sample_full_dsm_batch(geo_handler, utm_xs, utm_ys)
                    valid = compute_valid_mask(dop_tile, mr.kpts_dop, zs)
                    if valid.sum() >= 10:
                        pts_2d = mr.kpts_query[valid].astype(np.float64)
                        pts_3d = np.column_stack([utm_xs[valid], utm_ys[valid], zs[valid]]).astype(np.float64)
                        confs_v = mr.confidences[valid].astype(np.float32)
                        crop = (float(best_cx), float(best_cy), max(last_tw, last_th))
                        if verbose:
                            print(f"  Returning best-conf tile ({best_cx:.0f},{best_cy:.0f}) "
                                  f"med_conf={best_mc:.3f} for calibration")
                        return (None, pts_2d, pts_3d, confs_v, 0, None, crop,
                                np.zeros(len(pts_2d), dtype=bool), all_tile_scores)
            except Exception:
                pass

    return (None, np.zeros((0, 2)), np.zeros((0, 3)),
            np.zeros(0, dtype=np.float32), 0, None, None,
            np.zeros(0, dtype=bool), all_tile_scores)


def exhaustive_tile_localization(
    image: np.ndarray,
    geo_handler,
    matcher: BaseMatcher,
    image_size: Tuple[int, int],
    fov_vertical: float,
    num_matches: int = 3000,
    crop_size: float = 300.0,
    overlap: float = 0.5,
    confidence_threshold: float = 0.4,
    confidence_min_count: int = 50,
    min_inliers: int = 30,
    K: np.ndarray = None,
    verbose: bool = False,
    batch_size: int = 8,
    early_stop_inliers: int = 100,
    center_hint: Optional[Tuple[float, float]] = None,
    return_best_on_pnp_fail: bool = False,
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray, int, Optional[np.ndarray], Optional[Tuple[float, float, float]]]:
    """Exhaustive tile search over the entire DOP for localization.

    Two-pass approach:

    **Pass 1** — fast, count-based candidate selection: tile the DOP,
    batch-match every tile, score each tile by the number of 3D-lifted
    correspondences above the confidence threshold.  Any tile reaching
    ``confidence_min_count`` becomes a candidate; the candidates are sorted
    by confident-match count.

    **Pass 2** — targeted geometric verification: run PnP-RANSAC only on the
    top-K candidates (default ``top_k_pnp`` = 5).  The top-K shortlist is
    determined purely by the count from Pass 1, so count is the primary
    selection criterion.  PnP acts as a verifier to find which candidate has
    geometrically consistent matches (necessary because the highest-count
    tile can contain noisy matches while a lower-ranked tile provides the
    correct consensus).

    Early termination during Pass 1 fires as soon as a tile's confident-match
    count reaches ``early_stop_inliers``.

    Parameters
    ----------
    image : UAV query image (H, W, 3).
    geo_handler : SequenceGeoHandler or GeoTIFFHandler.
    matcher : Feature matcher (turbo or precise).
    image_size : (height, width) of the image.
    fov_vertical : Vertical FOV in degrees.
    crop_size : Tile size in metres.
    overlap : Fractional overlap between adjacent tiles (0–1).
    confidence_threshold : Primary confidence filter for correspondences.
    confidence_min_count : Minimum number of confident 3D correspondences
        for a tile to be considered (no relaxed threshold).
    min_inliers : Minimum PnP inliers required to accept the best candidate.
    K : Camera intrinsics (3,3) or None to derive from FOV.
    verbose : Print progress info.
    batch_size : Number of tiles matched per GPU batch (default 8).
    early_stop_inliers : Stop Pass-1 scan if a tile reaches this many
        confident 3D points (default 100).

    Returns
    -------
    position : (3,) UTM position from the winning candidate, or None.
    pts_2d : (N,2) PnP inlier 2D points.
    pts_3d : (N,3) PnP inlier 3D points.
    confs : (N,) confidences.
    num_inliers : int.
    rotation : (3,3) R_c2w or None.
    crop_spec : (cx, cy, crop_size) of the winning candidate, or None.
    inlier_mask : (N,) bool mask over the winning candidate point set (empty on failure)."""
    (sample_full_dsm_batch, compute_valid_mask, get_full_dop_image,
     localize_from_correspondences, _mean_reproj_error) = _get_loc_helpers()

    # In calibration mode (return_best_on_pnp_fail) we try more candidates so
    # that a correct tile ranked outside the default top-5 gets a chance at PnP.
    top_k_pnp = 20 if return_best_on_pnp_fail else 5

    fail = (None, np.zeros((0, 2)), np.zeros((0, 3)),
            np.zeros(0, dtype=np.float32), 0, None, None,
            np.zeros(0, dtype=bool), [])

    # ── Get DOP bounds ──────────────────────────────────────────────────
    if isinstance(geo_handler, SequenceGeoHandler):
        bounds = geo_handler.dop_bounds
    elif isinstance(geo_handler, GeoTIFFHandler):
        b = geo_handler.dop_bounds
        bounds = [b.left, b.bottom, b.right, b.top]
    else:
        return fail

    min_x, min_y, max_x, max_y = bounds
    t0 = _time.time()

    # ── Generate tile grid ──────────────────────────────────────────────
    stride = crop_size * (1.0 - overlap)
    half = crop_size / 2.0
    cx_range = np.arange(min_x + half, max_x - half + stride, stride)
    cy_range = np.arange(min_y + half, max_y - half + stride, stride)
    if len(cx_range) == 0:
        cx_range = np.array([(min_x + max_x) / 2.0])
    if len(cy_range) == 0:
        cy_range = np.array([(min_y + max_y) / 2.0])

    tiles = [(float(cx), float(cy)) for cx in cx_range for cy in cy_range]
    n_tiles = len(tiles)

    # Sort by distance from the hint so closest tiles are evaluated first,
    # maximising the effectiveness of early stopping.
    hint_x = float(center_hint[0]) if center_hint is not None else (min_x + max_x) / 2.0
    hint_y = float(center_hint[1]) if center_hint is not None else (min_y + max_y) / 2.0
    tiles.sort(key=lambda t: (t[0] - hint_x) ** 2 + (t[1] - hint_y) ** 2)

    if verbose:
        dop_w, dop_h = max_x - min_x, max_y - min_y
        print(f"  Exhaustive tile search: {n_tiles} tiles "
              f"({crop_size:.0f}m, {overlap*100:.0f}% overlap, "
              f"DOP={dop_w:.0f}x{dop_h:.0f}m)")

    # ── Pre-crop all DOP tiles (CPU, fast) ─────────────────────────────
    tile_data = []
    for cx, cy in tiles:
        dop_tile = geo_handler.crop_dop(cx, cy, crop_size)
        if dop_tile is not None:
            tile_data.append((cx, cy, dop_tile))

    if verbose and len(tile_data) < n_tiles:
        print(f"  Skipped {n_tiles - len(tile_data)} out-of-bounds tiles, "
              f"{len(tile_data)} valid")

    # ── Pass 1: batch-match every tile, collect candidates by conf count ─
    candidates = []   # list of dicts, sorted by n_conf descending
    # Per-tile score map: list of (cx, cy, n_conf, n_raw) for all evaluated tiles.
    # n_conf = confident matches (>= confidence_threshold), n_raw = total 3D-lifted matches.
    all_tile_scores: list = []
    tiles_evaluated = 0
    early_stopped = False

    pbar = _tqdm(total=len(tile_data), desc="Tile search", disable=not verbose)

    for batch_start in range(0, len(tile_data), batch_size):
        batch = tile_data[batch_start:batch_start + batch_size]
        dop_images = [td[2].data for td in batch]

        # --- Batched matching (single GPU pass) ---
        try:
            match_results = matcher.match_batch(
                image, dop_images, num_matches_per_crop=num_matches,
            )
        except Exception:
            # OOM fallback: sequential matching for this batch
            if verbose:
                print(f"    Batch OOM at tiles {batch_start}-{batch_start + len(batch)}, "
                      f"falling back to sequential")
            torch.cuda.empty_cache()
            match_results = []
            for dop_img in dop_images:
                try:
                    match_results.append(
                        matcher.match(image, dop_img, num_matches=num_matches))
                except Exception:
                    match_results.append(None)

        # --- Score each tile by confident 3D-lifted match count ---
        for (cx, cy, dop_tile), match_result in zip(batch, match_results):
            tiles_evaluated += 1
            pbar.update(1)

            if match_result is None:
                continue

            kpts_q = match_result.kpts_query
            kpts_d = match_result.kpts_dop
            conf = match_result.confidences

            if len(kpts_q) < confidence_min_count:
                all_tile_scores.append((float(cx), float(cy), 0, 0))
                continue

            # Lift to 3D via DSM
            utm_xs, utm_ys = dop_tile.pixel_to_utm_batch(kpts_d[:, 0], kpts_d[:, 1])
            zs = sample_full_dsm_batch(geo_handler, utm_xs, utm_ys)
            valid = compute_valid_mask(dop_tile, kpts_d, zs)
            if not valid.any():
                all_tile_scores.append((float(cx), float(cy), 0, 0))
                continue

            pts_2d = kpts_q[valid].astype(np.float64)
            pts_3d = np.column_stack([utm_xs[valid], utm_ys[valid], zs[valid]]).astype(np.float64)
            confs = conf[valid].astype(np.float32)

            conf_mask = confs >= confidence_threshold
            filt_2d = pts_2d[conf_mask]
            filt_3d = pts_3d[conf_mask]
            filt_cf = confs[conf_mask]

            n_conf = len(filt_2d)
            n_raw = int(valid.sum())
            # Record score for every evaluated tile regardless of threshold
            all_tile_scores.append((float(cx), float(cy), n_conf, n_raw))

            if n_conf < confidence_min_count:
                continue

            mean_conf = float(np.mean(filt_cf))

            if verbose:
                print(f"    Tile ({cx:.0f},{cy:.0f}): {n_conf} confident pts, "
                      f"mean_conf={mean_conf:.3f}")

            candidates.append({
                'pts_2d': filt_2d,
                'pts_3d': filt_3d,
                'confs': filt_cf,
                'n_conf': n_conf,
                'mean_conf': mean_conf,
                'crop': (cx, cy, crop_size),
            })

        # --- Early termination if a very good tile was already found ---
        # Disabled in calibration mode (return_best_on_pnp_fail) to scan all tiles.
        if candidates and not return_best_on_pnp_fail:
            best_so_far = max(c['n_conf'] for c in candidates)
            if best_so_far >= early_stop_inliers:
                early_stopped = True
                pbar.close()
                break

    if not early_stopped:
        pbar.close()

    elapsed_pass1 = _time.time() - t0

    if not candidates:
        if verbose:
            print(f"  Exhaustive search: no candidates found in {elapsed_pass1:.1f}s")
        return fail

    # Sort candidates by confident-match count (descending)
    candidates.sort(key=lambda c: c['n_conf'], reverse=True)

    if verbose:
        stop_info = (f" (early stop at {tiles_evaluated}/{len(tile_data)} tiles)"
                     if early_stopped else "")
        print(f"  Pass 1 done in {elapsed_pass1:.1f}s{stop_info}: "
              f"{len(candidates)} candidates, top-5 n_conf = "
              f"{[c['n_conf'] for c in candidates[:5]]}")

    # ── Pass 2: PnP on top-K candidates ────────────────────────────────
    best = None
    best_inliers = 0

    for rank, cand in enumerate(candidates[:top_k_pnp]):
        cx, cy, _ = cand['crop']
        half = crop_size / 2.0
        # Allow up to 2.0× the tile half-size outside tile centre — oblique
        # cameras can have their optical centre well outside the matched tile
        # while still observing the tile's content.  Positions beyond 2× are
        # almost certainly degenerate RANSAC attractors.
        pos_margin = 2.0 * half

        # Run PnP multiple times and select by inlier count, breaking ties
        # by lower mean reprojection error.  Near-planar terrain produces
        # multiple ~equal-inlier RANSAC attractors; the correct solution
        # typically has the tightest geometric consensus.
        pnp_runs = []
        for _ in range(10):
            pos_r, p2d_r, p3d_r, ni_r, rot_r, idx_r = \
                localize_from_correspondences(
                    cand['pts_2d'], cand['pts_3d'], image_size, fov_vertical,
                    verbose=False, K=K,
                    pnp_iterations=5000,
                )
            if pos_r is not None and ni_r > 0:
                # Reject solutions whose XY position is far outside the tile;
                # these are almost always degenerate RANSAC attractors.
                if (abs(pos_r[0] - cx) > pos_margin or
                        abs(pos_r[1] - cy) > pos_margin):
                    continue
                # Compute mean reprojection error on inlier subset for tie-breaking.
                K_used = K if K is not None else get_intrinsics(image_size, fov_vertical)
                reproj = _mean_reproj_error(
                    cand['pts_2d'], cand['pts_3d'], rot_r, pos_r, K_used, idx_r)
                pnp_runs.append((ni_r, reproj, pos_r, p2d_r, p3d_r, rot_r, idx_r))

        if not pnp_runs:
            n_inl, _re, position, pnp_2d, pnp_3d, est_rotation, pnp_idx = \
                0, float('inf'), None, np.zeros((0,2)), np.zeros((0,3)), None, np.zeros(0,dtype=int)
        else:
            # Primary sort: most inliers; secondary: lowest reprojection error.
            pnp_runs.sort(key=lambda r: (-r[0], r[1]))
            n_inl, _re, position, pnp_2d, pnp_3d, est_rotation, pnp_idx = pnp_runs[0]

        if verbose:
            print(f"    [PnP rank {rank+1}] Tile ({cx:.0f},{cy:.0f}): "
                  f"{n_inl} inliers "
                  + (f"pos=({position[0]:.0f},{position[1]:.0f},{position[2]:.0f})"
                     f" reproj={_re:.2f}px ({len(pnp_runs)} runs)"
                     if position is not None else "FAILED"))
            if pnp_runs and position is not None:
                # Show all unique solutions found across restarts
                seen_pos = set()
                for ni_r, re_r, pos_r, *_ in sorted(pnp_runs, key=lambda r: -r[0]):
                    key = (round(pos_r[0]), round(pos_r[1]), round(pos_r[2]))
                    if key not in seen_pos:
                        seen_pos.add(key)
                        print(f"      alt solution: pos=({key[0]},{key[1]},{key[2]})"
                              f" ni={ni_r} re={re_r:.2f}px")

        if position is None or n_inl < min_inliers:
            continue

        if n_inl > best_inliers:
            best_inliers = n_inl
            # Build a boolean mask so the caller can use either the full
            # candidate set (for calibration sweeps) or the PnP-verified
            # inlier subset (for rough PnP when intrinsics are known).
            inl_mask = np.zeros(len(cand['pts_2d']), dtype=bool)
            if len(pnp_idx) == n_inl:
                inl_mask[pnp_idx] = True
            else:
                # pnp_idx may index into a centroid-subtracted copy; best effort
                inl_mask[:n_inl] = True
            best = {
                'position': position,
                'pts_2d': cand['pts_2d'],
                'pts_3d': cand['pts_3d'],
                'confs': cand['confs'],
                'inl_mask': inl_mask,
                'inliers': n_inl,
                'rotation': est_rotation,
                'crop': cand['crop'],
            }

    elapsed = _time.time() - t0

    if best is None:
        if verbose:
            print(f"  Exhaustive search: PnP failed on all top-{top_k_pnp} "
                  f"candidates in {elapsed:.1f}s")
        if return_best_on_pnp_fail and candidates:
            # Return best-match-count candidate's correspondences so the caller
            # can run a FoV calibration sweep even without a valid PnP pose.
            best_cand = candidates[0]
            cx_b, cy_b, _ = best_cand['crop']
            if verbose:
                print(f"  Returning best match-count candidate ({cx_b:.0f},{cy_b:.0f}) "
                      f"for calibration ({best_cand['n_conf']} pts, PnP skipped)")
            return (None, best_cand['pts_2d'], best_cand['pts_3d'],
                    best_cand['confs'], 0, None, best_cand['crop'],
                    np.zeros(len(best_cand['pts_2d']), dtype=bool), all_tile_scores)
        # PnP failed on all candidates (calibration mode off): return None pose
        # but preserve tile scores so the debug visualization can show the heatmap
        return (None, np.zeros((0, 2)), np.zeros((0, 3)),
                np.zeros(0, dtype=np.float32), 0, None, None,
                np.zeros(0, dtype=bool), all_tile_scores)

    elapsed = _time.time() - t0

    if verbose:
        cx_b, cy_b, _ = best['crop']
        print(f"  Exhaustive search: best tile ({cx_b:.0f},{cy_b:.0f}) "
              f"with {best['inliers']} PnP inliers in {elapsed:.1f}s")

    return (best['position'], best['pts_2d'], best['pts_3d'],
            best['confs'], best['inliers'], best['rotation'],
            best['crop'], best['inl_mask'], all_tile_scores)

