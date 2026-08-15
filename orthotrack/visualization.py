"""
Tracking pipeline visualisations: keyframe match panels, tracking overlay,
refinement candidates, summary plots, and per-stage debug figures."""
from __future__ import annotations

import gc
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from orthotrack.types import FrameResult


def compute_footprint_polygon(
    position: np.ndarray,
    R_c2w: np.ndarray,
    image_size: Tuple[int, int],
    ground_z: float,
    intrinsics,
) -> Optional[List[Tuple[float, float]]]:
    """Project the 4 image corners onto a flat ground plane at *ground_z* (UTM Z).

    Returns a list of 4 (utm_x, utm_y) tuples forming the footprint quadrilateral,
    or None if the projection fails (e.g. any corner is behind the camera)."""
    from orthotrack.crop_strategy import get_intrinsics as _gi
    h, w = image_size
    K = getattr(intrinsics, 'K', None)
    if K is None:
        K = _gi((h, w), intrinsics.fov_vertical)
    corners_px = np.array(
        [[0.5, 0.5], [w - 0.5, 0.5], [w - 0.5, h - 0.5], [0.5, h - 0.5]],
        dtype=float,
    )
    K_inv = np.linalg.inv(K)
    dirs_cam = (K_inv @ np.hstack([corners_px, np.ones((4, 1))]).T).T  # (4,3)
    dirs_world = (R_c2w @ dirs_cam.T).T
    dirs_world /= np.linalg.norm(dirs_world, axis=1, keepdims=True)
    O = position
    pts = []
    for D in dirs_world:
        dz = D[2]
        if abs(dz) < 1e-6:
            return None
        t = (ground_z - O[2]) / dz
        if t < 0:
            return None
        P = O + t * D
        pts.append((float(P[0]), float(P[1])))
    return pts if len(pts) == 4 else None


# ------------------------------------------------------------------ #
#  Per-frame visualisations                                            #
# ------------------------------------------------------------------ #

def save_keyframe_visualization(
    frame_id: int,
    image: np.ndarray,
    dop_tile,
    kpts_query: np.ndarray,
    kpts_dop: np.ndarray,
    inlier_mask: np.ndarray,
    output_dir: Path,
    est_position: np.ndarray = None,
    gt_position: np.ndarray = None,
    accepted: bool = None,
    num_inliers: int = None,
    title_suffix: str = "",
    refined: bool = False,
    confidences: np.ndarray = None,
    geo_handler=None,
    crop_specs: List[Tuple[float, float, float]] = None,
    frame_type: str = "keyframe",
    reproj_error: float = None,
    keyframe_id: int = None,
    initial_num_pts: int = None,
    fig_ext: str = "png",
    processing_fps: float = None,
    trajectory_positions: List[Tuple[float, float]] = None,
    trajectory_keyframe_flags: List[bool] = None,
    keyframe_dop_points: np.ndarray = None,
    keyframe_dop_confs: np.ndarray = None,
    lod_overlay: np.ndarray = None,
    footprint_polygon: Optional[List[Tuple[float, float]]] = None,
):
    """
    Unified 4-panel keyframe / tracking visualisation.

    Panels:
        1. Query image with matched/tracked points (colored by confidence)
        2. Full DOP overview with footprint polygon (or crop rectangles), positions, matched points
        3. DOP crop with matched points (colored by confidence)
        4. Side-by-side match lines (colored by confidence)

    Parameters
    ----------
    frame_id : Frame number.
    image : Query image (H, W, 3).
    dop_tile : GeoTile for the primary DOP crop.
    kpts_query : (N, 2) keypoints in query image.
    kpts_dop : (N, 2) keypoints in dop_tile pixel coords.
    inlier_mask : (N,) bool mask of PnP inliers.
    output_dir : Output directory.
    est_position : (3,) estimated UTM position.
    gt_position : (3,) ground truth UTM position.
    accepted : Whether this keyframe was accepted (None for tracked frames).
    num_inliers : Number of PnP inliers.
    title_suffix : Extra string for the title.
    refined : Whether this is a refined keyframe.
    confidences : (N,) match confidence per point.
    geo_handler : SequenceGeoHandler for full DOP overview panel.
    crop_specs : List of (cx, cy, size) crop rectangles to draw.
    frame_type : "keyframe", "tracked", or "predicted".
    reproj_error : PnP reprojection error in pixels.
    keyframe_id : ID of the source keyframe (for tracked frames).
    initial_num_pts : Initial number of tracked points from keyframe.
    processing_fps : Pipeline processing speed (frames per second) for display.
    trajectory_positions : List of (utm_x, utm_y) for all estimated positions so far.
    trajectory_keyframe_flags : List of bools indicating which positions are keyframes.
    keyframe_dop_points : (K, 2) all original keyframe DOP pixel coords (for persistence).
    keyframe_dop_confs : (K,) confidences for keyframe DOP points."""
    import matplotlib.gridspec as gridspec

    # Determine if we have time-series data to show
    has_timeseries = (hasattr(save_keyframe_visualization, '_results_history')
                      and save_keyframe_visualization._results_history is not None
                      and len(save_keyframe_visualization._results_history) > 0)

    if has_timeseries:
        fig = plt.figure(figsize=(28, 12))
        gs = gridspec.GridSpec(2, 4, height_ratios=[2, 1], hspace=0.25, wspace=0.3)
        axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
        ax_reproj = fig.add_subplot(gs[1, :2])
        ax_npts = fig.add_subplot(gs[1, 2:])
    else:
        fig = plt.figure(figsize=(28, 12))
        gs = gridspec.GridSpec(2, 4, height_ratios=[2, 1], hspace=0.25, wspace=0.3)
        axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
        # Hide bottom row when no time-series data
        ax_reproj_placeholder = fig.add_subplot(gs[1, :2])
        ax_npts_placeholder = fig.add_subplot(gs[1, 2:])
        ax_reproj_placeholder.axis('off')
        ax_npts_placeholder.axis('off')
        ax_reproj = None
        ax_npts = None

    has_pts = kpts_query is not None and len(kpts_query) > 0
    has_conf = confidences is not None and len(confidences) > 0

    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    # --- Panel 1: Query image with points ---
    ax1 = axes[0]
    h_img, w_img = image.shape[:2]
    ax1.imshow(image)
    if lod_overlay is not None:
        ax1.imshow(lod_overlay, extent=[-0.5, w_img - 0.5, h_img - 0.5, -0.5], zorder=2)
    if has_pts:
        if has_conf:
            sc = ax1.scatter(kpts_query[:, 0], kpts_query[:, 1],
                             c=confidences, cmap=cmap, norm=norm, s=4, alpha=0.7)
        else:
            ax1.scatter(kpts_query[:, 0], kpts_query[:, 1], c='lime', s=4, alpha=0.6)
    title1 = f'Frame {frame_id}'
    if frame_type == "tracked" and keyframe_id is not None:
        title1 += f' (KF {keyframe_id})'
    title1 += f' \u2014 {len(kpts_query) if has_pts else 0} pts'
    ax1.set_title(title1, fontsize=10)
    ax1.set_xlim(0, w_img)
    ax1.set_ylim(h_img, 0)
    ax1.axis('off')

    # Info text box on query panel (unified format for keyframe and tracked)
    info_lines = []
    # Method label
    if frame_type == "keyframe":
        info_lines.append("Keyframe")
    elif frame_type == "tracked":
        info_lines.append("Tracked")
    elif frame_type == "predicted":
        info_lines.append("Predicted")
    # Points line — always present for consistency
    n_pts = len(kpts_query) if has_pts else 0
    if frame_type == "tracked" and initial_num_pts is not None and initial_num_pts > 0:
        pct = 100.0 * n_pts / initial_num_pts
        info_lines.append(f"Points: {n_pts}/{initial_num_pts} ({pct:.0f}%)")
    elif num_inliers is not None and num_inliers > 0:
        info_lines.append(f"Inliers: {num_inliers}/{n_pts}")
    else:
        info_lines.append(f"Points: {n_pts}")
    # Reproj / confidence — always one line
    if reproj_error is not None and reproj_error < 1e6:
        conf_str = f" | conf={np.mean(confidences):.2f}" if has_conf else ""
        info_lines.append(f"Reproj: {reproj_error:.2f}px{conf_str}")
    elif has_conf:
        info_lines.append(f"Mean conf: {np.mean(confidences):.3f}")
    else:
        info_lines.append("")  # placeholder for consistent height
    # Error line — always present
    if est_position is not None and gt_position is not None:
        err_3d = np.linalg.norm(est_position - gt_position)
        err_2d = np.linalg.norm(est_position[:2] - gt_position[:2])
        info_lines.append(f"Err: {err_3d:.1f}m (2D: {err_2d:.1f}m)")
        err_color = 'green' if err_3d < 5 else ('orange' if err_3d < 15 else 'red')
    else:
        info_lines.append("")  # placeholder
        err_color = 'gray'
    # Remove trailing empty placeholders
    while info_lines and info_lines[-1] == "":
        info_lines.pop()
    if info_lines:
        ax1.text(0.02, 0.98, "\n".join(info_lines), transform=ax1.transAxes,
                 fontsize=9, fontweight='bold', color='white',
                 verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=err_color, alpha=0.8))

    # --- Panel 2: Full DOP overview with crop rectangles ---
    ax2 = axes[1]
    full_dop = None
    if geo_handler is not None:
        try:
            # SequenceGeoHandler exposes .dop_data directly (already loaded)
            if hasattr(geo_handler, 'dop_data'):
                full_dop = geo_handler.dop_data
            elif hasattr(geo_handler, '_dop_data'):
                # GeoTIFFHandler: preload and transpose (bands,H,W) → (H,W,C)
                if geo_handler._dop_data is None:
                    geo_handler.preload(is_dsm=False)
                data = geo_handler._dop_data
                if data is not None:
                    full_dop = np.transpose(data[:3], (1, 2, 0)) if data.ndim == 3 else data
        except Exception:
            pass
    if full_dop is not None:
        ax2.imshow(full_dop)
        # Draw footprint polygon or crop rectangles
        crop_colors = ['cyan', 'yellow', 'orange', 'magenta', 'white']
        if footprint_polygon is not None and geo_handler is not None:
            px_pts = [geo_handler.utm_to_pixel(x, y) for (x, y) in footprint_polygon]
            poly = mpatches.Polygon(px_pts, closed=True, fill=False,
                                    edgecolor='cyan', linewidth=2,
                                    linestyle='-', alpha=0.9, zorder=7)
            ax2.add_patch(poly)
        elif crop_specs:
            for idx, (cx, cy, sz) in enumerate(crop_specs):
                half = sz / 2
                left, top = cx - half, cy + half
                right, bottom = cx + half, cy - half
                px_l, py_t = geo_handler.utm_to_pixel(left, top)
                px_r, py_b = geo_handler.utm_to_pixel(right, bottom)
                w_rect = px_r - px_l
                h_rect = py_b - py_t
                color = crop_colors[idx % len(crop_colors)]
                rect = mpatches.Rectangle((px_l, py_t), w_rect, h_rect,
                                          linewidth=2, edgecolor=color,
                                          facecolor='none', linestyle='--')
                ax2.add_patch(rect)
                ax2.text(px_l + 3, py_t + 15, f'{sz:.0f}m',
                         color=color, fontsize=7, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5))
        # Plot persistent keyframe DOP points (greyed out for lost ones)
        if keyframe_dop_points is not None and len(keyframe_dop_points) > 0 and dop_tile is not None:
            utm_xs_kf, utm_ys_kf = dop_tile.pixel_to_utm_batch(keyframe_dop_points[:, 0], keyframe_dop_points[:, 1])
            full_cols_kf = np.zeros(len(utm_xs_kf))
            full_rows_kf = np.zeros(len(utm_xs_kf))
            for j in range(len(utm_xs_kf)):
                fc, fr = geo_handler.utm_to_pixel(utm_xs_kf[j], utm_ys_kf[j])
                full_cols_kf[j], full_rows_kf[j] = fc, fr
            if keyframe_dop_confs is not None and len(keyframe_dop_confs) == len(keyframe_dop_points):
                ax2.scatter(full_cols_kf, full_rows_kf, c=keyframe_dop_confs, cmap=cmap, norm=norm,
                            s=2, alpha=0.3, zorder=2)
            else:
                ax2.scatter(full_cols_kf, full_rows_kf, c='gray', s=2, alpha=0.2, zorder=2)
        # Plot matched points projected to full DOP (current surviving points)
        if has_pts and kpts_dop is not None and len(kpts_dop) > 0 and dop_tile is not None:
            # Convert dop_tile pixel coords -> UTM -> full DOP pixel
            utm_xs_dop, utm_ys_dop = dop_tile.pixel_to_utm_batch(kpts_dop[:, 0], kpts_dop[:, 1])
            full_cols = np.zeros(len(utm_xs_dop))
            full_rows = np.zeros(len(utm_xs_dop))
            for j in range(len(utm_xs_dop)):
                fc, fr = geo_handler.utm_to_pixel(utm_xs_dop[j], utm_ys_dop[j])
                full_cols[j], full_rows[j] = fc, fr
            if has_conf:
                ax2.scatter(full_cols, full_rows, c=confidences, cmap=cmap, norm=norm,
                            s=4, alpha=0.7, zorder=3)
            else:
                ax2.scatter(full_cols, full_rows, c='lime', s=4, alpha=0.6, zorder=3)
        # Plot trajectory line
        if trajectory_positions is not None and len(trajectory_positions) > 1:
            traj_cols = np.zeros(len(trajectory_positions))
            traj_rows = np.zeros(len(trajectory_positions))
            for j, (tx, ty) in enumerate(trajectory_positions):
                tc, tr = geo_handler.utm_to_pixel(tx, ty)
                traj_cols[j], traj_rows[j] = tc, tr
            ax2.plot(traj_cols, traj_rows, '-', color='deepskyblue', linewidth=1.5,
                     alpha=0.8, zorder=5, label='Trajectory')
            # Mark keyframes on trajectory
            if trajectory_keyframe_flags is not None and len(trajectory_keyframe_flags) == len(trajectory_positions):
                kf_mask = np.array(trajectory_keyframe_flags, dtype=bool)
                if kf_mask.any():
                    ax2.scatter(traj_cols[kf_mask], traj_rows[kf_mask], c='red', s=30,
                                marker='o', zorder=6, label='Keyframes')
        # Plot GT and estimated positions
        if gt_position is not None:
            gx, gy = geo_handler.utm_to_pixel(gt_position[0], gt_position[1])
            ax2.scatter([gx], [gy], c='magenta', s=150, marker='*',
                        linewidths=1.5, zorder=10, label='GT')
        if est_position is not None:
            ex, ey = geo_handler.utm_to_pixel(est_position[0], est_position[1])
            ax2.scatter([ex], [ey], c='cyan', s=150, marker='x',
                        linewidths=2, zorder=10, label='Est')
        ax2.set_title('DOP Overview + Trajectory', fontsize=10)
        ax2.legend(loc='upper right', fontsize=7)
        ax2.axis('off')
    else:
        # Fallback: no full DOP available — show crop with positions
        ax2.text(0.5, 0.5, 'Full DOP not available',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.axis('off')

    # --- Panel 3: DOP crop with matched points ---
    ax3 = axes[2]
    if dop_tile is not None:
        ax3.imshow(dop_tile.data)
        if has_pts and kpts_dop is not None and len(kpts_dop) > 0:
            if has_conf:
                ax3.scatter(kpts_dop[:, 0], kpts_dop[:, 1],
                            c=confidences, cmap=cmap, norm=norm, s=4, alpha=0.7)
            else:
                ax3.scatter(kpts_dop[:, 0], kpts_dop[:, 1], c='lime', s=4, alpha=0.6)
        # Crop center
        if dop_tile.width > 0 and dop_tile.height > 0:
            cx_px, cy_px = dop_tile.width / 2, dop_tile.height / 2
            ax3.scatter([cx_px], [cy_px], c='red', s=80, marker='+',
                        linewidths=2, label='Crop center')
        # Estimated position
        if est_position is not None:
            est_px, est_py = dop_tile.utm_to_pixel(est_position[0], est_position[1])
            if 0 <= est_px < dop_tile.width and 0 <= est_py < dop_tile.height:
                ax3.scatter([est_px], [est_py], c='cyan', s=100, marker='x',
                            linewidths=2, label='Estimated')
        # GT position
        if gt_position is not None:
            gt_px, gt_py = dop_tile.utm_to_pixel(gt_position[0], gt_position[1])
            if 0 <= gt_px < dop_tile.width and 0 <= gt_py < dop_tile.height:
                ax3.scatter([gt_px], [gt_py], c='magenta', s=100, marker='*',
                            linewidths=2, label='GT')
        ax3.set_title(f'DOP Crop ({dop_tile.width}×{dop_tile.height})', fontsize=10)
        ax3.legend(loc='upper right', fontsize=7)
        ax3.axis('off')

        # Error text box
        if est_position is not None and gt_position is not None:
            error_3d = np.linalg.norm(est_position - gt_position)
            error_2d = np.linalg.norm(est_position[:2] - gt_position[:2])
            error_z = abs(est_position[2] - gt_position[2])
            error_text = f"3D: {error_3d:.1f}m\n2D: {error_2d:.1f}m\nZ:  {error_z:.1f}m"
            error_color = 'green' if error_3d < 5 else ('orange' if error_3d < 15 else 'red')
            ax3.text(0.02, 0.02, error_text, transform=ax3.transAxes,
                     fontsize=10, fontweight='bold', color='white',
                     verticalalignment='bottom',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=error_color, alpha=0.8))
    else:
        ax3.text(0.5, 0.5, 'No DOP crop', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=12)
        ax3.axis('off')

    # --- Panel 4: Match lines side-by-side ---
    ax4 = axes[3]
    if dop_tile is not None and has_pts and kpts_dop is not None and len(kpts_dop) > 0:
        h1, w1 = image.shape[:2]
        h2, w2 = dop_tile.data.shape[:2]
        max_h = max(h1, h2)
        combined = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
        combined[:h1, :w1] = image
        combined[:h2, w1:] = dop_tile.data
        ax4.imshow(combined)

        n_show = min(200, len(kpts_query))
        indices = np.linspace(0, len(kpts_query) - 1, n_show, dtype=int)
        for i in indices:
            x1, y1 = kpts_query[i]
            x2, y2 = kpts_dop[i]
            if has_conf:
                color = cmap(norm(confidences[i]))
            else:
                color = 'lime' if (inlier_mask is not None and len(inlier_mask) > i and inlier_mask[i]) else 'red'
            ax4.plot([x1, x2 + w1], [y1, y2], color=color, alpha=0.5, linewidth=0.5)

        if has_conf:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax4, fraction=0.03, pad=0.01)
            cbar.set_label('Confidence', fontsize=9)

        ax4.set_title('Correspondences', fontsize=10)
    else:
        ax4.text(0.5, 0.5, 'No correspondences', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=12)
    ax4.axis('off')

    # --- Time-series panels (bottom row) ---
    if has_timeseries and ax_reproj is not None and ax_npts is not None:
        results = save_keyframe_visualization._results_history
        frames_ids = [r.frame_id for r in results]
        # Reproj error evolution
        reproj_vals = []
        reproj_frames = []
        for r in results:
            if r.reproj_error is not None and r.reproj_error < 1e6:
                reproj_vals.append(r.reproj_error)
                reproj_frames.append(r.frame_id)
        if reproj_vals:
            # Color by method
            reproj_colors = []
            for r in results:
                if r.reproj_error is not None and r.reproj_error < 1e6:
                    if r.is_keyframe:
                        reproj_colors.append('red')
                    elif r.method == 'tracked':
                        reproj_colors.append('steelblue')
                    else:
                        reproj_colors.append('gray')
            ax_reproj.scatter(reproj_frames, reproj_vals, c=reproj_colors, s=10, alpha=0.7, zorder=3)
            ax_reproj.plot(reproj_frames, reproj_vals, '-', color='gray', alpha=0.3, linewidth=0.5, zorder=2)
            # Current frame marker
            if reproj_error is not None and reproj_error < 1e6:
                ax_reproj.scatter([frame_id], [reproj_error], c='orange', s=80, marker='D',
                                  zorder=5, edgecolors='black', linewidths=1, label='Current')
        # Draw adaptive threshold lines (abs = effective threshold with baseline lift)
        if hasattr(save_keyframe_visualization, '_threshold_history') and save_keyframe_visualization._threshold_history:
            th_data = save_keyframe_visualization._threshold_history
            th_frames = [t[0] for t in th_data]
            th_abs = [t[1] for t in th_data]
            th_rel = [t[2] for t in th_data]
            ax_reproj.plot(th_frames, th_abs, '--', color='#F44336', alpha=0.7, linewidth=1.2, label='Eff. abs threshold')
            # Only plot growth threshold where it's meaningful (not inf)
            valid_rel = [(f, v) for f, v in zip(th_frames, th_rel) if v < 100]
            if valid_rel:
                ax_reproj.plot([x[0] for x in valid_rel], [x[1] for x in valid_rel],
                               ':', color='#FF9800', alpha=0.7, linewidth=1.2, label='Growth threshold')

        # Mark keyframes with vertical lines and annotate trigger reason
        kf_frames = [r.frame_id for r in results if r.is_keyframe]
        kf_reasons_map = {r.frame_id: r.kf_reason for r in results if r.is_keyframe and hasattr(r, 'kf_reason')}
        for kf in kf_frames:
            ax_reproj.axvline(kf, color='#4CAF50', alpha=0.3, linewidth=0.8)
        # Annotate KF trigger reasons
        for kf in kf_frames:
            reason = kf_reasons_map.get(kf, "")
            if not reason or kf == frames_ids[0]:
                continue
            if "reproj_abs" in reason:
                label = "abs"
            elif "reproj_growth" in reason:
                label = "grw"
            elif "low_points" in reason:
                label = "pts"
            elif "spatial" in reason:
                label = "spt"
            else:
                label = reason[:4] if reason else ""
            if label:
                # Place at the top of the plot
                ax_reproj.annotate(
                    label, xy=(kf, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -2), textcoords='offset points',
                    fontsize=6, color='#D32F2F', fontweight='bold', ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='#D32F2F', alpha=0.8, lw=0.4),
                )

        ax_reproj.set_xlabel('Frame', fontsize=9)
        ax_reproj.set_ylabel('Reprojection Error (px)', fontsize=9)
        ax_reproj.set_title('Reprojection Error Evolution', fontsize=10)
        ax_reproj.legend(fontsize=6, loc='upper left', ncol=2)
        ax_reproj.grid(True, alpha=0.3)
        if reproj_vals:
            ax_reproj.set_ylim(0, min(15, max(reproj_vals) * 1.2 + 0.5))
        ax_reproj.set_xlim(frames_ids[0] - 1, frames_ids[-1] + 1)

        # Tracked points count
        npts_frames = [r.frame_id for r in results]
        npts_vals = [r.num_tracked_points for r in results]
        npts_colors = ['red' if r.is_keyframe else ('steelblue' if r.method == 'tracked' else 'gray') for r in results]
        ax_npts.bar(npts_frames, npts_vals, color=npts_colors, alpha=0.7, width=0.8)
        # Thresholds
        thresh_vals = [r.tracked_points_threshold for r in results]
        ax_npts.step(npts_frames, thresh_vals, where='post', color='red', linestyle='--',
                     alpha=0.8, linewidth=1.2, label='Adaptive threshold')
        # Hardcoded min threshold
        if hasattr(save_keyframe_visualization, '_keyframe_min_points'):
            min_pts = save_keyframe_visualization._keyframe_min_points
            ax_npts.axhline(min_pts, color='black', linestyle=':', alpha=0.5,
                            label=f'Min points ({min_pts})')
        # Mark keyframes
        for kf in kf_frames:
            ax_npts.axvline(kf, color='#4CAF50', alpha=0.3, linewidth=0.8)
        ax_npts.set_xlabel('Frame', fontsize=9)
        ax_npts.set_ylabel('Tracked Points', fontsize=9)
        ax_npts.set_title('Tracked Points Count', fontsize=10)
        ax_npts.legend(fontsize=7, loc='upper right')
        ax_npts.grid(True, alpha=0.3)
        ax_npts.set_xlim(frames_ids[0] - 1, frames_ids[-1] + 1)

    # --- Suptitle ---
    error_str = ""
    if est_position is not None and gt_position is not None:
        error_3d = np.linalg.norm(est_position - gt_position)
        error_2d = np.linalg.norm(est_position[:2] - gt_position[:2])
        error_str = f" | Error: {error_3d:.1f}m (2D: {error_2d:.1f}m)"

    status_str = ""
    if accepted is not None:
        status_str = " ✓ ACCEPTED" if accepted else " ✗ REJECTED"

    inlier_str = f" | Inliers: {num_inliers}" if num_inliers else ""
    refined_str = " [REFINED]" if refined else ""
    type_str = f" [{frame_type.upper()}]"
    suffix_display = f" | {title_suffix}" if title_suffix else ""
    fps_str = f" | {processing_fps:.1f} FPS" if processing_fps is not None else ""

    status_color = 'black'
    if accepted is True:
        status_color = 'green'
    elif accepted is False:
        status_color = 'red'

    fig.suptitle(
        f'Frame {frame_id}{type_str}{refined_str}{status_str}{inlier_str}{error_str}{fps_str}{suffix_display}',
        fontsize=13, fontweight='bold', color=status_color,
    )

    # Use constrained_layout=False and manual subplots_adjust for consistent sizing
    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.05, hspace=0.25, wspace=0.3)

    # Determine save path
    name = f"frame_{frame_id:06d}_{frame_type}"
    if refined:
        name += "_refined"
    if title_suffix:
        safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in title_suffix)
        safe = safe.strip("_")
        if safe:
            name += f"_{safe}"

    save_path = output_dir / "keyframes" / f"{name}.{fig_ext}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100)
    fig.clf()
    plt.close(fig)
    gc.collect()


def save_multicrop_visualization(
    frame_id: int,
    image: np.ndarray,
    crop_data_list: List[Dict],
    output_dir: Path,
    est_position: np.ndarray = None,
    gt_position: np.ndarray = None,
    total_inliers: int = None,
    title: str = "",
    fig_ext: str = "png",
):
    """
    Visualise per-crop matching results (one row per crop).

    Each row shows: query with points | DOP crop with points | match lines.
    All plots use confidence coloring (RdYlGn).

    Parameters
    ----------
    crop_data_list : list of dicts, each with:
        - 'dop_tile': GeoTile
        - 'kpts_query': (N, 2)
        - 'kpts_dop': (N, 2)
        - 'confidences': (N,)
        - 'name': str label (e.g. "Crop 1 (300m)")"""
    if not crop_data_list:
        return

    n_crops = len(crop_data_list)
    fig, axes = plt.subplots(n_crops, 3, figsize=(21, 5.5 * n_crops), squeeze=False)

    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    error_3d = np.linalg.norm(est_position - gt_position) if (est_position is not None and gt_position is not None) else None
    error_2d = np.linalg.norm(est_position[:2] - gt_position[:2]) if (est_position is not None and gt_position is not None) else None

    for row, cd in enumerate(crop_data_list):
        name = cd.get('name', f'Crop {row+1}')
        dop_tile = cd['dop_tile']
        kpts_q = cd.get('kpts_query')
        kpts_d = cd.get('kpts_dop')
        conf = cd.get('confidences')
        has_pts = kpts_q is not None and len(kpts_q) > 0
        has_conf = conf is not None and len(conf) > 0

        # Col 1: query
        ax1 = axes[row, 0]
        ax1.imshow(image)
        if has_pts:
            if has_conf:
                ax1.scatter(kpts_q[:, 0], kpts_q[:, 1], c=conf, cmap=cmap, norm=norm, s=4, alpha=0.7)
            else:
                ax1.scatter(kpts_q[:, 0], kpts_q[:, 1], c='lime', s=4, alpha=0.6)
        n_pts = len(kpts_q) if has_pts else 0
        mean_c = f", mean conf {np.mean(conf):.2f}" if has_conf else ""
        ax1.set_title(f'{name}: Query ({n_pts} pts{mean_c})', fontsize=9)
        ax1.axis('off')

        # Col 2: DOP crop
        ax2 = axes[row, 1]
        ax2.imshow(dop_tile.data)
        if has_pts and kpts_d is not None and len(kpts_d) > 0:
            if has_conf:
                ax2.scatter(kpts_d[:, 0], kpts_d[:, 1], c=conf, cmap=cmap, norm=norm, s=4, alpha=0.7)
            else:
                ax2.scatter(kpts_d[:, 0], kpts_d[:, 1], c='lime', s=4, alpha=0.6)
        if gt_position is not None:
            gt_px, gt_py = dop_tile.utm_to_pixel(gt_position[0], gt_position[1])
            if 0 <= gt_px < dop_tile.width and 0 <= gt_py < dop_tile.height:
                ax2.scatter([gt_px], [gt_py], c='magenta', s=100, marker='*', linewidths=2)
        if est_position is not None:
            est_px, est_py = dop_tile.utm_to_pixel(est_position[0], est_position[1])
            if 0 <= est_px < dop_tile.width and 0 <= est_py < dop_tile.height:
                ax2.scatter([est_px], [est_py], c='cyan', s=100, marker='x', linewidths=2)
        ax2.set_title(f'{name}: DOP ({dop_tile.width}×{dop_tile.height})', fontsize=9)
        ax2.axis('off')

        # Col 3: match lines
        ax3 = axes[row, 2]
        h1, w1 = image.shape[:2]
        h2, w2 = dop_tile.data.shape[:2]
        max_h = max(h1, h2)
        combined = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
        combined[:h1, :w1] = image
        combined[:h2, w1:] = dop_tile.data
        ax3.imshow(combined)
        if has_pts and kpts_d is not None and len(kpts_d) > 0:
            n_show = min(100, len(kpts_q))
            indices = np.linspace(0, len(kpts_q) - 1, n_show, dtype=int)
            for i in indices:
                x1, y1 = kpts_q[i]
                x2, y2 = kpts_d[i]
                color = cmap(norm(conf[i])) if has_conf else 'lime'
                ax3.plot([x1, x2 + w1], [y1, y2], color=color, alpha=0.5, linewidth=0.5)
        ax3.set_title(f'{name}: Correspondences', fontsize=9)
        ax3.axis('off')

    # Colorbar on last row
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:, 2].tolist(), fraction=0.03, pad=0.01)
    cbar.set_label('Confidence', fontsize=9)

    error_str = f" | Error: {error_3d:.1f}m (2D: {error_2d:.1f}m)" if error_3d is not None else ""
    inlier_str = f" | Total inliers: {total_inliers}" if total_inliers is not None else ""
    title_str = title if title else f"Frame {frame_id} — Multi-Crop Matching"
    fig.suptitle(f'{title_str}{inlier_str}{error_str}', fontsize=13, fontweight='bold')

    plt.tight_layout()
    save_path = output_dir / "keyframes" / f"keyframe_{frame_id:04d}_crops.{fig_ext}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=80, bbox_inches='tight')
    fig.clf()
    plt.close(fig)
    gc.collect()


# ------------------------------------------------------------------ #
#  Summary / results plots                                             #
# ------------------------------------------------------------------ #

def plot_results(
    results: List[FrameResult],
    filename: str,
    keyframe_min_points: int = 100,
    fig_ext: str = "png",
):
    """6-panel summary plot (errors, trajectory, histograms)."""
    successful = [r for r in results if r.success and r.position_error is not None]
    if not successful:
        print("No results with ground-truth errors — skipping error summary plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    colors = {'keyframe': 'red', 'tracked': 'blue', 'predicted': 'orange'}

    # 1. 3D position error
    ax1 = axes[0, 0]
    for r in successful:
        ax1.scatter(r.frame_id, r.position_error, c=colors.get(r.method, 'gray'), s=30, alpha=0.7)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('3D Position Error (m)')
    ax1.set_title('3D Position Error (Red=KF, Blue=Tracked, Orange=Predicted)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, min(100, max(r.position_error for r in successful) * 1.1))

    # 2. Rotation error
    ax2 = axes[0, 1]
    has_rot = [r for r in successful if r.rotation_error is not None]
    if has_rot:
        for r in has_rot:
            ax2.scatter(r.frame_id, r.rotation_error, c=colors.get(r.method, 'gray'), s=30, alpha=0.7)
        ax2.set_ylabel('Rotation Error (deg)')
        ax2.set_ylim(0, min(180, max(r.rotation_error for r in has_rot) * 1.1))
    else:
        ax2.text(0.5, 0.5, 'No rotation errors', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_xlabel('Frame')
    ax2.set_title('Rotation Error')
    ax2.grid(True, alpha=0.3)

    # 3. Trajectory
    ax3 = axes[0, 2]
    gt_x = [r.gt_x for r in successful]
    gt_y = [r.gt_y for r in successful]
    est_x = [r.est_x for r in successful]
    est_y = [r.est_y for r in successful]
    ax3.plot(gt_x, gt_y, 'g-', label='GT', linewidth=2)
    ax3.plot(est_x, est_y, 'b-', label='Est', linewidth=1.5, alpha=0.7)
    kf_x = [r.est_x for r in successful if r.is_keyframe]
    kf_y = [r.est_y for r in successful if r.is_keyframe]
    ax3.scatter(kf_x, kf_y, c='red', s=50, marker='o', label='KF', zorder=5)
    ax3.set_xlabel('UTM E (m)')
    ax3.set_ylabel('UTM N (m)')
    ax3.set_title('Trajectory')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    # 4. Tracked points & inliers
    ax4 = axes[1, 0]
    frames = [r.frame_id for r in results]
    tracked_pts = [r.num_tracked_points for r in results]
    inliers = [r.num_inliers for r in results]
    bw = 0.35
    fa = np.array(frames)
    ax4.bar(fa - bw / 2, tracked_pts, bw, alpha=0.7, color='steelblue', label='Tracked')
    ax4.bar(fa + bw / 2, inliers, bw, alpha=0.7, color='green', label='PnP Inliers')
    thresholds = [r.tracked_points_threshold for r in results]
    ax4.step(frames, thresholds, where='post', color='red', linestyle='--', alpha=0.8, label='Threshold (25%)')
    ax4.axhline(keyframe_min_points, color='black', linestyle=':', alpha=0.5,
                label=f'Min ({keyframe_min_points})')
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Count')
    ax4.set_title('Tracked Points & PnP Inliers')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Error histogram
    ax5 = axes[1, 1]
    errors = [r.position_error for r in successful]
    ax5.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    ax5.axvline(np.median(errors), color='r', linestyle='--', label=f'Median: {np.median(errors):.2f}m')
    ax5.set_xlabel('3D Position Error (m)')
    ax5.set_ylabel('Count')
    ax5.set_title('3D Error Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Rotation error histogram
    ax6 = axes[1, 2]
    if has_rot:
        rot_errors = [r.rotation_error for r in has_rot]
        ax6.hist(rot_errors, bins=30, alpha=0.7, edgecolor='black', color='coral')
        ax6.axvline(np.median(rot_errors), color='r', linestyle='--',
                     label=f'Median: {np.median(rot_errors):.2f}deg')
        ax6.set_xlabel('Rotation Error (deg)')
        ax6.set_ylabel('Count')
        ax6.set_title('Rotation Error Distribution')
    else:
        ax6.text(0.5, 0.5, 'No rotation errors', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Rotation Error Distribution')
    ax6.grid(True, alpha=0.3)

    # Summary suptitle
    pos_errors = [r.position_error for r in successful]
    times = [r.processing_time for r in results if r.processing_time > 0]
    total_time = sum(times) if times else 1.0
    fps = len(results) / total_time
    mean_err = np.mean(pos_errors)
    median_err = np.median(pos_errors)
    n_kf = sum(1 for r in results if r.is_keyframe)
    n_tr = sum(1 for r in results if r.method == 'tracked')

    fig.suptitle(
        f"Mean: {mean_err:.2f}m | Median: {median_err:.2f}m | "
        f"FPS: {fps:.2f} ({total_time:.1f}s) | "
        f"KF: {n_kf}  Tracked: {n_tr}  Frames: {len(results)}",
        fontsize=13, fontweight='bold', y=1.01,
    )

    plt.tight_layout()
    # Replace extension with fig_ext if different from default
    output_path = Path(filename)
    if fig_ext != "png":
        output_path = output_path.with_suffix(f".{fig_ext}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    fig.clf()
    plt.close(fig)
    gc.collect()
    print(f"Plot saved to {output_path}")


# ================================================================== #
#  Per-stage debug visualizations                                     #
# ================================================================== #

class StageDebugVisualizer:
    """Save per-stage debug figures to *output_dir*.

    Parameters
    ----------
    output_dir:   Root directory; one sub-dir ``stageN/`` per stage is created.
    geo_handler:  Used to project 3-D points onto the DOP map.
    intrinsics:   Camera intrinsics (for reprojection overlays).
    lod:          Optional :class:`utils.lod.LoD` instance for edge overlays."""

    def __init__(
        self,
        output_dir: str,
        geo_handler: Any,
        intrinsics: Any,
        lod=None,  # Optional[LoD]
    ) -> None:
        self.output_dir  = Path(output_dir)
        self.geo_handler = geo_handler
        self.intrinsics  = intrinsics
        self.lod = lod
        self._gt_position: Optional[np.ndarray] = None  # set per-frame via set_gt_pose
        self._gt_R_c2w:    Optional[np.ndarray] = None  # set per-frame via set_gt_pose
        self._gt_fov:      Optional[float]       = None  # reference FoV (from intrinsics.json)
        self._coarse_fov:  Optional[float]       = None  # set by coarse_stage for fine to use
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_gt_pose(self, position: Optional[np.ndarray],
                    R_c2w: Optional[np.ndarray] = None) -> None:
        """Set GT position (UTM x, y, z) and rotation for the current frame.

        Call this before the stage methods so GT can be overlaid as reference."""
        self._gt_position = np.array(position, dtype=float) if position is not None else None
        self._gt_R_c2w    = np.array(R_c2w,    dtype=float) if R_c2w    is not None else None

    def set_gt_fov(self, fov_deg: Optional[float]) -> None:
        """Set the reference (GT) FoV so calibration error can be displayed."""
        self._gt_fov = float(fov_deg) if fov_deg is not None else None

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #


    def _save_named(self, frame_id: int, name: str, fig) -> None:
        import matplotlib.pyplot as plt
        path = self.output_dir / f"frame_{frame_id:06d}_{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=80, bbox_inches="tight")
        plt.close(fig)

    def _footprint_center(self, position: np.ndarray, R_c2w: np.ndarray,
                          image_size=None):
        """Return (fx, fy) — visible footprint centre from ``compute_visible_dop_crop``."""
        if image_size is None or not hasattr(self.intrinsics, 'fov_vertical'):
            raise ValueError("_footprint_center requires image_size and intrinsics.fov_vertical")
        from orthotrack.crop_strategy import compute_visible_dop_crop
        cx, cy, _size = compute_visible_dop_crop(
            position, R_c2w, image_size,
            self.intrinsics.fov_vertical, self.geo_handler,
            K=self.intrinsics.K if hasattr(self.intrinsics, 'K') else None)
        return float(cx), float(cy)

    def _footprint_polygon_corners(
        self,
        position: np.ndarray,
        R_c2w: np.ndarray,
        image_size,
        ground_z: float,
    ):
        """Project the 4 image corners onto the ground plane at *ground_z*.

        Returns a list of 4 (x, y) UTM tuples forming the footprint quadrilateral,
        or None if the projection fails (e.g. corners behind camera)."""
        h, w = image_size
        K = getattr(self.intrinsics, 'K', None)
        if K is None:
            from orthotrack.crop_strategy import get_intrinsics as _gi
            K = _gi((h, w), self.intrinsics.fov_vertical)
        corners_px = np.array(
            [[0.5, 0.5], [w - 0.5, 0.5], [w - 0.5, h - 0.5], [0.5, h - 0.5]],
            dtype=float,
        )
        K_inv = np.linalg.inv(K)
        dirs_cam = (K_inv @ np.hstack([corners_px, np.ones((4, 1))]).T).T  # (4,3)
        dirs_world = (R_c2w @ dirs_cam.T).T
        dirs_world /= np.linalg.norm(dirs_world, axis=1, keepdims=True)
        O = position
        pts = []
        for D in dirs_world:
            dz = D[2]
            if abs(dz) < 1e-6:
                return None
            t = (ground_z - O[2]) / dz
            if t < 0:
                return None
            P = O + t * D
            pts.append((float(P[0]), float(P[1])))
        return pts if len(pts) == 4 else None

    def render_lod_overlay(self, R_c2w, position, image_size):
        """Render LoD structural edges as a (H, W, 4) RGBA float32 overlay, or None on failure."""
        if self.lod is None:
            return None
        try:
            from scipy.ndimage import binary_dilation
            H, W = int(image_size[0]), int(image_size[1])
            R_w2c = np.asarray(R_c2w, dtype=np.float64).T
            pos   = np.asarray(position, dtype=np.float64)
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = R_w2c
            w2c[:3, 3]  = -R_w2c @ pos
            K = np.asarray(self.intrinsics.K, dtype=np.float64)
            try:
                depth = self.lod.render_depth(w2c, K, (H, W), device="gpu")
            except Exception:
                depth = self.lod.render_depth(w2c, K, (H, W), device="cpu")
            edge_map = self.lod.render_edges(w2c, K, (H, W), depth=depth)
            edge_map = binary_dilation(edge_map.astype(bool)).astype(np.float32)
            wire_rgba = np.zeros((H, W, 4), dtype=np.float32)
            wire_rgba[:, :, 1] = edge_map   # G  → cyan = (0, 1, 1)
            wire_rgba[:, :, 2] = edge_map   # B
            wire_rgba[:, :, 3] = edge_map * 0.95
            return wire_rgba
        except Exception:
            return None

    def _lod_segments(self, ax, R_c2w, position, image_size):
        """Overlay LoD structural edges on UAV image axes using dense edge rendering."""
        wire_rgba = self.render_lod_overlay(R_c2w, position, image_size)
        if wire_rgba is None:
            return
        H, W = wire_rgba.shape[:2]
        ax.imshow(wire_rgba, extent=[-0.5, W - 0.5, H - 0.5, -0.5], zorder=2)

    def _lod_on_dop(self, ax, dop_r, R_c2w, position, image_size,
                    color, linewidth=0.5, alpha=0.55, max_segs=8000):
        """Overlay LoD edges on the DOP panel using camera visibility (top-down view)."""
        if self.lod is None:
            return
        K = getattr(self.intrinsics, 'K', None)
        if K is None:
            return
        try:
            from matplotlib.collections import LineCollection
            H, W = int(image_size[0]), int(image_size[1])
            R_w2c = np.asarray(R_c2w, dtype=np.float64).T
            pos   = np.asarray(position, dtype=np.float64)
            verts = self.lod.vertices_abs.astype(np.float64)
            fcs   = self.lod.faces.astype(np.int32)

            # Camera-space transform for visibility
            pts_c    = (R_w2c @ (verts - pos).T).T
            in_front = pts_c[:, 2] > 0.5
            # Project to image plane for frustum check
            px_img = np.full(len(verts), np.nan)
            py_img = np.full(len(verts), np.nan)
            x_h = (K @ pts_c[in_front].T).T
            px_img[in_front] = x_h[:, 0] / x_h[:, 2]
            py_img[in_front] = x_h[:, 1] / x_h[:, 2]
            mg = max(W, H) * 0.15
            in_view = (in_front &
                       (px_img >= -mg) & (px_img <= W + mg) &
                       (py_img >= -mg) & (py_img <= H + mg))

            # Map vertices to DOP pixel coords (orthographic: use world x,y)
            bw = dop_r.bounds[2] - dop_r.bounds[0]
            bh = dop_r.bounds[3] - dop_r.bounds[1]
            dh, dw = dop_r.data.shape[:2]
            dop_px = (verts[:, 0] - dop_r.bounds[0]) / bw * dw
            dop_py = (dop_r.bounds[3] - verts[:, 1]) / bh * dh

            # Build all 3 edges per face (fast, no deduplication needed for vis)
            a = np.concatenate([fcs[:, 0], fcs[:, 1], fcs[:, 0]])
            b = np.concatenate([fcs[:, 1], fcs[:, 2], fcs[:, 2]])

            # Keep edges where at least one endpoint is visible from this camera
            ok = in_view[a] | in_view[b]
            a, b = a[ok], b[ok]
            xa, ya = dop_px[a], dop_py[a]
            xb, yb = dop_px[b], dop_py[b]

            # Keep segments within DOP bounds
            in_dop = (
                (np.minimum(xa, xb) < dw * 1.05) & (np.maximum(xa, xb) > -dw * 0.05) &
                (np.minimum(ya, yb) < dh * 1.05) & (np.maximum(ya, yb) > -dh * 0.05)
            )
            xa, ya, xb, yb = xa[in_dop], ya[in_dop], xb[in_dop], yb[in_dop]
            if len(xa) == 0:
                return
            if len(xa) > max_segs:
                idx = np.round(np.linspace(0, len(xa) - 1, max_segs)).astype(np.int32)
                xa, ya, xb, yb = xa[idx], ya[idx], xb[idx], yb[idx]
            segs = np.stack([np.column_stack([xa, ya]),
                             np.column_stack([xb, yb])], axis=1)
            ax.add_collection(LineCollection(segs, linewidths=linewidth,
                                             colors=color, alpha=alpha))
        except Exception:
            pass

    def _reproj_errors(
        self,
        pts3d: np.ndarray,
        pts2d: np.ndarray,
        R_c2w: np.ndarray,
        position: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Return per-point reprojection errors (px) or None."""
        K = getattr(self.intrinsics, 'K', None)
        if (K is None or pts3d is None or pts2d is None
                or R_c2w is None or position is None
                or len(pts3d) == 0):
            return None
        pts3d = np.asarray(pts3d, dtype=float)
        pts2d = np.asarray(pts2d, dtype=float)
        R_w2c = np.asarray(R_c2w, dtype=float).T
        pos   = np.asarray(position, dtype=float)
        pts_c = (R_w2c @ (pts3d - pos).T).T
        ok    = pts_c[:, 2] > 0.1
        if not np.any(ok):
            return None
        x_h     = (K @ pts_c[ok].T).T
        p_proj  = x_h[:, :2] / x_h[:, 2:3]
        errs    = np.full(len(pts3d), np.nan)
        errs[ok] = np.linalg.norm(p_proj - pts2d[ok], axis=1)
        return errs

    def _pose_delta_str(
        self,
        position: Optional[np.ndarray],
        R_c2w: Optional[np.ndarray],
    ) -> str:
        """Return a short Δpose string vs GT, or empty string if no GT."""
        if position is None or self._gt_position is None:
            return ""
        dxy  = float(np.linalg.norm(position[:2] - self._gt_position[:2]))
        dz   = float(abs(position[2] - self._gt_position[2]))
        dtrs = float(np.linalg.norm(position - self._gt_position))
        s    = f"\u0394xy={dxy:.1f}m  \u0394z={dz:.1f}m  |t|={dtrs:.1f}m"
        if R_c2w is not None and self._gt_R_c2w is not None:
            Rd   = R_c2w @ self._gt_R_c2w.T
            tr   = np.clip(float(np.trace(Rd)), -1.0, 3.0)
            rdeg = float(np.rad2deg(np.arccos((tr - 1) / 2)))
            s   += f"  rot={rdeg:.1f}\u00b0"
        return s


    # ------------------------------------------------------------------ #
    #  Primary stage visualizations                                        #
    # ------------------------------------------------------------------ #

    def _draw_dop_footprint(
        self,
        ax,
        image,
        position,        # estimated camera position (may be None)
        R_c2w,           # estimated rotation (may be None)
        pts_2d,          # query keypoints (N,2) for scatter
        pts_3d,          # 3D correspondences (N,3) for DOP scatter + hull
        pts_cf,          # confidence per point (N,)
        n_inliers: int,
        title_prefix: str,
        *,
        show_gt: bool = True,
        fov_delta_str: str = "",
    ) -> None:
        """Draw a DOP footprint panel onto *ax* (right-column panel)."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon as MplPoly
            h_img, w_img = image.shape[:2]
            _CONF_CMAP = 'plasma'
            _CONF_VMIN, _CONF_VMAX = 0.0, 1.0

            have_pose = position is not None

            # Anchor: footprint centre or correspondence centroid
            if have_pose:
                fp_cx, fp_cy = (
                    self._footprint_center(position, R_c2w, image_size=(h_img, w_img))
                    if R_c2w is not None
                    else (float(position[0]), float(position[1]))
                )
            else:
                fp_cx = float(np.median(pts_3d[:, 0])) if pts_3d is not None and len(pts_3d) >= 3 else 0.0
                fp_cy = float(np.median(pts_3d[:, 1])) if pts_3d is not None and len(pts_3d) >= 3 else 0.0

            # Pre-compute polygon corners
            _pred_corners_utm = None
            if have_pose and R_c2w is not None and pts_3d is not None and len(pts_3d) >= 3:
                _gz = float(np.median(pts_3d[:, 2]))
                _pred_corners_utm = self._footprint_polygon_corners(
                    position, R_c2w, (h_img, w_img), _gz)

            gt_fp_cx, gt_fp_cy = None, None
            _gt_corners_utm = None
            if show_gt and self._gt_position is not None:
                if self._gt_R_c2w is not None:
                    gt_fp_cx, gt_fp_cy = self._footprint_center(
                        self._gt_position, self._gt_R_c2w, image_size=(h_img, w_img))
                    _gt_gz = (float(np.median(pts_3d[:, 2]))
                              if pts_3d is not None and len(pts_3d) >= 3
                              else float(position[2]) - 300.0 if have_pose else 0.0)
                    _gt_corners_utm = self._footprint_polygon_corners(
                        self._gt_position, self._gt_R_c2w, (h_img, w_img), _gt_gz)
                else:
                    gt_fp_cx = float(self._gt_position[0])
                    gt_fp_cy = float(self._gt_position[1])

            # Compute DOP extent
            if pts_3d is not None and len(pts_3d) > 0:
                all_x = list(pts_3d[:, 0]) + [fp_cx]
                all_y = list(pts_3d[:, 1]) + [fp_cy]
                if have_pose:
                    all_x.append(float(position[0]))
                    all_y.append(float(position[1]))
                if _pred_corners_utm is not None:
                    all_x += [c[0] for c in _pred_corners_utm]
                    all_y += [c[1] for c in _pred_corners_utm]
                if self._gt_position is not None and show_gt:
                    all_x.append(float(self._gt_position[0]))
                    all_y.append(float(self._gt_position[1]))
                if gt_fp_cx is not None:
                    all_x.append(gt_fp_cx); all_y.append(gt_fp_cy)
                if _gt_corners_utm is not None:
                    all_x += [c[0] for c in _gt_corners_utm]
                    all_y += [c[1] for c in _gt_corners_utm]
                extent = max(max(all_x) - min(all_x), max(all_y) - min(all_y))
                dop_cx = (max(all_x) + min(all_x)) / 2
                dop_cy = (max(all_y) + min(all_y)) / 2
                dop_size = max(extent * 1.3, 100.0)
            elif have_pose:
                dop_cx, dop_cy, dop_size = fp_cx, fp_cy, 200.0
            else:
                ax.axis('off'); ax.set_title(f"{title_prefix} (no data)", fontsize=9); return

            dop_r = self.geo_handler.crop_dop(dop_cx, dop_cy, dop_size)
            if dop_r is None:
                ax.axis('off'); ax.set_title(f"{title_prefix} (DOP unavailable)", fontsize=9); return

            ax.imshow(dop_r.data)
            bw = dop_r.bounds[2] - dop_r.bounds[0]
            bh = dop_r.bounds[3] - dop_r.bounds[1]
            iw, ih = dop_r.data.shape[1], dop_r.data.shape[0]

            # 3D correspondence scatter (colored by confidence)
            if pts_3d is not None and len(pts_3d) > 0:
                px = (pts_3d[:, 0] - dop_r.bounds[0]) / bw * iw
                py = (dop_r.bounds[3] - pts_3d[:, 1]) / bh * ih
                _colors = (pts_cf if (pts_cf is not None and len(pts_cf) == len(pts_3d)) else 'cyan')
                ax.scatter(px, py, s=4, c=_colors, cmap=_CONF_CMAP,
                           vmin=_CONF_VMIN, vmax=_CONF_VMAX, linewidths=0, alpha=0.7)

            # Footprint polygon
            _fp_poly_drawn = False
            if _pred_corners_utm is not None:
                _cpx = [(cx_ - dop_r.bounds[0]) / bw * iw for cx_, _ in _pred_corners_utm]
                _cpy = [(dop_r.bounds[3] - cy_) / bh * ih for _, cy_ in _pred_corners_utm]
                _poly = MplPoly(list(zip(_cpx, _cpy)), closed=True, fill=False,
                                edgecolor='red', linewidth=1.8, linestyle='-',
                                alpha=0.85, zorder=5, label='pred footprint')
                ax.add_patch(_poly)
                ax.plot(sum(_cpx) / 4, sum(_cpy) / 4, 'r+', markersize=14, markeredgewidth=2)
                _fp_poly_drawn = True
            if not _fp_poly_drawn and pts_3d is not None and len(pts_3d) >= 3:
                try:
                    from scipy.spatial import ConvexHull as _CH
                    _pts2d = np.column_stack([
                        (pts_3d[:, 0] - dop_r.bounds[0]) / bw * iw,
                        (dop_r.bounds[3] - pts_3d[:, 1]) / bh * ih,
                    ])
                    _hull = _CH(_pts2d)
                    _hv = _pts2d[_hull.vertices]
                    _poly = MplPoly(_hv, closed=True, fill=False, edgecolor='orange',
                                    linewidth=1.8, linestyle='--', alpha=0.85,
                                    zorder=5, label='corr hull')
                    ax.add_patch(_poly)
                    ax.plot(float(_hv[:, 0].mean()), float(_hv[:, 1].mean()),
                            color='orange', marker='+', markersize=14, markeredgewidth=2)
                except Exception:
                    pass

            # GT footprint polygon
            if _gt_corners_utm is not None:
                _gcpx = [(cx_ - dop_r.bounds[0]) / bw * iw for cx_, _ in _gt_corners_utm]
                _gcpy = [(dop_r.bounds[3] - cy_) / bh * ih for _, cy_ in _gt_corners_utm]
                _gt_poly = MplPoly(list(zip(_gcpx, _gcpy)), closed=True, fill=False,
                                   edgecolor='lime', linewidth=1.8, linestyle='-',
                                   alpha=0.85, zorder=5, label='GT footprint')
                ax.add_patch(_gt_poly)
                ax.plot(sum(_gcpx) / 4, sum(_gcpy) / 4, color='lime',
                        marker='+', markersize=14, markeredgewidth=2)

            # Camera markers
            if have_pose:
                cam_px = (float(position[0]) - dop_r.bounds[0]) / bw * iw
                cam_py = (dop_r.bounds[3] - float(position[1])) / bh * ih
                ax.plot(cam_px, cam_py, 'rx', markersize=12, markeredgewidth=2, label='pred cam')
            if show_gt and self._gt_position is not None:
                gcam_px = (float(self._gt_position[0]) - dop_r.bounds[0]) / bw * iw
                gcam_py = (dop_r.bounds[3] - float(self._gt_position[1])) / bh * ih
                ax.plot(gcam_px, gcam_py, 'gx', markersize=12, markeredgewidth=2, label='GT cam')

            # Title
            _fp_src = 'cam pose' if _fp_poly_drawn else 'corr hull'
            if have_pose:
                _t = [f"{title_prefix}  z={position[2]:.0f}m  inl={n_inliers}  [{_fp_src}]"]
                if show_gt and self._gt_position is not None:
                    xy_err = float(np.linalg.norm(position[:2] - self._gt_position[:2]))
                    _t.append(f"\u0394xy={xy_err:.1f}m")
                if fov_delta_str:
                    _t.append(fov_delta_str)
            else:
                _t = [f"{title_prefix}  [FAILED — {_fp_src}]"]
                if fov_delta_str:
                    _t.append(fov_delta_str)
            ax.set_title("  ".join(_t), fontsize=9)
            ax.legend(fontsize=7, loc='upper right')
        except Exception:
            ax.set_title(f"{title_prefix} (error)", fontsize=9)
        ax.axis('off')

    def coarse_stage(
        self,
        frame_id: int,
        image: np.ndarray,
        # full-DOP match (coarse search)
        all_kpts_q: Optional[np.ndarray] = None,
        all_kpts_dop: Optional[np.ndarray] = None,
        all_confs: Optional[np.ndarray] = None,
        utm_x: Optional[float] = None,
        utm_y: Optional[float] = None,

        fd_accepted: bool = False,
        # fd PnP result (even when not accepted, for visualization)
        fd_2d: Optional[np.ndarray] = None,
        fd_3d: Optional[np.ndarray] = None,
        fd_cf: Optional[np.ndarray] = None,
        fd_position: Optional[np.ndarray] = None,
        fd_R_c2w: Optional[np.ndarray] = None,
        fd_inl: int = 0,
        fd_med_conf: float = 0.0,
        fd_inl_idx: Optional[np.ndarray] = None,
        # tile-search fallback
        tile_kpts_q: Optional[np.ndarray] = None,
        tile_kpts_dop: Optional[np.ndarray] = None,
        tile_confs: Optional[np.ndarray] = None,
        tile_crop: Any = None,
        tile_accepted: bool = False,
        tile_inl_idx: Optional[np.ndarray] = None,  # boolean mask or index array for PnP inliers
        tile_position: Optional[np.ndarray] = None,  # tile PnP camera position
        tile_R_c2w: Optional[np.ndarray] = None,     # tile PnP rotation
        tile_score_map: Optional[List] = None,
        confidence_threshold: float = 0.4,
        confidence_min_count: int = 50,
        # final rough pose (accepted from either 1a or 1b)
        rough_position: Optional[np.ndarray] = None,
        rough_R_c2w: Optional[np.ndarray] = None,
        roi_inl: int = 0,
        calib_fov: Optional[float] = None,
        roi_2d: Optional[np.ndarray] = None,
        roi_3d: Optional[np.ndarray] = None,
        roi_cf: Optional[np.ndarray] = None,

        coarse_matcher_name: Optional[str] = None,
    ) -> None:
        """Coarse stage: 2-row figure saved to ``coarse/``.

        Row 0 — Full-DOP match: image with correspondences (left) + DOP footprint (right).
        Row 1 — Tile search:    image with correspondences (left) + DOP tile heatmap (right).
                 Shown even when bypassed (labels the reason)."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            h_img, w_img = image.shape[:2]
            aspect = w_img / h_img
            fig = plt.figure(figsize=(aspect * 10, 16))
            gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.30, wspace=0.08,
                                    height_ratios=[1, 1, 0.85])

            _CONF_CMAP = 'plasma'
            _CONF_VMIN, _CONF_VMAX = 0.0, 1.0
            full_dop_img = None
            full_dop_scale = 1.0
            try:
                from orthotrack.localization import get_full_dop_image
                _fdop = get_full_dop_image(self.geo_handler)
                if _fdop is not None:
                    _fh, _fw = _fdop.shape[:2]
                    _max_side = 1024
                    if max(_fh, _fw) > _max_side:
                        full_dop_scale = _max_side / max(_fh, _fw)
                        import cv2 as _cv2
                        full_dop_img = _cv2.resize(
                            _fdop,
                            (int(_fw * full_dop_scale), int(_fh * full_dop_scale)),
                            interpolation=_cv2.INTER_AREA)
                    else:
                        full_dop_img = _fdop
            except Exception:
                pass
            _sm = f" [{coarse_matcher_name}]" if coarse_matcher_name else ""

            # ─── Row 0: full-DOP match ────────────────────────────────────
            n_all = len(all_kpts_q) if all_kpts_q is not None else 0
            c_all = (all_confs if (all_confs is not None and len(all_confs) == n_all)
                     else (np.ones(n_all) if n_all > 0 else None))
            fd_had_region = n_all >= 20
            if fd_accepted:
                lbl1 = "PnP accepted"
            elif fd_had_region:
                lbl1 = "PnP failed (region kept)"
            else:
                lbl1 = "no region"

            ax00 = fig.add_subplot(gs[0, 0])
            ax00.imshow(image)
            # Show fd_2d (PnP-ready correspondences): outliers red, inliers by confidence
            if fd_2d is not None and len(fd_2d) > 0:
                fd_2d_arr = np.asarray(fd_2d)
                fd_cf_arr = np.asarray(fd_cf) if fd_cf is not None and len(fd_cf) == len(fd_2d_arr) else np.ones(len(fd_2d_arr))
                # Build inlier mask
                if fd_inl_idx is not None and len(fd_inl_idx) > 0:
                    _inl_mask = np.zeros(len(fd_2d_arr), dtype=bool)
                    _inl_mask[fd_inl_idx] = True
                else:
                    _inl_mask = np.zeros(len(fd_2d_arr), dtype=bool)
                # Draw outliers in red
                if np.any(~_inl_mask):
                    ax00.scatter(fd_2d_arr[~_inl_mask, 0], fd_2d_arr[~_inl_mask, 1],
                                 s=2, c='red', linewidths=0, alpha=0.5, zorder=2)
                # Draw inliers by confidence
                if np.any(_inl_mask):
                    ax00.scatter(fd_2d_arr[_inl_mask, 0], fd_2d_arr[_inl_mask, 1],
                                 s=3, c=fd_cf_arr[_inl_mask], cmap=_CONF_CMAP,
                                 vmin=_CONF_VMIN, vmax=_CONF_VMAX, linewidths=0, alpha=0.85, zorder=3)
                # Stats: line 1 = raw correspondences, line 2 = inlier subset
                _n_raw = len(fd_2d_arr)
                _raw_cf = fd_cf_arr
                _stats1 = (f"raw:  n={_n_raw}  med={np.median(_raw_cf):.2f}  mean={_raw_cf.mean():.2f}"
                           f"  min={_raw_cf.min():.2f}  max={_raw_cf.max():.2f}")
                _n_inl = int(_inl_mask.sum())
                if _n_inl > 0:
                    _inl_cf = fd_cf_arr[_inl_mask]
                    _stats2 = (f"inl:  n={_n_inl}  med={np.median(_inl_cf):.2f}  mean={_inl_cf.mean():.2f}"
                               f"  min={_inl_cf.min():.2f}  max={_inl_cf.max():.2f}")
                else:
                    _stats2 = f"inl:  n=0"
                ax00.text(0.01, 0.01, f"{_stats1}\n{_stats2}", transform=ax00.transAxes,
                          fontsize=7, color='white', va='bottom',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
            elif all_kpts_q is not None and n_all > 0:
                # Fall back to showing all raw matches if no fd PnP data
                ax00.scatter(all_kpts_q[:, 0], all_kpts_q[:, 1],
                             s=2, c=c_all, cmap=_CONF_CMAP,
                             vmin=_CONF_VMIN, vmax=_CONF_VMAX, linewidths=0, alpha=0.7)
                _cf = np.asarray(c_all)
                _stats = (f"raw:  n={n_all}  med={np.median(_cf):.2f}  mean={_cf.mean():.2f}"
                          f"  min={_cf.min():.2f}  max={_cf.max():.2f}")
                ax00.text(0.01, 0.01, _stats, transform=ax00.transAxes,
                          fontsize=7, color='white', va='bottom',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
            _pnp_note = f"  PnP_inl={fd_inl}" if fd_accepted else ""
            _fd_has_lod = (self.lod is not None
                           and fd_position is not None and fd_R_c2w is not None)
            if _fd_has_lod:
                self._lod_segments(ax00, fd_R_c2w, fd_position, (h_img, w_img))
            ax00.set_title(
                f"Coarse: full-DOP{' + LoD' if _fd_has_lod else ''}{_sm} [{lbl1}]{_pnp_note}",
                fontsize=9)
            ax00.axis('off')

            # Row 0 right: DOP footprint for fd result
            ax01 = fig.add_subplot(gs[0, 1])
            ax01.axis('off')
            # Build raw 3D points from all_kpts_dop for the footprint hull
            _raw_pts3d = None
            _raw_pts_cf = None
            if all_kpts_dop is not None and len(all_kpts_dop) >= 3:
                try:
                    from orthotrack.localization import sample_full_dsm_batch
                    if hasattr(self.geo_handler, 'dop_transform'):
                        _t = self.geo_handler.dop_transform
                        _utm_xs = _t.a * all_kpts_dop[:, 0] + _t.b * all_kpts_dop[:, 1] + _t.c
                        _utm_ys = _t.d * all_kpts_dop[:, 0] + _t.e * all_kpts_dop[:, 1] + _t.f
                    elif hasattr(self.geo_handler, 'pixel_to_utm_batch'):
                        _utm_xs, _utm_ys = self.geo_handler.pixel_to_utm_batch(all_kpts_dop[:, 0], all_kpts_dop[:, 1])
                    else:
                        _utm_xs = np.array([self.geo_handler.pixel_to_utm(kp[0], kp[1])[0] for kp in all_kpts_dop])
                        _utm_ys = np.array([self.geo_handler.pixel_to_utm(kp[0], kp[1])[1] for kp in all_kpts_dop])
                    _zs = sample_full_dsm_batch(self.geo_handler, _utm_xs, _utm_ys)
                    _valid = ~np.isnan(_zs) & (_zs > -100)
                    if _valid.sum() >= 3:
                        _raw_pts3d = np.column_stack([_utm_xs[_valid], _utm_ys[_valid], _zs[_valid]])
                        _raw_pts_cf = (np.asarray(all_confs)[_valid]
                                       if all_confs is not None and len(all_confs) == len(all_kpts_dop)
                                       else np.ones(_valid.sum(), dtype=np.float32))
                except Exception:
                    pass
            _fp_pts3d = _raw_pts3d if _raw_pts3d is not None else fd_3d
            _fp_cf = _raw_pts_cf if _raw_pts_cf is not None else fd_cf
            _fd_fov_delta = (f"\u0394FoV={calib_fov - self._gt_fov:+.1f}\u00b0"
                             if calib_fov is not None and self._gt_fov is not None
                             else "")
            if _fp_pts3d is not None and len(_fp_pts3d) >= 3:
                self._draw_dop_footprint(
                    ax01, image,
                    position=fd_position,
                    R_c2w=fd_R_c2w,
                    pts_2d=fd_2d,
                    pts_3d=_fp_pts3d,
                    pts_cf=_fp_cf,
                    n_inliers=fd_inl,
                    title_prefix="DOP footprint (fd)",
                    fov_delta_str=_fd_fov_delta,
                )
            elif full_dop_img is not None:
                ax01.imshow(full_dop_img)
                ax01.set_title("DOP footprint (fd — no 3D)", fontsize=9)
            else:
                ax01.set_title("DOP footprint (unavailable)", fontsize=9)

            # ─── Row 1: tile-search ───────────────────────────────────────
            tile2_ran = tile_kpts_q is not None or (tile_score_map is not None and len(tile_score_map) > 0)
            n_tile = len(tile_kpts_q) if tile_kpts_q is not None else 0
            c_tile = (tile_confs if (tile_confs is not None and len(tile_confs) == n_tile)
                      else (np.ones(n_tile) if n_tile > 0 else None))
            if tile_accepted:
                lbl2 = "accepted"
            elif tile2_ran:
                lbl2 = "rejected"
            elif fd_had_region:
                lbl2 = "bypassed (fd had region)"
            else:
                lbl2 = "not run"

            ax10 = fig.add_subplot(gs[1, 0])
            ax10.imshow(image)
            _n_tile_inl = 0  # track inlier count for use in row-1-right
            if tile_kpts_q is not None and n_tile > 0:
                _ct_arr = np.asarray(c_tile)

                # Build boolean inlier mask
                if tile_inl_idx is not None and len(tile_inl_idx) > 0:
                    _tii = np.asarray(tile_inl_idx)
                    if _tii.dtype == bool:
                        _t_inl_mask = _tii[:n_tile] if len(_tii) >= n_tile else np.pad(
                            _tii, (0, n_tile - len(_tii)))
                    else:
                        _t_inl_mask = np.zeros(n_tile, dtype=bool)
                        _valid_idx = _tii[_tii < n_tile]
                        if len(_valid_idx):
                            _t_inl_mask[_valid_idx] = True
                else:
                    _t_inl_mask = np.zeros(n_tile, dtype=bool)

                # Outliers in red
                if np.any(~_t_inl_mask):
                    ax10.scatter(tile_kpts_q[~_t_inl_mask, 0], tile_kpts_q[~_t_inl_mask, 1],
                                 s=2, c='red', linewidths=0, alpha=0.5, zorder=2)
                # Inliers coloured by confidence
                if np.any(_t_inl_mask):
                    ax10.scatter(tile_kpts_q[_t_inl_mask, 0], tile_kpts_q[_t_inl_mask, 1],
                                 s=3, c=_ct_arr[_t_inl_mask], cmap=_CONF_CMAP,
                                 vmin=_CONF_VMIN, vmax=_CONF_VMAX, linewidths=0, alpha=0.85, zorder=3)

                from matplotlib.colors import Normalize as _Normalize
                from matplotlib.cm import ScalarMappable as _ScalarMappable
                _sm_cbar10 = _ScalarMappable(
                    cmap=_CONF_CMAP,
                    norm=_Normalize(vmin=_CONF_VMIN, vmax=_CONF_VMAX))
                _sm_cbar10.set_array([])
                _cbar_ax10 = ax10.inset_axes([1.01, 0.05, 0.04, 0.90])
                plt.colorbar(_sm_cbar10, cax=_cbar_ax10)
                _cbar_ax10.set_ylabel('confidence', fontsize=7)
                _cbar_ax10.tick_params(labelsize=6)

                _stats1 = (f"raw:  n={n_tile}  med={np.median(_ct_arr):.2f}"
                           f"  mean={_ct_arr.mean():.2f}"
                           f"  min={_ct_arr.min():.2f}  max={_ct_arr.max():.2f}")
                _n_inl_t = int(_t_inl_mask.sum())
                _n_tile_inl = _n_inl_t  # expose for row-1-right
                if _n_inl_t > 0:
                    _inl_cf = _ct_arr[_t_inl_mask]
                    _stats2 = (f"inl:  n={_n_inl_t}  med={np.median(_inl_cf):.2f}"
                               f"  mean={_inl_cf.mean():.2f}"
                               f"  min={_inl_cf.min():.2f}  max={_inl_cf.max():.2f}")
                else:
                    _stats2 = "inl:  n=0"
                # Third line: reprojection errors + Δpose vs GT
                _stats3_parts = []
                if (_n_inl_t > 0 and tile_kpts_dop is not None and tile_kpts_q is not None
                        and tile_position is not None and tile_R_c2w is not None
                        and len(tile_kpts_dop) == n_tile):
                    _re = self._reproj_errors(
                        tile_kpts_dop[_t_inl_mask], tile_kpts_q[_t_inl_mask],
                        tile_R_c2w, tile_position)
                    if _re is not None:
                        _re_ok = _re[np.isfinite(_re)]
                        if len(_re_ok) > 0:
                            _stats3_parts.append(
                                f"reproj:  med={np.median(_re_ok):.1f}px"
                                f"  mean={_re_ok.mean():.1f}px  max={_re_ok.max():.1f}px")
                _dp = self._pose_delta_str(tile_position, tile_R_c2w)
                if _dp:
                    _stats3_parts.append(_dp)
                _stats3 = "  ".join(_stats3_parts)
                _overlay_text = f"{_stats1}\n{_stats2}"
                if _stats3:
                    _overlay_text += f"\n{_stats3}"
                ax10.text(0.01, 0.01, _overlay_text, transform=ax10.transAxes,
                          fontsize=7, color='white', va='bottom',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
            _tile_has_lod = (self.lod is not None
                             and tile_position is not None and tile_R_c2w is not None)
            if _tile_has_lod:
                self._lod_segments(ax10, tile_R_c2w, tile_position, (h_img, w_img))
            ax10.set_title(f"Coarse: tile-search{' + LoD' if _tile_has_lod else ''}{_sm} [{lbl2}]", fontsize=9)
            ax10.axis('off')

            # Row 1 right: DOP footprint (tile) when accepted, otherwise plain DOP
            if tile_accepted and tile_kpts_dop is not None and len(tile_kpts_dop) >= 3:
                ax11 = fig.add_subplot(gs[1, 1])
                self._draw_dop_footprint(
                    ax11, image,
                    position=tile_position,
                    R_c2w=tile_R_c2w,
                    pts_2d=tile_kpts_q,
                    pts_3d=tile_kpts_dop,
                    pts_cf=tile_confs,
                    n_inliers=_n_tile_inl,
                    title_prefix="DOP footprint (tile)",
                )
            else:
                ax11 = fig.add_subplot(gs[1, 1])
                ax11.axis('off')
                if full_dop_img is not None:
                    ax11.imshow(full_dop_img)
                reason = ("rejected" if tile2_ran
                          else "bypassed (fd)" if fd_had_region else "not run")
                ax11.set_title(f"DOP footprint (tile) [{reason}]", fontsize=9)

            # ─── Row 2: tile heatmap (all pyramid levels, spans both cols) ──
            if tile2_ran and tile_score_map and full_dop_img is not None:
                try:
                    import matplotlib.patches as mpatches
                    import matplotlib.colors as mcolors
                    from matplotlib.cm import ScalarMappable
                    from collections import defaultdict

                    _b = self.geo_handler.dop_bounds
                    if hasattr(_b, '__len__'):
                        dop_min_x, dop_min_y = float(_b[0]), float(_b[1])
                        dop_max_x, dop_max_y = float(_b[2]), float(_b[3])
                    else:
                        dop_min_x, dop_min_y = float(_b.left), float(_b.bottom)
                        dop_max_x, dop_max_y = float(_b.right), float(_b.top)
                    dop_w_utm = dop_max_x - dop_min_x
                    dop_h_utm = dop_max_y - dop_min_y
                    img_h_px, img_w_px = full_dop_img.shape[:2]

                    tile_size_m = float(tile_crop[2]) if tile_crop is not None else 100.0
                    sel_cx = float(tile_crop[0]) if tile_crop is not None else None
                    sel_cy = float(tile_crop[1]) if tile_crop is not None else None

                    def _entry_tile_wh(e, fallback=tile_size_m):
                        if len(e) >= 7:
                            return float(e[4]), float(e[5])
                        elif len(e) > 4:
                            ts = float(e[4])
                            return ts, ts
                        return fallback, fallback

                    def _entry_kpts(e):
                        if len(e) >= 7:
                            return e[6]
                        elif len(e) > 5:
                            return e[5]
                        return None

                    tiles_by_level = defaultdict(list)
                    for entry in tile_score_map:
                        tw, th = _entry_tile_wh(entry)
                        tiles_by_level[(tw, th)].append(entry)
                    sorted_level_keys = sorted(tiles_by_level.keys(),
                                               key=lambda k: -(k[0] * k[1]))
                    n_levels_vis = len(sorted_level_keys)

                    all_scores = [float(e[2]) for e in tile_score_map]
                    max_score = max(all_scores) if any(s > 0 for s in all_scores) else 1.0
                    cmap_h = plt.get_cmap('RdYlGn')
                    norm_h = mcolors.Normalize(vmin=0, vmax=max_score)

                    inner_gs = gridspec.GridSpecFromSubplotSpec(
                        1, n_levels_vis, subplot_spec=gs[2, :], wspace=0.04)

                    lbl2_txt = "accepted" if tile_accepted else "rejected"
                    ax_lvl = None
                    for lvl_i, (tw_lvl, th_lvl) in enumerate(sorted_level_keys):
                        ax_lvl = fig.add_subplot(inner_gs[lvl_i])
                        ax_lvl.imshow(full_dop_img)
                        ax_lvl.axis('off')

                        for entry in tiles_by_level[(tw_lvl, th_lvl)]:
                            cx_t, cy_t, score = float(entry[0]), float(entry[1]), float(entry[2])
                            tw_e, th_e = _entry_tile_wh(entry)
                            kpts_vis = _entry_kpts(entry)
                            px_left = (cx_t - tw_e / 2 - dop_min_x) / dop_w_utm * img_w_px
                            py_top  = (dop_max_y - (cy_t + th_e / 2)) / dop_h_utm * img_h_px
                            pw = tw_e / dop_w_utm * img_w_px
                            ph = th_e / dop_h_utm * img_h_px
                            ax_lvl.add_patch(mpatches.Rectangle(
                                (px_left, py_top), pw, ph,
                                linewidth=0.5, edgecolor='gray',
                                facecolor=cmap_h(norm_h(score)), alpha=0.35))
                            if (sel_cx is not None
                                    and abs(cx_t - sel_cx) < 1.0
                                    and abs(cy_t - sel_cy) < 1.0):
                                ax_lvl.add_patch(mpatches.Rectangle(
                                    (px_left, py_top), pw, ph,
                                    linewidth=2.5, edgecolor='lime', facecolor='none'))
                            if kpts_vis is not None and len(kpts_vis) > 0:
                                px_k = (kpts_vis[:, 0] - dop_min_x) / dop_w_utm * img_w_px
                                py_k = (dop_max_y - kpts_vis[:, 1]) / dop_h_utm * img_h_px
                                ax_lvl.scatter(px_k, py_k, c=kpts_vis[:, 2],
                                               cmap='plasma', vmin=0.0, vmax=1.0,
                                               s=1.0, alpha=0.7, linewidths=0)

                        n_lvl = len(tiles_by_level[(tw_lvl, th_lvl)])
                        base_grid_n = max(1, round(dop_w_utm / tw_lvl))
                        title_l1 = (f"L{lvl_i + 1}: {base_grid_n}\u00d7{base_grid_n} base"
                                    f"  ({int(tw_lvl)}m tiles, 50% ovlp)"
                                    f"  n={n_lvl}")
                        if lvl_i == n_levels_vis - 1:
                            title_l1 += f"  [{lbl2_txt}]  thresh={confidence_threshold:.2f}"
                        ax_lvl.set_title(title_l1, fontsize=7)

                    if ax_lvl is not None:
                        sm = ScalarMappable(cmap=cmap_h, norm=norm_h)
                        sm.set_array([])
                        cbar = plt.colorbar(sm, ax=ax_lvl, fraction=0.05, pad=0.02, shrink=0.8)
                        cbar.set_label('median conf', fontsize=6)
                        cbar.ax.tick_params(labelsize=5)
                except Exception:
                    ax2 = fig.add_subplot(gs[2, :])
                    ax2.axis('off')
                    ax2.set_title("DOP tile heatmap (error)", fontsize=9)
            else:
                ax2 = fig.add_subplot(gs[2, :])
                ax2.axis('off')
                if full_dop_img is not None:
                    ax2.imshow(full_dop_img)
                    if not tile2_ran and fd_had_region:
                        try:
                            import matplotlib.patches as mpatches
                            _b = self.geo_handler.dop_bounds
                            if hasattr(_b, '__len__'):
                                _bmin_x = float(_b[0]); _bmin_y = float(_b[1])
                                _bmax_x = float(_b[2]); _bmax_y = float(_b[3])
                            else:
                                _bmin_x = float(_b.left);  _bmin_y = float(_b.bottom)
                                _bmax_x = float(_b.right); _bmax_y = float(_b.top)
                            _dw = _bmax_x - _bmin_x; _dh = _bmax_y - _bmin_y
                            _ih, _iw = full_dop_img.shape[:2]
                            _ts = float(np.clip(min(_dw, _dh) * 0.15, 100.0, 400.0))
                            _stride = _ts * 0.5; _half = _ts / 2.0
                            _cxs = np.arange(_bmin_x + _half, _bmax_x - _half + _stride, _stride)
                            _cys = np.arange(_bmin_y + _half, _bmax_y - _half + _stride, _stride)
                            for _cx in _cxs:
                                for _cy in _cys:
                                    _px = (_cx - _ts / 2 - _bmin_x) / _dw * _iw
                                    _py = (_bmax_y - (_cy + _ts / 2)) / _dh * _ih
                                    ax2.add_patch(mpatches.Rectangle(
                                        (_px, _py), _ts / _dw * _iw, _ts / _dh * _ih,
                                        linewidth=0.8, edgecolor='gray',
                                        facecolor='none', alpha=0.6))
                        except Exception:
                            pass
                reason = ("not run" if not tile2_ran
                          else "bypassed (fd had region)" if fd_had_region else "search bypassed")
                ax2.set_title(f"DOP tile heatmap [{reason}]", fontsize=9)

            # ─── Suptitle ─────────────────────────────────────────────────
            err_title = ""
            if self._gt_position is not None and rough_position is not None:
                xy_e = float(np.linalg.norm(rough_position[:2] - self._gt_position[:2]))
                dz_e = float(abs(rough_position[2] - self._gt_position[2]))
                t_e  = float(np.linalg.norm(rough_position - self._gt_position))
                err_title = f"  \u0394xy={xy_e:.1f}m  \u0394z={dz_e:.1f}m  |t|={t_e:.1f}m"
                if rough_R_c2w is not None and self._gt_R_c2w is not None:
                    R_diff = rough_R_c2w @ self._gt_R_c2w.T
                    _trace = np.clip(float(np.trace(R_diff)), -1.0, 3.0)
                    rot_e  = float(np.rad2deg(np.arccos((_trace - 1) / 2)))
                    err_title += f"  rot={rot_e:.1f}\u00b0"
            if calib_fov is not None:
                self._coarse_fov = calib_fov
                if self._gt_fov is not None:
                    err_title += f"  \u0394FoV={calib_fov - self._gt_fov:+.1f}\u00b0"

            n_fd_raw = n_all
            fig.suptitle(
                f"Coarse stage  frame={frame_id}  "
                f"[fd={lbl1}, tile={lbl2}]  "
                f"raw={n_fd_raw}  inliers={roi_inl}{err_title}",
                fontsize=10, y=0.995,
            )
            self._save_named(frame_id, "coarse", fig)
        except Exception:
            pass

    def fine_stage(
        self,
        frame_id: int,
        image: np.ndarray,
        # crop specs & tiles (from DSM footprint computation)
        crop_specs: Optional[List] = None,
        crop_tiles: Optional[List] = None,
        match_results: Optional[List] = None,
        # final PnP result
        pts_2d: Optional[np.ndarray] = None,
        pts_3d: Optional[np.ndarray] = None,
        confs: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        R_c2w: Optional[np.ndarray] = None,
        num_inliers: int = 0,
        # recalibration note
        recalib_fov: Optional[float] = None,
        recalib_was_run: bool = False,
        fine_matcher_name: Optional[str] = None,
        # per-crop 3D correspondences for per-crop reproj stats
        per_crop_pts2d: Optional[List] = None,
        per_crop_pts3d: Optional[List] = None,
        reproj_threshold: float = 12.0,
    ) -> None:
        """Fine stage: 2-row figure saved to ``fine/``.

        Row 0 — query image | DOP crop tiles (correspondences + stats)
        Row 1 — query + LoD overlay | Fine PnP DOP | conf histogram"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib import gridspec as _gs

            crop_specs    = crop_specs    or []
            crop_tiles    = crop_tiles    or []
            match_results = match_results or []
            per_crop_pts2d = per_crop_pts2d or [None] * len(crop_specs)
            per_crop_pts3d = per_crop_pts3d or [None] * len(crop_specs)
            n_crops  = len(crop_specs)
            has_pos  = position is not None
            has_gt   = self._gt_position is not None
            has_pts  = pts_2d is not None and len(pts_2d) > 0
            image_size = image.shape[:2]  # (h, w)

            # n_cols must accommodate row 0 (query + crops) and row 1 (≥ 3 panels)
            n_cols = max(n_crops + 1, 3)

            # Pre-compute FoV info for use throughout this function
            _est_fov = (recalib_fov if recalib_was_run and recalib_fov is not None
                        else self._coarse_fov)
            _fov_delta_str = (f"\u0394FoV={_est_fov - self._gt_fov:+.1f}\u00b0"
                              if _est_fov is not None and self._gt_fov is not None
                              else "")

            fig = plt.figure(figsize=(4 * n_cols, 9))
            gs  = _gs.GridSpec(2, n_cols, figure=fig, hspace=0.30, wspace=0.05)

            # ── Row 0: query image ────────────────────────────────────────────
            ax_q = fig.add_subplot(gs[0, 0])
            ax_q.axis('off')
            ax_q.imshow(image)
            ax_q.set_title("Query image", fontsize=9)

            # ── Row 0: DOP crop tiles + correspondences ───────────────────────
            for i, (mr, tile, (cx, cy, sz)) in enumerate(
                    zip(match_results, crop_tiles, crop_specs), start=1):
                ax = fig.add_subplot(gs[0, i])
                ax.axis('off')
                if tile is not None:
                    ax.imshow(tile.data)
                    n_match = 0
                    try:
                        if mr is not None and hasattr(mr, 'kpts_dop') and len(mr.kpts_dop):
                            n_match = len(mr.kpts_dop)
                            c_vals = (mr.confidences
                                      if hasattr(mr, 'confidences') and
                                      mr.confidences is not None and
                                      len(mr.confidences) == n_match
                                      else np.ones(n_match))
                            ax.scatter(mr.kpts_dop[:, 0], mr.kpts_dop[:, 1],
                                       s=3, c=c_vals, cmap='plasma',
                                       vmin=0.0, vmax=1.0, linewidths=0, alpha=0.7)
                            # Three-line stats: raw | reproj inliers | all-reproj + Δpose
                            _ca = np.asarray(c_vals)
                            _stats1 = (f"raw:  n={n_match}  med={np.median(_ca):.2f}"
                                       f"  mean={_ca.mean():.2f}"
                                       f"  min={_ca.min():.2f}  max={_ca.max():.2f}")
                            _crop_2d = (per_crop_pts2d[i - 1]
                                        if per_crop_pts2d is not None and i - 1 < len(per_crop_pts2d)
                                        else None)
                            _crop_3d = (per_crop_pts3d[i - 1]
                                        if per_crop_pts3d is not None and i - 1 < len(per_crop_pts3d)
                                        else None)
                            _re = (self._reproj_errors(_crop_3d, _crop_2d, R_c2w, position)
                                   if _crop_3d is not None and _crop_2d is not None and has_pos
                                   else None)
                            if _re is not None:
                                _re_fin = _re[np.isfinite(_re)]
                                _inl_m  = np.isfinite(_re) & (_re < reproj_threshold)
                                _n_inl  = int(_inl_m.sum())
                                _inl_re = _re[_inl_m]
                                if _n_inl > 0:
                                    # Line 2: confidence stats for reproj-inlier points
                                    _inl_cf = _ca[_inl_m]
                                    _stats2 = (f"inl(prj<{reproj_threshold:.0f}px):  n={_n_inl}"
                                               f"  med={np.median(_inl_cf):.2f}"
                                               f"  mean={_inl_cf.mean():.2f}"
                                               f"  min={_inl_cf.min():.2f}  max={_inl_cf.max():.2f}")
                                else:
                                    _stats2 = f"inl(prj<{reproj_threshold:.0f}px):  n=0"
                                _s3 = []
                                if len(_re_fin) > 0:
                                    _s3.append(f"all:  med={np.median(_re_fin):.1f}px"
                                               f"  max={_re_fin.max():.1f}px")
                                _dp = self._pose_delta_str(position, R_c2w)
                                if _dp:
                                    _s3.append(_dp)
                                if _fov_delta_str:
                                    _s3.append(_fov_delta_str)
                                _stats3 = "  ".join(_s3)
                            else:
                                _hi = _ca >= 0.5
                                _n_hi = int(_hi.sum())
                                if _n_hi > 0:
                                    _hi_cf = _ca[_hi]
                                    _stats2 = (f"hi-conf(≥0.5):  n={_n_hi}"
                                               f"  med={np.median(_hi_cf):.2f}"
                                               f"  mean={_hi_cf.mean():.2f}"
                                               f"  max={_hi_cf.max():.2f}")
                                else:
                                    _stats2 = "hi-conf(≥0.5):  n=0"
                                _stats3 = _fov_delta_str
                            _overlay = (f"{_stats1}\n{_stats2}"
                                        + (f"\n{_stats3}" if _stats3 else ""))
                            ax.text(0.01, 0.01, _overlay,
                                    transform=ax.transAxes, fontsize=7,
                                    color='white', va='bottom',
                                    bbox=dict(boxstyle='round,pad=0.2',
                                              facecolor='black', alpha=0.5))
                    except Exception:
                        pass
                    _fm = f" [{fine_matcher_name}]" if fine_matcher_name else ""
                    ax.set_title(f"Crop {i} ({sz:.0f}m){_fm}  n={n_match}", fontsize=9)

            # ── Row 1 col 0: query + LoD overlay (or inlier scatter) ─────────
            # LoD panel ~half width, PnP DOP ~2/3 of remainder, hist the rest
            _c1 = max(n_cols // 2, 1)
            _c2 = _c1 + max(int(round((n_cols - _c1) * 2 / 3)), 1)
            _c2 = min(_c2, n_cols - 1)  # hist needs at least 1 col

            ax_lod = fig.add_subplot(gs[1, 0:_c1])
            ax_lod.axis('off')
            ax_lod.imshow(image)
            if has_pts:
                c_v = (confs if confs is not None and len(confs) == len(pts_2d)
                       else np.ones(len(pts_2d)))
                ax_lod.scatter(pts_2d[:, 0], pts_2d[:, 1],
                               s=3, c=c_v, cmap='plasma',
                               vmin=0.0, vmax=1.0, linewidths=0, alpha=0.6)
            has_lod = (self.lod is not None and has_pos and R_c2w is not None)
            if has_lod:
                self._lod_segments(ax_lod, R_c2w, position, image_size)
                ax_lod.set_title(f"UAV + LoD  [{num_inliers} inl]", fontsize=9)
            else:
                ax_lod.set_title(f"UAV correspondences  [{num_inliers} inl]", fontsize=9)

            # ── Row 1 middle span: Fine PnP DOP overview ─────────────────────
            def _dop_pixel(dop, ux, uy):
                bw = dop.bounds[2] - dop.bounds[0]
                bh = dop.bounds[3] - dop.bounds[1]
                return ((ux - dop.bounds[0]) / bw * dop.data.shape[1],
                        (dop.bounds[3] - uy)  / bh * dop.data.shape[0])

            ax_dop = fig.add_subplot(gs[1, _c1:_c2])
            ax_dop.axis('off')
            if has_pos:
                try:
                    c_vals = (confs if has_pts and confs is not None
                              and len(confs) == len(pts_2d) else None)
                    if has_pts and pts_3d is not None and len(pts_3d) > 0:
                        all_x = list(pts_3d[:, 0]) + [float(position[0])]
                        all_y = list(pts_3d[:, 1]) + [float(position[1])]
                        if has_gt:
                            all_x.append(float(self._gt_position[0]))
                            all_y.append(float(self._gt_position[1]))
                        extent   = max(max(all_x) - min(all_x), max(all_y) - min(all_y))
                        dop_cx   = (max(all_x) + min(all_x)) / 2
                        dop_cy   = (max(all_y) + min(all_y)) / 2
                        dop_size = max(extent * 1.3, 80.0)
                    else:
                        dop_cx, dop_cy, dop_size = float(position[0]), float(position[1]), 200.0
                    dop = self.geo_handler.crop_dop(dop_cx, dop_cy, dop_size)
                    if dop is not None:
                        ax_dop.imshow(dop.data)
                        if has_pts and pts_3d is not None:
                            px, py = _dop_pixel(dop, pts_3d[:, 0], pts_3d[:, 1])
                            ax_dop.scatter(px, py, s=4,
                                           c=(c_vals if c_vals is not None
                                              else np.ones(len(pts_3d))),
                                           cmap='plasma', vmin=0., vmax=1.,
                                           linewidths=0, alpha=0.5)
                        cam_px, cam_py = _dop_pixel(dop, float(position[0]), float(position[1]))
                        ax_dop.plot(cam_px, cam_py, 'rx', markersize=12,
                                    markeredgewidth=2, label='pred cam')
                        if has_gt:
                            gt_px, gt_py = _dop_pixel(dop,
                                                       float(self._gt_position[0]),
                                                       float(self._gt_position[1]))
                            ax_dop.plot(gt_px, gt_py, 'gx', markersize=12,
                                        markeredgewidth=2, label='GT cam')
                        # LoD overlay on DOP: GT pose in green, pred pose in red
                        has_lod_dop = (self.lod is not None and R_c2w is not None)
                        if has_lod_dop:
                            if has_gt and self._gt_R_c2w is not None:
                                self._lod_on_dop(ax_dop, dop, self._gt_R_c2w,
                                                 self._gt_position, image_size,
                                                 color='lime', alpha=0.5)
                            self._lod_on_dop(ax_dop, dop, R_c2w, position,
                                             image_size, color='red', alpha=0.5)
                        ax_dop.legend(fontsize=7, loc='upper right')
                        # Build detailed title
                        _pnp_parts = [f"Fine PnP  [{num_inliers} inl]  z={position[2]:.0f}m"]
                        if has_gt:
                            _dxy = np.linalg.norm(position[:2] - self._gt_position[:2])
                            _dz  = abs(position[2] - self._gt_position[2])
                            _dt  = np.linalg.norm(position - self._gt_position)
                            _pnp_parts.append(f"\u0394xy={_dxy:.1f}m")
                            _pnp_parts.append(f"\u0394z={_dz:.1f}m")
                            _pnp_parts.append(f"|t|={_dt:.1f}m")
                            if R_c2w is not None and self._gt_R_c2w is not None:
                                _Rd = R_c2w @ self._gt_R_c2w.T
                                _tr = np.clip(float(np.trace(_Rd)), -1.0, 3.0)
                                _rot = float(np.rad2deg(np.arccos((_tr - 1) / 2)))
                                _pnp_parts.append(f"rot={_rot:.1f}\u00b0")
                        if recalib_was_run and recalib_fov is not None:
                            if self._gt_fov is not None:
                                _pnp_parts.append(f"\u0394FoV={recalib_fov - self._gt_fov:+.1f}\u00b0")
                            else:
                                _pnp_parts.append(f"recalib={recalib_fov:.1f}\u00b0")
                        ax_dop.set_title("  ".join(_pnp_parts), fontsize=9)
                except Exception:
                    ax_dop.set_title("DOP (unavailable)", fontsize=9)

            # ── Row 1 right span: confidence histogram ────────────────────────
            ax_hist = fig.add_subplot(gs[1, _c2:])
            if has_pts:
                ax_hist.axis('on')
                c_vals = (confs if confs is not None and len(confs) == len(pts_2d)
                          else np.ones(len(pts_2d)))
                ax_hist.hist(c_vals, bins=30, color='steelblue', edgecolor='none')
                ax_hist.axvline(float(np.median(c_vals)), color='tomato', linewidth=1.5,
                                label=f"med={np.median(c_vals):.2f}")
                ax_hist.set_xlabel("confidence", fontsize=8)
                ax_hist.set_ylabel("count", fontsize=8)
                ax_hist.set_title(f"Conf dist  n={len(c_vals)}", fontsize=9)
                ax_hist.legend(fontsize=7)
            else:
                ax_hist.axis('off')

            # ── Suptitle with pose errors ─────────────────────────────────────
            err_title = ""
            if has_gt and position is not None:
                xy_e = float(np.linalg.norm(position[:2] - self._gt_position[:2]))
                dz_e = float(abs(position[2] - self._gt_position[2]))
                t_e  = float(np.linalg.norm(position - self._gt_position))
                err_title = f"  \u0394xy={xy_e:.1f}m  \u0394z={dz_e:.1f}m  |t|={t_e:.1f}m"
                if R_c2w is not None and self._gt_R_c2w is not None:
                    R_diff = R_c2w @ self._gt_R_c2w.T
                    _trace = np.clip(float(np.trace(R_diff)), -1.0, 3.0)
                    rot_e  = float(np.rad2deg(np.arccos((_trace - 1) / 2)))
                    err_title += f"  rot={rot_e:.1f}\u00b0"
            # Confidence colorbar (plasma cmap 0→1)
            import matplotlib.cm as _mcm
            import matplotlib.colors as _mcolors
            _sm = _mcm.ScalarMappable(
                norm=_mcolors.Normalize(vmin=0.0, vmax=1.0), cmap='plasma'
            )
            _sm.set_array([])
            try:
                fig.colorbar(_sm, ax=ax_hist, label='confidence',
                             shrink=0.8, pad=0.04)
            except Exception:
                pass
            est_fov = (recalib_fov if recalib_was_run and recalib_fov is not None
                       else self._coarse_fov)
            if est_fov is not None and self._gt_fov is not None:
                err_title += f"  \u0394FoV={est_fov - self._gt_fov:+.1f}\u00b0"

            fig.suptitle(
                f"Fine stage  frame={frame_id}  [{num_inliers} final inliers]{err_title}",
                fontsize=10, y=0.995,
            )
            self._save_named(frame_id, "fine", fig)
        except Exception:
            pass
