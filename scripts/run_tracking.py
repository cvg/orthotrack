#!/usr/bin/env python3
"""
Thin CLI entry-point for the OrthoTrack tracking pipeline.

All heavy logic lives in the ``orthotrack`` package — this file only does
argument parsing, config assembly, pipeline invocation and result I/O."""

import argparse
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

from orthotrack.pipeline import DEFAULT_MAX_IMAGE_DIM, TrackingPipeline


from utils.system import generate_demo_video, get_device_info


def main():
    parser = argparse.ArgumentParser(description="OrthoTrack Visual Tracking Pipeline")

    # --- Input sources (mutually-exclusive groups) ---
    parser.add_argument("--info", type=str, help="Path to info.json file")
    parser.add_argument("--sequence_dir", "--sequence-dir", type=str,
                        help="Path to pre-processed sequence directory")
    parser.add_argument("--gt_poses", "--gt-poses", type=str,
                        help="Path to ground truth poses (CSV or JSON)")
    parser.add_argument("--footage", type=str,
                        help="Path to video file or directory with frames")
    parser.add_argument("--dop", type=str, nargs='+',
                        help="Path(s) to DOP GeoTIFF file(s). Provide multiple files for multi-tile areas.")
    parser.add_argument("--dsm", type=str, nargs='+',
                        help="Path(s) to DSM GeoTIFF file(s). Provide multiple files for multi-tile areas.")
    parser.add_argument("--dop_year", "--dop-year", type=str, default="last",
                        help="DOP year to use: 'last' (most recent, default) or a specific year (e.g. 2020)")

    # --- Output / matcher ---
    parser.add_argument("--output", "-o", type=str, default="results_tracking",
                        help="Output directory")
    parser.add_argument("--fine_matcher", "--fine-matcher", type=str, default="precise",
                        help="Matcher for fine-stage crop matching: RoMa setting "
                             "(turbo/fast/base/precise) or IMCUI key "
                             "(e.g. superpoint+lightglue, loftr, Mast3R). Default: precise.")
    parser.add_argument("--coarse_matcher", "--coarse-matcher", type=str, default="base",
                        help="Matcher for coarse-stage search/re-localization. "
                             "Default: base.")
    parser.add_argument("--min_matching_agl", "--min-matching-agl", type=float, default=50.0,
                        help="Minimum AGL (m) to attempt matching. Frames below this are skipped. "
                             "Set to 0 to disable. Default: 50.")

    # --- Frame control ---
    parser.add_argument("--fps", type=float, default=24.0, help="Video FPS")
    parser.add_argument("--start_frame", "--start-frame", type=int, default=0)
    parser.add_argument("--end_frame", "--end-frame", type=int, default=None)
    parser.add_argument("--skip_frames", "--skip-frames", type=int, default=1)

    # --- Keyframe / tracking ---
    parser.add_argument("--keyframe_min_points", "--keyframe-min-points",
                        type=int, default=100)
    parser.add_argument("--keyframe_max_interval", "--keyframe-max-interval",
                        type=int, default=None,
                        help="Force keyframe every N frames (None = quality-only adaptive triggering)")
    parser.add_argument("--keyframe_reproj_threshold", "--keyframe-reproj-threshold",
                        type=float, default=2.0,
                        help="Reproj error growth factor for keyframe trigger (e.g. 2.0 = trigger when reproj grows 2x vs baseline)")
    parser.add_argument("--save_keyframe_vis", "--save-keyframe-vis",
                        action="store_true",
                        help="Save visualization for keyframes (also enables stage debug figures)")
    parser.add_argument("--save_tracking_vis", "--save-tracking-vis",
                        action="store_true",
                        help="Save visualization for every tracked frame (slow)")
    parser.add_argument("--save_vis", "--save-vis",
                        action="store_true",
                        help="Save all visualizations: keyframes, tracked frames, and stage debug figures. "
                             "Equivalent to --save_keyframe_vis --save_tracking_vis.")
    parser.add_argument("--vis_interval", "--vis-interval", type=int, default=1,
                        help="Save tracking visualizations every N frames (default 1 = every frame).")
    parser.add_argument("--flow_method", "--flow-method", type=str, default="lk",
                        help="Inter-frame tracking method: 'lk' (auto-selects GPU/CPU), 'waft', ptlflow model name "
                             "(e.g. 'raft', 'flowformer', 'gma', 'sea_raft', ...), "
                             "or foundation model name for 3D reconstruction tracking "
                             "(e.g. 'da3', 'da3_nested', 'vggt', 'pi3', 'pi3x', "
                             "'mapanything', 'dust3r', 'mast3r', 'pow3r')")
    parser.add_argument("--num_matches", "--num-matches", type=int, default=3000)
    parser.add_argument("--grace_ramp_frames", "--grace-ramp-frames", type=int, default=15,
                        help="Frames over which reproj threshold grace period decays (default: 15)")
    parser.add_argument("--min_kf_interval", "--min-kf-interval", type=int, default=5,
                        help="Minimum frames between consecutive keyframes — warmup cooldown (default: 5)")
    parser.add_argument("--growth_decay_frames", "--growth-decay-frames", type=int, default=100,
                        help="Frames over which reproj growth factor decays to min_growth_margin (default: 100)")
    parser.add_argument("--min_growth_margin", "--min-growth-margin", type=float, default=0.35,
                        help="Minimum relative reproj growth margin after full decay (default: 0.35 = 35%%)")

    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--fig_ext", "--fig-ext", type=str, default="png",
                        help="Figure file extension (e.g. png, pdf, svg)")
    parser.add_argument("--max_keyframes", "--max-keyframes", type=int, default=None,
                        help="Stop tracking after this many keyframes (default: run all)")

    parser.add_argument("--tracking_mode", "--tracking-mode", type=str, default="default",
                        choices=["default", "localize_every_frame", "dsm_tracking_only"],
                        help="Tracking mode: 'default' (OrthoTrack with keyframe + LK tracking), "
                             "'localize_every_frame' (RoMaV2 at every frame, no tracking), "
                             "'dsm_tracking_only' (single first-frame localization, then DSM-projected PnP)")
    parser.add_argument("--filter_by_trackability", "--filter-by-trackability",
                        action="store_true",
                        help="Filter RoMa-matched points by Shi-Tomasi trackability score before LK tracking")
    parser.add_argument("--max_tracking_points", "--max-tracking-points", type=int, default=0,
                        help="Cap keyframe tracking points to top-N by confidence (0 = no cap, suggested: 3000)")
    parser.add_argument("--single_crop_min_inliers_ratio", "--single-crop-min-inliers-ratio",
                        type=float, default=0.30,
                        help="Minimum fraction of num_matches required as PnP inliers to accept single-crop "
                             "(default: 0.30; PnP RANSAC typically yields ~30-40%% inlier rate)")
    parser.add_argument("--accumulate_points", "--accumulate-points",
                        action="store_true",
                        help="Merge surviving tracked points with new keyframe matches instead of discarding them")

    # --- Sensor prior (simulated GPS + IMU) ---
    parser.add_argument("--use_prior", "--use-prior",
                        action="store_true",
                        help="Use simulated GPS + IMU prior for crop positioning "
                             "(replaces brute-force ROI detection with noisy GPS position)")
    parser.add_argument("--prior_gps_sigma", "--prior-gps-sigma",
                        type=float, default=3.0,
                        help="GPS horizontal noise std in metres (default: 3.0)")
    parser.add_argument("--prior_gps_vertical_sigma", "--prior-gps-vertical-sigma",
                        type=float, default=5.0,
                        help="GPS vertical noise std in metres (default: 5.0)")
    parser.add_argument("--prior_imu_sigma", "--prior-imu-sigma",
                        type=float, default=1.0,
                        help="IMU roll/pitch noise std in degrees (default: 1.0)")
    parser.add_argument("--prior_imu_yaw_sigma", "--prior-imu-yaw-sigma",
                        type=float, default=4.0,
                        help="IMU yaw/heading noise std in degrees (default: 4.0)")
    parser.add_argument("--prior_seed", "--prior-seed",
                        type=int, default=42,
                        help="Random seed for sensor noise generation (default: 42)")

    # --- Image resizing ---
    parser.add_argument("--max_image_dim", "--max-image-dim",
                        type=int, default=DEFAULT_MAX_IMAGE_DIM,
                        help="Auto-downscale input images so the largest dimension "
                             "does not exceed this value. 0 = no resizing. "
                             f"Default: {DEFAULT_MAX_IMAGE_DIM} (see orthotrack.pipeline.DEFAULT_MAX_IMAGE_DIM). "
                             "Intrinsics in results.json are always reported at the "
                             "original (unscaled) resolution.")

    # --- LOD mesh for debug overlay ---
    parser.add_argument("--lod_obj_dir", "--lod-obj-dir", type=str, nargs='+', default=None,
                        help="Path(s) to OBJ/PLY or CityGML (.gml) files for LoD overlay. "
                             "Provide multiple files for multi-tile areas (optional).")

    # (--fine_matcher and --coarse_matcher defined near --min_matching_agl)

    # --- Force FoV self-calibration ---
    parser.add_argument("--force_calibration", "--force-calibration",
                        action="store_true",
                        help="Ignore any loaded intrinsics.json and force coarse-stage FoV sweep "
                             "self-calibration. Useful for validating calibration quality.")
    parser.add_argument("--intrinsics", type=str, default=None,
                        help="Explicit path to intrinsics.json (overrides auto-detection from footage directory).")

    # --- DSM degradation (rebuttal sensitivity sweep) ---
    parser.add_argument("--dsm_scale", "--dsm-scale", type=float, default=1.0,
                        help="Resolution factor in (0, 1]. Effective DSM GSD becomes "
                             "baseline_gsd / dsm_scale. 1.0 = no change. Default: 1.0.")
    parser.add_argument("--dsm_sigma_z", "--dsm-sigma-z", type=float, default=0.0,
                        help="Gaussian vertical noise std (m) added to DSM. Default: 0.0.")
    parser.add_argument("--dsm_noise_seed", "--dsm-noise-seed", type=int, default=0,
                        help="Seed for DSM vertical-noise RNG. Default: 0.")
    parser.add_argument("--dop_scale", "--dop-scale", type=float, default=1.0,
                        help="Resolution factor in (0, 1] for the DOP. Effective DOP GSD becomes "
                             "baseline_gsd / dop_scale. 1.0 = no change. Default: 1.0.")

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    #  Resolve matcher names                                              #
    # ------------------------------------------------------------------ #
    fine_matcher_str   = args.fine_matcher    # e.g. "precise", "turbo", "superpoint+lightglue"
    coarse_matcher_str = args.coarse_matcher  # e.g. "turbo"

    # ------------------------------------------------------------------ #
    #  Parse dop_year                                                     #
    # ------------------------------------------------------------------ #
    dop_year_val = args.dop_year
    if dop_year_val != 'last':
        try:
            dop_year_val = int(dop_year_val)
        except ValueError:
            dop_year_val = 'last'  # fallback

    # ------------------------------------------------------------------ #
    #  Assemble pipeline config                                           #
    # ------------------------------------------------------------------ #
    if args.info:
        info_path = Path(args.info)
        base_dir = info_path.parent
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        files = info["files"]
        config = {
            "dop_path": str(base_dir / files.get("dop", files.get("dop_tiles", [""])[0])),
            "dsm_path": str(base_dir / files.get("dsm", files.get("dsm_tiles", [""])[0])),
            "gt_json_path": str(base_dir / files["poses"]),
            "footage_dir": str(base_dir / files["video"]),
            "output_dir": args.output,
            "fine_matcher_name": fine_matcher_str,
            "coarse_matcher_setting": coarse_matcher_str,
            "fps": args.fps,
            "keyframe_min_points": args.keyframe_min_points,
            "keyframe_max_interval": args.keyframe_max_interval,
            "num_crop_candidates": 5,
            "save_keyframe_vis": args.save_keyframe_vis or args.save_vis,
            "save_tracking_vis": args.save_tracking_vis or args.save_vis,
            "vis_interval": args.vis_interval,
            "grace_ramp_frames": args.grace_ramp_frames,
            "min_kf_interval": args.min_kf_interval,
            "growth_decay_frames": args.growth_decay_frames,
            "min_growth_margin": args.min_growth_margin,
            "fig_ext": args.fig_ext,
            "max_keyframes": args.max_keyframes,
            "tracking_mode": args.tracking_mode,
            "filter_by_trackability": args.filter_by_trackability,
            "max_tracking_points": args.max_tracking_points,
            "single_crop_min_inliers_ratio": args.single_crop_min_inliers_ratio,
            "accumulate_points": args.accumulate_points,
            "use_prior": args.use_prior,
            "prior_gps_sigma": args.prior_gps_sigma,
            "prior_gps_vertical_sigma": args.prior_gps_vertical_sigma,
            "prior_imu_sigma": args.prior_imu_sigma,
            "prior_imu_yaw_sigma": args.prior_imu_yaw_sigma,
            "prior_seed": args.prior_seed,
        }

    elif args.sequence_dir:
        seq_dir = Path(args.sequence_dir)
        if not seq_dir.exists():
            parser.error(f"Sequence directory not found: {seq_dir}")

        footage_path = args.footage
        if not footage_path:
            if (seq_dir / "rendering").exists():
                footage_path = str(seq_dir / "rendering")
            elif (seq_dir / "video.mp4").exists():
                footage_path = str(seq_dir / "video.mp4")
            else:
                parser.error(f"No footage found in {seq_dir}. Specify --footage.")

        gt_path = args.gt_poses
        if not gt_path:
            if (seq_dir / "poses.csv").exists():
                gt_path = str(seq_dir / "poses.csv")
            else:
                parser.error(f"No poses.csv found in {seq_dir}. Specify --gt_poses.")

        config = {
            "sequence_dir": str(seq_dir),
            "gt_json_path": gt_path,
            "footage_dir": footage_path,
            "output_dir": args.output,
            "fine_matcher_name": fine_matcher_str,
            "coarse_matcher_setting": coarse_matcher_str,
            "fps": args.fps,
            "keyframe_min_points": args.keyframe_min_points,
            "keyframe_max_interval": args.keyframe_max_interval,
            "num_crop_candidates": 5,
            "save_keyframe_vis": args.save_keyframe_vis or args.save_vis,
            "save_tracking_vis": args.save_tracking_vis or args.save_vis,
            "vis_interval": args.vis_interval,
            "flow_method": args.flow_method,
            "num_matches": args.num_matches,
            "grace_ramp_frames": args.grace_ramp_frames,
            "min_kf_interval": args.min_kf_interval,
            "growth_decay_frames": args.growth_decay_frames,
            "min_growth_margin": args.min_growth_margin,
            "dop_year": dop_year_val,
            "fig_ext": args.fig_ext,
            "max_keyframes": args.max_keyframes,
            "tracking_mode": args.tracking_mode,
            "filter_by_trackability": args.filter_by_trackability,
            "max_tracking_points": args.max_tracking_points,
            "single_crop_min_inliers_ratio": args.single_crop_min_inliers_ratio,
            "accumulate_points": args.accumulate_points,
            "use_prior": args.use_prior,
            "prior_gps_sigma": args.prior_gps_sigma,
            "prior_gps_vertical_sigma": args.prior_gps_vertical_sigma,
            "prior_imu_sigma": args.prior_imu_sigma,
            "prior_imu_yaw_sigma": args.prior_imu_yaw_sigma,
            "prior_seed": args.prior_seed,
        }

    else:
        if not all([args.footage, args.dop, args.dsm]):
            parser.error(
                "Either --info, --sequence_dir, or all of "
                "(--footage, --dop, --dsm) are required. --gt_poses is optional."
            )
        config = {
            "dop_path": args.dop,
            "dsm_path": args.dsm,
            "gt_json_path": args.gt_poses,  # may be None
            "footage_dir": args.footage,
            "output_dir": args.output,
            "fine_matcher_name": fine_matcher_str,
            "coarse_matcher_setting": coarse_matcher_str,
            "fps": args.fps,
            "keyframe_min_points": args.keyframe_min_points,
            "keyframe_max_interval": args.keyframe_max_interval,
            "num_crop_candidates": 5,
            "save_keyframe_vis": args.save_keyframe_vis or args.save_vis,
            "save_tracking_vis": args.save_tracking_vis or args.save_vis,
            "vis_interval": args.vis_interval,
            "flow_method": args.flow_method,
            "num_matches": args.num_matches,
            "grace_ramp_frames": args.grace_ramp_frames,
            "min_kf_interval": args.min_kf_interval,
            "growth_decay_frames": args.growth_decay_frames,
            "min_growth_margin": args.min_growth_margin,
            "fig_ext": args.fig_ext,
            "max_keyframes": args.max_keyframes,
            "tracking_mode": args.tracking_mode,
            "filter_by_trackability": args.filter_by_trackability,
            "max_tracking_points": args.max_tracking_points,
            "single_crop_min_inliers_ratio": args.single_crop_min_inliers_ratio,
            "accumulate_points": args.accumulate_points,
            "use_prior": args.use_prior,
            "prior_gps_sigma": args.prior_gps_sigma,
            "prior_gps_vertical_sigma": args.prior_gps_vertical_sigma,
            "prior_imu_sigma": args.prior_imu_sigma,
            "prior_imu_yaw_sigma": args.prior_imu_yaw_sigma,
            "prior_seed": args.prior_seed,
        }

    # ------------------------------------------------------------------ #
    #  Create and run pipeline                                            #
    # ------------------------------------------------------------------ #
    config["max_image_dim"] = args.max_image_dim
    if args.lod_obj_dir is not None:
        config["lod_obj_dir"] = args.lod_obj_dir
    if args.force_calibration:
        config["force_calibration"] = True
    if args.intrinsics is not None:
        config["intrinsics_path"] = args.intrinsics
    config["dsm_scale"] = args.dsm_scale
    config["dsm_sigma_z"] = args.dsm_sigma_z
    config["dsm_noise_seed"] = args.dsm_noise_seed
    config["dop_scale"] = args.dop_scale
    pipeline = TrackingPipeline(**config)
    pipeline.keyframe_reproj_threshold = args.keyframe_reproj_threshold
    
    print("Output directory:", pipeline.output_dir)

    total_frames = (
        len(pipeline.gt_reader.poses)
        if pipeline.gt_reader is not None
        else pipeline.num_frames
    )
    start = args.start_frame
    end = args.end_frame if args.end_frame is not None else total_frames
    skip = args.skip_frames
    frame_indices = list(range(start, end, skip))

    results = pipeline.run_sequence(frame_indices, verbose=args.verbose)
    pipeline.print_summary(results)
    pipeline.plot_results(results, str(pipeline.output_dir / f"tracking_results.{args.fig_ext}"))

    # ------------------------------------------------------------------ #
    #  Save results                                                       #
    # ------------------------------------------------------------------ #
    device_info = get_device_info()
    intrinsics = pipeline.intrinsics
    # Intrinsics are stored at the internally-used (possibly downscaled) resolution.
    # Scale back to the original video/image resolution so that the saved values
    # can be applied directly to the unscaled footage.
    _img_scale = pipeline._image_scale  # 1.0 if no resize, <1.0 if downscaled
    _inv = 1.0 / _img_scale if _img_scale > 0 else 1.0
    intrinsics_info = {
        "fov_vertical": round(intrinsics.fov_vertical, 4) if intrinsics else None,
        "fx": round(intrinsics.fx * _inv, 4) if intrinsics and intrinsics.fx is not None else None,
        "fy": round(intrinsics.fy * _inv, 4) if intrinsics and intrinsics.fy is not None else None,
        "cx": round(intrinsics.cx * _inv, 4) if intrinsics and intrinsics.cx is not None else None,
        "cy": round(intrinsics.cy * _inv, 4) if intrinsics and intrinsics.cy is not None else None,
        "width": round(intrinsics.width * _inv) if intrinsics and intrinsics.width else None,
        "height": round(intrinsics.height * _inv) if intrinsics and intrinsics.height else None,
        "calibrated": pipeline._intrinsics_calibrated,
        "tracking_image_scale": round(_img_scale, 6),
    }
    results_data = {
        "total_frames": len(results),
        "keyframes": sum(1 for r in results if r.is_keyframe),
        "tracked": sum(1 for r in results if r.method == "tracked"),
        "predicted": sum(1 for r in results if r.method == "predicted"),
        "device": device_info,
        "intrinsics": intrinsics_info,
        "frames": [asdict(r) for r in results],
    }
    output_path = pipeline.output_dir / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    csv_path = pipeline.output_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "frame_id,is_keyframe,success,est_x,est_y,est_z,est_qw,est_qx,est_qy,est_qz,gt_x,gt_y,gt_z,gt_qw,gt_qx,gt_qy,gt_qz,"
            "position_error,horizontal_error,vertical_error,rotation_error,"
            "reproj_error,"
            "num_tracked_points,num_inliers,tracked_points_threshold,method,"
            "processing_time,kf_reason,baseline_reproj\n"
        )
        for r in results:
            f.write(
                f"{r.frame_id},{r.is_keyframe},{r.success},"
                f"{r.est_x},{r.est_y},{r.est_z},"
                f"{r.est_qw},{r.est_qx},{r.est_qy},{r.est_qz},"
                f"{r.gt_x},{r.gt_y},{r.gt_z},"
                f"{r.gt_qw},{r.gt_qx},{r.gt_qy},{r.gt_qz},"
                f"{r.position_error},{r.horizontal_error},{r.vertical_error},{r.rotation_error},"
                f"{r.reproj_error},"
                f"{r.num_tracked_points},{r.num_inliers},{r.tracked_points_threshold},"
                f"{r.method},{r.processing_time},{r.kf_reason},{r.baseline_reproj}\n"
            )
    print(f"CSV saved to {csv_path}")



if __name__ == "__main__":
    main()
