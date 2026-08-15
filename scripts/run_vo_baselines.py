#!/usr/bin/env python3
"""
Run Visual Odometry / SLAM baselines on MovingDrone sequences.

Evaluates with all alignment modes (same protocol as foundation model comparison):
- first_frame:       anchor at first GT frame, no scale (metric drift)
- first_frame_scale: anchor at first GT frame + LS scale (non-metric drift)
- ate_sim3:          global Sim(3) Umeyama (non-metric trajectory shape)"""

import argparse
import json
import subprocess
import sys
import numpy as np
from pathlib import Path

from orthotrack.baselines.vo_wrapper import (
    VOBaselineWrapper,
    FivePointVO,
    DROIDSLAMWrapper,
    DPVOWrapper,
)
from orthotrack.baselines.orbslam3_wrapper import ORBSLAM3Wrapper
from orthotrack.baselines.sfm_wrapper import COLMAPWrapper, GLOMAPWrapper


# ------------------------------------------------------------------ #
#  Registry of available methods                                      #
# ------------------------------------------------------------------ #

def build_method(name: str, device: str = "cuda") -> VOBaselineWrapper:
    """Instantiate a VO/SLAM method by name."""
    if name == "five_point_vo":
        vo = FivePointVO(max_features=4000, match_ratio=0.75, ransac_thresh=1.0,
                         target_width=960)
        return vo
    elif name == "five_point_vo_fast":
        vo = FivePointVO(max_features=2000, match_ratio=0.75, ransac_thresh=1.5,
                         target_width=640)
        vo.name = "five_point_vo_fast"
        return vo
    elif name == "droid_slam":
        return DROIDSLAMWrapper(device=device)
    elif name == "dpvo":
        return DPVOWrapper(device=device, use_loop_closure=False)
    elif name == "dpv_slam":
        return DPVOWrapper(device=device, use_loop_closure=True)
    elif name == "orb_slam3":
        return ORBSLAM3Wrapper(max_image_dim=1280, nFeatures=1200)
    elif name == "orb_slam3_fast":
        wrapper = ORBSLAM3Wrapper(max_image_dim=960, nFeatures=800)
        wrapper.name = "orb_slam3_fast"
        return wrapper
    elif name == "colmap":
        use_gpu = (device == "cuda")
        return COLMAPWrapper(use_gpu=use_gpu, max_num_features=8192,
                             matching_type="sequential")
    elif name == "glomap":
        use_gpu = (device == "cuda")
        return GLOMAPWrapper(use_gpu=use_gpu, max_num_features=8192,
                             matching_type="sequential")
    else:
        raise ValueError(f"Unknown method: {name}. Available: "
                         "five_point_vo, five_point_vo_fast, droid_slam, dpvo, dpv_slam, "
                         "orb_slam3, orb_slam3_fast, colmap, glomap")


# ------------------------------------------------------------------ #
#  Sequence runner                                                    #
# ------------------------------------------------------------------ #

def run_on_sequence(
    method: VOBaselineWrapper,
    sequence: str,
    data_root: str,
    output_root: str,
    stride: int = 1,
    max_frames: int = None,
) -> dict:
    """Run a VO method on one MovingDrone sequence and return multi-mode metrics."""
    seq_dir = Path(data_root) / "scenes" / sequence
    video_path = str(seq_dir / "video.mp4")
    gt_poses_path = str(seq_dir / "poses.csv")

    # Find intrinsics file (json or txt)
    intrinsics_path = seq_dir / "intrinsics.json"
    if not intrinsics_path.exists():
        intrinsics_path = seq_dir / "intrinsics.txt"
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"No intrinsics file found in {seq_dir}")
    intrinsics_path = str(intrinsics_path)

    # Determine frame indices from GT poses
    import csv
    frame_ids = []
    with open(gt_poses_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_ids.append(int(row['frame_id']))

    # Apply stride
    frame_indices = frame_ids[::stride]

    # Apply max_frames
    if max_frames is not None and max_frames > 0:
        frame_indices = frame_indices[:max_frames]

    print(f"\n{'=' * 60}")
    print(f"  {method.name} on {sequence}")
    print(f"  Frames: {len(frame_indices)} (stride={stride}, "
          f"range={frame_indices[0]}-{frame_indices[-1]})")
    print(f"{'=' * 60}")

    # Output directory
    output_dir = str(Path(output_root) / sequence / method.name)

    # Run and evaluate with all alignment modes
    result = method.run_sequence(
        video_path=video_path,
        intrinsics_path=intrinsics_path,
        gt_poses_path=gt_poses_path,
        frame_indices=frame_indices,
        output_dir=output_dir,
    )

    # Tag with sequence name
    result['sequence'] = sequence

    return result


# ------------------------------------------------------------------ #
#  Aggregation across sequences                                       #
# ------------------------------------------------------------------ #

ALIGNMENT_MODES = [
    ('first_frame', 'First-frame (no scale)'),
    ('first_frame_scale', 'First-frame + scale'),
    ('ate_sim3', 'Sim(3) alignment'),
]


def aggregate_results(all_results: list) -> dict:
    """Aggregate per-sequence multi-mode results into overall metrics.

    Following the project convention: per-sequence metrics averaged across sequences.

    Returns:
        dict[method_name] -> {
            'method': str,
            'n_sequences': int,
            'fps': float,
            'first_frame': {aggregated metrics},
            'first_frame_scale': {aggregated metrics},
            'ate_sim3': {aggregated metrics},
        }"""
    by_method = {}
    for r in all_results:
        method = r['method']
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(r)

    agg = {}
    for method, results in by_method.items():
        method_agg = {
            'method': method,
            'n_sequences': len(results),
            'fps': float(np.mean([r['fps'] for r in results])),
        }

        for mode_key, _ in ALIGNMENT_MODES:
            # Collect metrics for this mode across sequences
            mode_metrics = [r[mode_key] for r in results if mode_key in r]
            if not mode_metrics:
                continue

            # Aggregate: mean of per-sequence values
            metric_keys = [
                'pos_rmse', 'pos_median', 'pos_mean',
                'rot_median', 'rot_mean',
                'recall_0.5m', 'recall_1.0m', 'recall_2.0m', 'recall_5.0m', 'recall_10.0m',
                'recall_1.0deg', 'recall_2.0deg', 'recall_5.0deg', 'recall_10.0deg',
                'recall_1m_1deg', 'recall_2m_2deg', 'recall_5m_5deg', 'recall_10m_10deg',
            ]
            mode_agg = {'n_valid': len(mode_metrics)}
            for key in metric_keys:
                vals = [m[key] for m in mode_metrics if m.get(key) is not None]
                mode_agg[key] = float(np.mean(vals)) if vals else None

            method_agg[mode_key] = mode_agg

        agg[method] = method_agg

    return agg


def print_results_table(agg: dict, mode_key: str = 'first_frame_scale', mode_label: str = ''):
    """Print a formatted table for one alignment mode."""
    if not mode_label:
        mode_label = mode_key

    print(f"\n  [{mode_label}]")
    header = (f"{'Method':<20s} {'Scale':>6s} {'RMSE':>7s} {'TE':>7s} {'RE':>7s} "
              f"{'R@1m1d':>7s} {'R@2m2d':>7s} {'R@5m5d':>7s} {'FPS':>6s} {'Seqs':>5s}")
    print(header)
    print("-" * len(header))

    for method, data in agg.items():
        mode = data.get(mode_key, {})
        if not mode:
            print(f"{method:<20s}  {'NO DATA':>40s}")
            continue

        def fmt(v, d=2):
            return f"{v:.{d}f}" if v is not None else "-"

        print(f"{method:<20s} "
              f"{'n/a':>6s} "
              f"{fmt(mode.get('pos_rmse')):>7s} "
              f"{fmt(mode.get('pos_median')):>7s} "
              f"{fmt(mode.get('rot_median')):>7s} "
              f"{fmt(mode.get('recall_1m_1deg'), 1):>7s} "
              f"{fmt(mode.get('recall_2m_2deg'), 1):>7s} "
              f"{fmt(mode.get('recall_5m_5deg'), 1):>7s} "
              f"{fmt(data.get('fps'), 1):>6s} "
              f"{data.get('n_sequences', 0):>5d}")


def print_all_tables(agg: dict):
    """Print results tables for all alignment modes."""
    print("\n" + "=" * 90)
    print("  VO/SLAM Baseline Results (per-alignment-mode)")
    print("=" * 90)

    for mode_key, mode_label in ALIGNMENT_MODES:
        print_results_table(agg, mode_key, mode_label)

    print("=" * 90)
    print("\nNote: 'first_frame_scale' provides rotation-aware alignment (best for RE).")
    print("      'ate_sim3' provides global trajectory fit (often best for ATE/TE).")


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

# orb_slam3 is CPU-only, but uses substantial memory per sequence →
# run in subprocess anyway to isolate memory between sequences.
GPU_METHODS = {"droid_slam", "dpvo", "dpv_slam"}
SFM_METHODS = {"colmap", "glomap"}  # SfM: heavy disk I/O + subprocess isolation
SUBPROCESS_METHODS = GPU_METHODS | {"orb_slam3", "orb_slam3_fast"} | SFM_METHODS


def _run_single_subprocess(
    method_name: str,
    sequence: str,
    data_root: str,
    output_dir: str,
    stride: int,
    max_frames: int | None,
    device: str,
) -> dict | None:
    """Run a *single* method+sequence in a subprocess to isolate GPU memory.

    The subprocess runs this same script with ``--_subprocess_worker`` flag,
    which executes just that one method+sequence and writes summary.json.
    We then read the summary back."""
    cmd = [
        sys.executable, __file__,
        "--_subprocess_worker",
        "--sequences", sequence,
        "--methods", method_name,
        "--data_root", data_root,
        "--output_dir", output_dir,
        "--stride", str(stride),
        "--device", device,
    ]
    if max_frames is not None:
        cmd += ["--max_frames", str(max_frames)]

    print(f"\n>>> Launching subprocess: {method_name} on {sequence}")
    result = subprocess.run(cmd, env={**__import__('os').environ, "PYTHONPATH": "."},
                            capture_output=False, text=True)

    if result.returncode != 0:
        print(f"  !! Subprocess failed (exit={result.returncode})")
        return None

    # Read back the saved summary.json
    summary_path = Path(output_dir) / sequence / method_name / "summary.json"
    if not summary_path.exists():
        # Some wrappers save with wrapper.name which may differ (e.g. dpv_slam)
        summary_path = Path(output_dir) / sequence / method_name / "summary.json"
    if not summary_path.exists():
        print(f"  !! No summary.json produced at {summary_path}")
        return None

    with open(summary_path) as f:
        data = json.load(f)
    data["sequence"] = sequence
    return data


def main():
    parser = argparse.ArgumentParser(description="Run VO/SLAM baselines on MovingDrone")
    parser.add_argument("--sequences", nargs="+", required=True,
                        help="MovingDrone sequence names")
    parser.add_argument("--methods", nargs="+", default=["five_point_vo"],
                        help="Methods to run (five_point_vo, five_point_vo_fast, "
                             "droid_slam, dpvo, dpv_slam)")
    parser.add_argument("--data_root", default="data/MovingDrone",
                        help="Path to MovingDrone dataset root")
    parser.add_argument("--output_dir", default="outputs/vo_baselines",
                        help="Output directory")
    parser.add_argument("--stride", type=int, default=1,
                        help="Frame stride (1 = every frame)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Max frames per sequence (for quick testing)")
    parser.add_argument("--device", default="cuda",
                        help="Device for GPU methods")
    parser.add_argument("--table_mode", default=None,
                        choices=['first_frame', 'first_frame_scale', 'ate_sim3', 'all'],
                        help="Which alignment mode to show in summary table (default: all)")
    # Hidden flag for subprocess worker mode
    parser.add_argument("--_subprocess_worker", action="store_true",
                        help=argparse.SUPPRESS)
    # Flag to skip already-computed results
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip method+sequence pairs that already have summary.json")
    args = parser.parse_args()

    # Worker mode: run exactly one method on one sequence, then exit
    if args._subprocess_worker:
        assert len(args.methods) == 1 and len(args.sequences) == 1
        method = build_method(args.methods[0], device=args.device)
        run_on_sequence(
            method=method,
            sequence=args.sequences[0],
            data_root=args.data_root,
            output_root=args.output_dir,
            stride=args.stride,
            max_frames=args.max_frames,
        )
        return

    all_results = []

    for method_name in args.methods:
        for seq in args.sequences:
            # Check if we should skip
            if args.skip_existing:
                summary_path = Path(args.output_dir) / seq / method_name / "summary.json"
                if summary_path.exists():
                    print(f"\n--- Skipping {method_name}/{seq} (already exists)")
                    with open(summary_path) as f:
                        data = json.load(f)
                    data["sequence"] = seq
                    all_results.append(data)
                    continue

            # GPU/subprocess methods: isolate in subprocess to avoid OOM between sequences
            if method_name in SUBPROCESS_METHODS:
                result = _run_single_subprocess(
                    method_name=method_name,
                    sequence=seq,
                    data_root=args.data_root,
                    output_dir=args.output_dir,
                    stride=args.stride,
                    max_frames=args.max_frames,
                    device=args.device,
                )
                if result is not None:
                    all_results.append(result)
            else:
                # CPU methods run in-process (no GPU memory issues)
                try:
                    method = build_method(method_name, device=args.device)
                    result = run_on_sequence(
                        method=method,
                        sequence=seq,
                        data_root=args.data_root,
                        output_root=args.output_dir,
                        stride=args.stride,
                        max_frames=args.max_frames,
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"ERROR running {method_name} on {seq}: {e}")
                    import traceback
                    traceback.print_exc()

    # Aggregate and display
    if all_results:
        # Save all per-sequence results
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "all_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)

        agg = aggregate_results(all_results)

        if args.table_mode is None or args.table_mode == 'all':
            print_all_tables(agg)
        else:
            label = dict(ALIGNMENT_MODES).get(args.table_mode, args.table_mode)
            print_results_table(agg, args.table_mode, label)

        with open(out_dir / "aggregate_results.json", 'w') as f:
            json.dump(agg, f, indent=2, default=str)

        print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
