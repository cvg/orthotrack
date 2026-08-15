#!/usr/bin/env python3
"""
run_depth_benchmark.py
======================
Monocular depth estimator benchmark on MovingDrone test sequences.

For each (sequence, frame), the script:
    1. Extracts the video frame and GT ray-distance depth (from depth/*.npz).
    2. Converts GT to Z-depth (camera-space Z component).
    3. Runs the selected depth estimator.
    4. Computes depth metrics in two modes:
       - abs    : no scale alignment (metric estimators only)
       - scaled : median scale alignment (pred *= median(gt)/median(pred))
    5. Saves per-frame results to outputs/depth_benchmark/{sequence}/{estimator}/
       and per-sequence metrics to results.json.

Writes per-sequence metrics under the chosen --output_dir."""

import argparse
import json
import time
import warnings
import traceback
from pathlib import Path

import cv2
import numpy as np
from utils.image import list_video_frames, read_video_frame
from utils.depth import load_depth_npz

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT    = _PROJECT_ROOT / 'data' / 'MovingDrone'
_OUTPUT_ROOT  = _PROJECT_ROOT / 'outputs' / 'depth_benchmark'

# Stride: evaluate every STRIDE-th frame (matching the foundation benchmark stride=10)
STRIDE = 10

# Depth evaluation range for aerial UAV imagery (metres)
MIN_DEPTH = 10.0    # ignore near-ground pixels (< 10 m rarely valid in aerial)
MAX_DEPTH = 2000.0  # beyond 2 km depth maps are unreliable


def load_splits(splits_file=None):
    path = Path(splits_file or (_DATA_ROOT / 'splits.json'))
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def get_sequences(requested: list, split: str = 'test') -> list:
    """Return sequence names for the requested split or an explicit list."""
    splits = load_splits()
    if requested and requested[0] != 'all':
        return sorted(set(requested))
    # 'all' → union of test_inPlace + test_outPlace (same as foundation benchmark)
    test_seqs = sorted(
        set(splits.get('test_inPlace', [])) | set(splits.get('test_outPlace', []))
    )
    return test_seqs








def load_intrinsics(seq_dir: Path):
    """Load 3x3 intrinsics matrix and raw (fx, fy, cx, cy) from intrinsics.json."""
    intr_path = seq_dir / 'intrinsics.json'
    if not intr_path.exists():
        raise FileNotFoundError(f"intrinsics.json not found: {intr_path}")
    import json
    with open(intr_path) as f:
        d = json.load(f)
    fx  = float(d['fx'])
    fy  = float(d['fy'])
    cx  = float(d['cx'])
    cy  = float(d['cy'])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return K, fx, fy, cx, cy


def benchmark_sequence(
    estimator,
    seq_name: str,
    output_base: Path,
    stride: int = STRIDE,
    save_depth_maps: bool = False,
) -> dict:
    """
    Run depth benchmark for one sequence.

    Args:
        estimator:       Instantiated BaseDepthEstimator.
        seq_name:        MovingDrone sequence name.
        output_base:     Directory where results will be saved.
        stride:          Frame stride for subsampling.
        save_depth_maps: If True, save compressed predicted depth .npz files.

    Returns:
        Sequence-level results dict with metrics and timing."""
    from utils.depth import compute_depth_metrics, aggregate_depth_metrics, raydist_to_zdepth

    seq_dir   = _DATA_ROOT / 'scenes' / seq_name
    out_dir   = output_base / seq_name / estimator.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing results (skip if complete)
    results_path = out_dir / 'results.json'
    if results_path.exists():
        try:
            with open(results_path) as f:
                existing = json.load(f)
            if existing.get('status') == 'complete':
                print(f"  Skipping {seq_name}/{estimator.name} (already done)")
                return existing
        except Exception:
            pass

    print(f"\n  Sequence: {seq_name}")
    print(f"  Output:   {out_dir}")

    # Load intrinsics
    K, fx, fy, cx, cy = load_intrinsics(seq_dir)
    depth_files = sorted(seq_dir.glob('depth/depth_*.npz'))

    if not depth_files:
        print(f"  [SKIP] No depth files in {seq_dir}/depth/")
        return {'status': 'no_depth', 'sequence': seq_name, 'estimator': estimator.name}

    # Determine frame indices to evaluate
    frame_indices   = list_video_frames(seq_dir, stride=stride)
    n_depth_files   = len(depth_files)
    # Align: frame_idx -> depth file index (assume 1:1 correspondence)
    valid_frame_idx = [fi for fi in frame_indices if fi < n_depth_files]

    if not valid_frame_idx:
        print(f"  [SKIP] No valid frames for {seq_name}")
        return {'status': 'no_frames', 'sequence': seq_name, 'estimator': estimator.name}

    print(f"  Evaluating {len(valid_frame_idx)} frames (stride={stride}) ...")

    per_frame_abs    = []
    per_frame_scaled = []
    total_time       = 0.0
    errors           = []

    for frame_idx in valid_frame_idx:
        try:
            # Load frame image
            image_rgb = read_video_frame(seq_dir, frame_idx)

            # Load GT ray-distance depth + convert to Z-depth
            t_hit = load_depth_npz(depth_files[frame_idx])
            gt_depth_z = raydist_to_zdepth(t_hit, fx, fy, cx, cy)

            # Run estimator
            t0 = time.time()
            pred_depth = estimator.estimate(image_rgb, intrinsics=K)
            elapsed = time.time() - t0
            total_time += elapsed

            # Resize pred to GT size if needed
            if pred_depth.shape != gt_depth_z.shape:
                pred_depth = cv2.resize(
                    pred_depth, (gt_depth_z.shape[1], gt_depth_z.shape[0]),
                    interpolation=cv2.INTER_LINEAR)

            # Clamp negative predictions
            pred_depth = np.maximum(pred_depth, 0.0)

            # Compute metrics
            m_abs    = compute_depth_metrics(gt_depth_z, pred_depth,
                                              scale_correct=False,
                                              min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)
            m_scaled = compute_depth_metrics(gt_depth_z, pred_depth,
                                              scale_correct=True,
                                              min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)

            m_abs['frame_idx'] = frame_idx
            m_scaled['frame_idx'] = frame_idx
            per_frame_abs.append(m_abs)
            per_frame_scaled.append(m_scaled)

            # Optionally save predicted depth
            if save_depth_maps:
                np.savez_compressed(
                    out_dir / f'depth_{frame_idx:04d}.npz',
                    depth=pred_depth.astype(np.float32)
                )

        except Exception as e:
            err_msg = f"frame {frame_idx}: {e}"
            print(f"    [ERROR] {err_msg}")
            errors.append(err_msg)
            traceback.print_exc()

    if not per_frame_abs:
        result = {
            'status': 'failed', 'sequence': seq_name, 'estimator': estimator.name,
            'errors': errors,
        }
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=2)
        return result

    # Aggregate per-sequence metrics
    seq_metrics_abs    = aggregate_depth_metrics(per_frame_abs)
    seq_metrics_scaled = aggregate_depth_metrics(per_frame_scaled)

    n_frames_ok = len(per_frame_abs)
    fps = n_frames_ok / total_time if total_time > 0 else 0.0

    result = {
        'status': 'complete',
        'sequence': seq_name,
        'estimator': estimator.name,
        'is_metric': estimator.is_metric,
        'n_frames': n_frames_ok,
        'n_errors': len(errors),
        'inference_time_total': total_time,
        'fps': fps,
        'abs': seq_metrics_abs,
        'scaled': seq_metrics_scaled,
        'per_frame_abs': per_frame_abs,
        'per_frame_scaled': per_frame_scaled,
    }

    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  [ABS]    abs_rel={seq_metrics_abs['abs_rel']:.4f}  "
          f"rmse={seq_metrics_abs['rmse']:.2f}m  delta_1={seq_metrics_abs['delta_1']:.4f}")
    print(f"  [SCALED] abs_rel={seq_metrics_scaled['abs_rel']:.4f}  "
          f"rmse={seq_metrics_scaled['rmse']:.2f}m  delta_1={seq_metrics_scaled['delta_1']:.4f}")
    print(f"  FPS={fps:.2f}  ({n_frames_ok} frames in {total_time:.1f}s)")

    return result


def main():
    parser = argparse.ArgumentParser(description='Monocular depth estimator benchmark.')
    parser.add_argument('--estimator', type=str, default='moge2',
                        help='Depth estimator name (from DEPTH_ESTIMATOR_REGISTRY).')
    parser.add_argument('--sequences', nargs='+', default=['all'],
                        help='Sequence names or "all" for full test set.')
    parser.add_argument('--stride', type=int, default=STRIDE,
                        help=f'Frame subsampling stride (default={STRIDE}).')
    parser.add_argument('--output_dir', type=str, default=str(_OUTPUT_ROOT),
                        help='Output directory for results.')
    parser.add_argument('--save_depth_maps', action='store_true',
                        help='Save predicted depth maps as .npz files.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='PyTorch device.')
    parser.add_argument('--list', action='store_true',
                        help='List available estimators and exit.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Recompute even if results.json already exists.')
    args = parser.parse_args()

    if args.list:
        from orthotrack.depth_estimators import DEPTH_ESTIMATOR_REGISTRY
        print("\nAvailable depth estimators:")
        for k, v in DEPTH_ESTIMATOR_REGISTRY.items():
            metric_str = "metric" if v['metric'] else "relative"
            print(f"  {k:25s}  [{metric_str:8s}]  {v['paper']}")
        return

    output_dir = Path(args.output_dir)

    # Resolve sequences
    sequences = get_sequences(args.sequences)
    if not sequences:
        print("[ERROR] No sequences found. Check --sequences or data/MovingDrone/splits.json")
        return
    print(f"Sequences ({len(sequences)}): {sequences}")

    # Load estimator
    print(f"\nLoading estimator: {args.estimator} ...")
    from orthotrack.depth_estimators import get_depth_estimator, DEPTH_ESTIMATOR_REGISTRY
    try:
        estimator = get_depth_estimator(args.estimator, device=args.device)
    except Exception as e:
        print(f"[FATAL] Could not load estimator '{args.estimator}': {e}")
        traceback.print_exc()
        return
    print(f"  Loaded: {estimator}")

    # If --overwrite: remove existing results
    if args.overwrite:
        for seq in sequences:
            rp = output_dir / seq / args.estimator / 'results.json'
            if rp.exists():
                rp.unlink()
                print(f"  Removed {rp}")

    # Run benchmark
    all_results = []
    for seq in sequences:
        result = benchmark_sequence(
            estimator=estimator,
            seq_name=seq,
            output_base=output_dir,
            stride=args.stride,
            save_depth_maps=args.save_depth_maps,
        )
        all_results.append(result)

    # Compute overall cross-sequence averages (mean of per-sequence metrics)
    complete = [r for r in all_results if r.get('status') == 'complete']
    if complete:
        print(f"\n{'='*70}")
        print(f"Cross-sequence summary ({len(complete)}/{len(all_results)} sequences):")
        print(f"{'='*70}")
        for variant in ('abs', 'scaled'):
            vals = {k: [] for k in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log',
                                     'delta_1', 'delta_2', 'delta_3']}
            for r in complete:
                m = r.get(variant, {})
                for k in vals:
                    v = m.get(k)
                    if v is not None and not np.isnan(v):
                        vals[k].append(v)
            means = {k: float(np.mean(v)) if v else float('nan') for k, v in vals.items()}
            fps_list = [r.get('fps', 0) for r in complete if r.get('fps', 0) > 0]
            fps_mean = float(np.mean(fps_list)) if fps_list else 0.0
            print(f"\n  [{variant.upper():6s}] abs_rel={means['abs_rel']:.4f}  "
                  f"sq_rel={means['sq_rel']:.4f}  rmse={means['rmse']:.2f}  "
                  f"rmse_log={means['rmse_log']:.4f}  "
                  f"delta_1={means['delta_1']:.4f}  delta_2={means['delta_2']:.4f}  "
                  f"delta_3={means['delta_3']:.4f}  FPS={fps_mean:.2f}")

    # Save cross-sequence summary
    summary = {
        'estimator': args.estimator,
        'sequences': sequences,
        'n_complete': len(complete),
        'n_total': len(all_results),
        'stride': args.stride,
        'per_sequence': all_results,
    }
    summary_path = output_dir / f'summary_{args.estimator}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()
