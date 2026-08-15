#!/usr/bin/env python3
"""
Resample AeroPose trajectory JSON files to achieve realistic drone speeds.

The spatial path is preserved; only the temporal sampling changes — more frames
are interpolated along the same path so the drone "flies slower".

Target speed distribution (based on real consumer/racing drones):
  - 40% consumer slow:   15–30 km/h  (DJI Mini filming)
  - 30% consumer cruise: 30–50 km/h  (DJI Mavic/Air cruise)
  - 20% prosumer:        50–72 km/h  (DJI Mavic 3 max)
  - 10% racing/sport:    72–100 km/h (FPV sport mode)

Usage:
    # Test on a single sequence
    python scripts/resample_trajectories.py --sequences airport2

    # Process all sequences (with backup)
    python scripts/resample_trajectories.py --all

    # Custom output dir for testing
    python scripts/resample_trajectories.py --sequences airport2 --output-dir data/AeroPose/trajectories_resampled
"""

import json
import os
import sys
import shutil
import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

# ─── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAJ_DIR = PROJECT_ROOT / "data" / "AeroPose" / "trajectories"
TRAJ_BACKUP_DIR = PROJECT_ROOT / "data" / "AeroPose" / "trajectories_original"

# ─── Target Speed Distribution ─────────────────────────────────────────────────
SEED = 42
SPEED_BINS = [
    # (fraction, min_kmh, max_kmh)
    (0.40, 15, 30),    # consumer slow
    (0.30, 30, 50),    # consumer cruise
    (0.20, 50, 72),    # prosumer
    (0.10, 72, 100),   # racing/sport
]


def compute_ecef_speeds(frames: list[dict], fps: int) -> np.ndarray:
    """Compute per-frame speed (m/s) from ECEF positions."""
    positions = np.array([[f["position"]["x"], f["position"]["y"], f["position"]["z"]]
                          for f in frames])
    diffs = np.diff(positions, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    speeds = dists * fps
    return speeds


def assign_target_speeds(n_sequences: int, current_max_speeds_kmh: np.ndarray
                         ) -> np.ndarray:
    """Assign target max speeds from the realistic distribution.

    Sequences are ranked by current speed; slower ones get slower targets,
    faster ones get faster targets (preserving relative ordering).
    """
    rng = np.random.RandomState(SEED)
    sorted_indices = np.argsort(current_max_speeds_kmh)
    target_speeds = np.zeros(n_sequences)

    cum_frac = 0.0
    for frac, lo, hi in SPEED_BINS:
        start_rank = int(cum_frac * n_sequences)
        end_rank = int((cum_frac + frac) * n_sequences)
        if end_rank <= start_rank:
            end_rank = start_rank + 1
        for rank in range(start_rank, min(end_rank, n_sequences)):
            idx = sorted_indices[rank]
            target_speeds[idx] = rng.uniform(lo, hi)
        cum_frac += frac

    # Handle any remaining sequences (rounding)
    for idx in range(n_sequences):
        if target_speeds[idx] == 0:
            target_speeds[idx] = rng.uniform(15, 50)

    return target_speeds


def unwrap_angles(angles: np.ndarray) -> np.ndarray:
    """Unwrap angle sequence to avoid discontinuities at ±180°.

    Uses numpy's unwrap which handles the 2π discontinuity.
    """
    return np.rad2deg(np.unwrap(np.deg2rad(angles)))


def generate_correlated_noise(n_frames: int, fps: int, amplitude: float,
                              cutoff_hz: float, rng: np.random.RandomState
                              ) -> np.ndarray:
    """Generate smooth, correlated noise via filtered random walk.

    Produces band-limited noise that looks like realistic drone vibration/drift
    rather than white noise jitter.

    Args:
        n_frames: Number of frames
        amplitude: Noise amplitude (standard deviation of output)
        cutoff_hz: Low-pass filter cutoff frequency
        rng: Random state for reproducibility

    Returns:
        Smoothly varying noise signal of shape (n_frames,)
    """
    # Guard: need at least 13 samples for the Butterworth filter (order 5, filtfilt)
    if n_frames < 13:
        return np.zeros(n_frames)

    # Raw random walk
    raw = rng.randn(n_frames)

    # Low-pass Butterworth filter for smooth correlation
    nyquist = fps / 2.0
    normalized_cutoff = min(cutoff_hz / nyquist, 0.95)  # clamp to valid range
    b, a = butter(5, normalized_cutoff, btype='low')
    smoothed = filtfilt(b, a, raw)

    # Normalize to desired amplitude
    if smoothed.std() > 1e-10:
        smoothed = smoothed / smoothed.std() * amplitude

    return smoothed


def add_trajectory_noise(frames: list[dict], fps: int, noise_level: float,
                         rng: np.random.RandomState) -> list[dict]:
    """Add realistic correlated noise to trajectory frames.

    Simulates drone vibrations, wind gusts, GPS jitter, and gimbal wobble.

    Noise components (at noise_level=0.5):
      - Position XY:  ±0.10 m  (GPS + wind)
      - Position Z:   ±0.05 m  (altitude hold oscillation)
      - Roll/Pitch:   ±0.15°   (gimbal stabilization residual)
      - Yaw:          ±0.10°   (heading drift)

    Args:
        frames: List of camera frame dicts
        fps: Frame rate
        noise_level: Noise intensity 0.0–1.0
        rng: Random state

    Returns:
        Modified frames with noise added
    """
    n = len(frames)
    if noise_level <= 0 or n < 13:
        return frames

    # Noise parameters scaled by noise_level
    # Position noise: two frequency bands (low-freq drift + medium-freq vibration)
    pos_xy_amp = 0.20 * noise_level    # meters
    pos_z_amp = 0.10 * noise_level      # meters
    rot_rp_amp = 0.30 * noise_level     # degrees (roll/pitch)
    rot_yaw_amp = 0.20 * noise_level    # degrees (yaw)

    # Generate noise for each channel
    # Low-freq drift (wind gusts) — 0.3 Hz cutoff
    drift_x = generate_correlated_noise(n, fps, pos_xy_amp * 0.7, 0.3, rng)
    drift_y = generate_correlated_noise(n, fps, pos_xy_amp * 0.7, 0.3, rng)
    drift_z = generate_correlated_noise(n, fps, pos_z_amp * 0.7, 0.2, rng)

    # Medium-freq vibration (mechanical) — 2 Hz cutoff
    vib_x = generate_correlated_noise(n, fps, pos_xy_amp * 0.3, 2.0, rng)
    vib_y = generate_correlated_noise(n, fps, pos_xy_amp * 0.3, 2.0, rng)
    vib_z = generate_correlated_noise(n, fps, pos_z_amp * 0.3, 2.0, rng)

    # Rotation noise — medium freq (gimbal residual)
    noise_roll = generate_correlated_noise(n, fps, rot_rp_amp, 1.5, rng)
    noise_pitch = generate_correlated_noise(n, fps, rot_rp_amp, 1.5, rng)
    noise_yaw = generate_correlated_noise(n, fps, rot_yaw_amp, 0.5, rng)

    # Apply noise
    for i in range(n):
        frames[i]["position"]["x"] += drift_x[i] + vib_x[i]
        frames[i]["position"]["y"] += drift_y[i] + vib_y[i]
        frames[i]["position"]["z"] += drift_z[i] + vib_z[i]

        # Also apply to geodetic coordinates (approximate: convert m to degrees)
        # At ~52°N latitude: 1° lat ≈ 111,320m, 1° lon ≈ 65,000m
        lat_scale = 1.0 / 111320.0
        lon_scale = 1.0 / 65000.0
        frames[i]["coordinate"]["latitude"] += (drift_x[i] + vib_x[i]) * lat_scale
        frames[i]["coordinate"]["longitude"] += (drift_y[i] + vib_y[i]) * lon_scale
        frames[i]["coordinate"]["altitude"] += drift_z[i] + vib_z[i]

        frames[i]["rotation"]["x"] += noise_roll[i]
        frames[i]["rotation"]["y"] += noise_pitch[i]
        frames[i]["rotation"]["z"] += noise_yaw[i]

    return frames


def resample_trajectory(data: dict, target_max_speed_kmh: float,
                        noise_level: float = 0.0,
                        rng: np.random.RandomState = None,
                        verbose: bool = True) -> dict:
    """Resample a trajectory JSON to achieve a target maximum speed.

    Uses cubic spline interpolation for smooth results on:
      - ECEF position (x, y, z)
      - Geodetic coordinate (lat, lon, alt)
      - Rotation (x, y, z) with angle unwrapping
      - fovVertical

    Args:
        data: Trajectory JSON dict
        target_max_speed_kmh: Target max speed in km/h
        verbose: Print details

    Returns:
        Modified data dict with resampled cameraFrames
    """
    frames = data["cameraFrames"]
    fps = data["frameRate"]
    n_orig = len(frames)

    # ── 1. Compute current max speed ───────────────────────────────────────
    speeds = compute_ecef_speeds(frames, fps)
    current_max_kmh = speeds.max() * 3.6

    # Stretch factor: how many times more frames we need
    target_max_ms = target_max_speed_kmh / 3.6
    current_max_ms = speeds.max()
    stretch = current_max_ms / target_max_ms

    if stretch <= 1.0:
        if verbose:
            print(f"    Already below target ({current_max_kmh:.0f} ≤ "
                  f"{target_max_speed_kmh:.0f} km/h), skipping")
        return data

    n_new = int(np.ceil(n_orig * stretch))

    if verbose:
        print(f"    Current max: {current_max_kmh:.0f} km/h → "
              f"Target max: {target_max_speed_kmh:.0f} km/h")
        print(f"    Stretch: {stretch:.2f}x  ({n_orig} → {n_new} frames)")

    # ── 2. Extract channels ────────────────────────────────────────────────
    pos_x = np.array([f["position"]["x"] for f in frames])
    pos_y = np.array([f["position"]["y"] for f in frames])
    pos_z = np.array([f["position"]["z"] for f in frames])

    coord_lat = np.array([f["coordinate"]["latitude"] for f in frames])
    coord_lon = np.array([f["coordinate"]["longitude"] for f in frames])
    coord_alt = np.array([f["coordinate"]["altitude"] for f in frames])

    rot_x = np.array([f["rotation"]["x"] for f in frames])
    rot_y = np.array([f["rotation"]["y"] for f in frames])
    rot_z = np.array([f["rotation"]["z"] for f in frames])

    fov = np.array([f["fovVertical"] for f in frames])

    # ── 3. Unwrap rotation angles for smooth interpolation ─────────────────
    rot_x_uw = unwrap_angles(rot_x)
    rot_y_uw = unwrap_angles(rot_y)
    rot_z_uw = unwrap_angles(rot_z)

    # ── 4. Build cubic splines (natural boundary) ──────────────────────────
    t_orig = np.linspace(0, 1, n_orig)
    t_new = np.linspace(0, 1, n_new)

    channels = {
        "pos_x": pos_x, "pos_y": pos_y, "pos_z": pos_z,
        "coord_lat": coord_lat, "coord_lon": coord_lon, "coord_alt": coord_alt,
        "rot_x": rot_x_uw, "rot_y": rot_y_uw, "rot_z": rot_z_uw,
        "fov": fov,
    }

    interpolated = {}
    for name, values in channels.items():
        cs = CubicSpline(t_orig, values, bc_type="natural")
        interpolated[name] = cs(t_new)

    # ── 5. Build new cameraFrames ──────────────────────────────────────────
    new_frames = []
    for i in range(n_new):
        frame = {
            "position": {
                "x": float(interpolated["pos_x"][i]),
                "y": float(interpolated["pos_y"][i]),
                "z": float(interpolated["pos_z"][i]),
            },
            "rotation": {
                "x": float(interpolated["rot_x"][i]),
                "y": float(interpolated["rot_y"][i]),
                "z": float(interpolated["rot_z"][i]),
            },
            "coordinate": {
                "latitude": float(interpolated["coord_lat"][i]),
                "longitude": float(interpolated["coord_lon"][i]),
                "altitude": float(interpolated["coord_alt"][i]),
            },
            "fovVertical": float(interpolated["fov"][i]),
        }
        new_frames.append(frame)

    # ── 6. Add trajectory noise if requested ────────────────────────────────
    if noise_level > 0 and rng is not None:
        if verbose:
            print(f"    Adding trajectory noise (level={noise_level:.2f})…")
        new_frames = add_trajectory_noise(new_frames, fps, noise_level, rng)

    # ── 7. Update JSON metadata ────────────────────────────────────────────
    data["cameraFrames"] = new_frames
    data["numFrames"] = n_new
    data["durationSeconds"] = n_new / fps

    # ── 7. Verify result ───────────────────────────────────────────────────
    new_speeds = compute_ecef_speeds(new_frames, fps)
    new_max_kmh = new_speeds.max() * 3.6
    new_mean_kmh = new_speeds.mean() * 3.6

    if verbose:
        print(f"    Result: max={new_max_kmh:.1f} km/h, "
              f"mean={new_mean_kmh:.1f} km/h, frames={n_new}")

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Resample AeroPose trajectories for realistic drone speeds."
    )
    parser.add_argument("--sequences", nargs="+", default=None,
                        help="Specific sequence names to resample (e.g. airport2)")
    parser.add_argument("--all", action="store_true",
                        help="Resample all trajectories")
    parser.add_argument("--input-dir", type=Path, default=TRAJ_DIR,
                        help="Input trajectory directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: overwrite in-place with backup)")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip backup of original files")
    parser.add_argument("--target-speed", type=float, default=None,
                        help="Override: use a fixed target max speed (km/h) for all")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for target speed assignment")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Trajectory noise level 0.0–1.0 (default: 0, disabled). "
                             "Adds realistic drone vibrations and drift.")
    args = parser.parse_args()

    # ── Collect trajectory files ───────────────────────────────────────────
    if args.all:
        traj_files = sorted(args.input_dir.glob("*.json"))
    elif args.sequences:
        traj_files = []
        for name in args.sequences:
            p = args.input_dir / f"{name}.json"
            if not p.exists():
                print(f"  ✗ Not found: {p}")
                sys.exit(1)
            traj_files.append(p)
    else:
        parser.error("Specify --sequences or --all")

    print("=" * 65)
    print("AeroPose Trajectory Speed Normalization")
    print("=" * 65)
    print(f"  Input:    {args.input_dir}")
    print(f"  Output:   {args.output_dir or 'in-place (with backup)'}")
    print(f"  Sequences: {len(traj_files)}")
    if args.noise > 0:
        print(f"  Noise:    {args.noise:.2f}")
    print()

    # ── Load all trajectories to compute global target distribution ────────
    print("Loading trajectories…")
    all_data = {}
    all_max_speeds = []
    for tf in traj_files:
        with open(tf) as f:
            d = json.load(f)
        name = tf.stem
        all_data[name] = d
        speeds = compute_ecef_speeds(d["cameraFrames"], d["frameRate"])
        all_max_speeds.append(speeds.max() * 3.6)
    all_max_speeds = np.array(all_max_speeds)

    print(f"  Current max speeds: "
          f"{all_max_speeds.min():.0f}–{all_max_speeds.max():.0f} km/h "
          f"(mean {all_max_speeds.mean():.0f})")

    # ── Assign target speeds ───────────────────────────────────────────────
    if args.target_speed:
        target_speeds = np.full(len(traj_files), args.target_speed)
        print(f"  Fixed target max speed: {args.target_speed:.0f} km/h")
    else:
        target_speeds = assign_target_speeds(len(traj_files), all_max_speeds)
        print(f"  Target max speeds: "
              f"{target_speeds.min():.0f}–{target_speeds.max():.0f} km/h "
              f"(mean {target_speeds.mean():.0f})")
    print()

    # ── Backup originals ──────────────────────────────────────────────────
    output_dir = args.output_dir
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    elif not args.no_backup:
        TRAJ_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        for tf in traj_files:
            backup_path = TRAJ_BACKUP_DIR / tf.name
            if not backup_path.exists():
                shutil.copy2(tf, backup_path)
        print(f"  Backed up originals to {TRAJ_BACKUP_DIR}")
        print()

    # ── Resample each trajectory ───────────────────────────────────────────
    print("Resampling trajectories…")
    results = []
    names = list(all_data.keys())
    noise_rng = np.random.RandomState(args.seed + 1000)  # separate RNG for noise
    for i, (name, data) in enumerate(all_data.items()):
        print(f"  [{i+1}/{len(all_data)}] {name}")
        resampled = resample_trajectory(data, target_speeds[i],
                                        noise_level=args.noise,
                                        rng=noise_rng)

        # Save
        dst = (output_dir / f"{name}.json") if output_dir else (args.input_dir / f"{name}.json")
        with open(dst, "w") as f:
            json.dump(resampled, f)

        # Record result
        new_speeds = compute_ecef_speeds(resampled["cameraFrames"], resampled["frameRate"])
        results.append({
            "name": name,
            "old_max_kmh": all_max_speeds[i],
            "new_max_kmh": new_speeds.max() * 3.6,
            "new_mean_kmh": new_speeds.mean() * 3.6,
            "target_kmh": target_speeds[i],
            "old_frames": len(all_data[name]["cameraFrames"]),
            "new_frames": len(resampled["cameraFrames"]),
        })

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("Summary")
    print("=" * 65)
    print(f"  {'Sequence':<20} {'OldMax':>8} {'Target':>8} {'NewMax':>8} "
          f"{'OldFr':>7} {'NewFr':>7}")
    print("  " + "-" * 63)
    for r in results:
        print(f"  {r['name']:<20} {r['old_max_kmh']:>7.0f}h "
              f"{r['target_kmh']:>7.0f}h {r['new_max_kmh']:>7.1f}h "
              f"{r['old_frames']:>7} {r['new_frames']:>7}")

    new_maxes = [r["new_max_kmh"] for r in results]
    new_means = [r["new_mean_kmh"] for r in results]
    total_new_frames = sum(r["new_frames"] for r in results)
    print()
    print(f"  New max speeds: {min(new_maxes):.0f}–{max(new_maxes):.0f} km/h")
    print(f"  New mean speeds: {min(new_means):.0f}–{max(new_means):.0f} km/h")
    print(f"  Total frames: {total_new_frames:,}")
    print(f"  Output: {output_dir or args.input_dir}")
    print()


if __name__ == "__main__":
    main()
