from pathlib import Path
import numpy as np
import cv2
import random
from typing import Optional, Tuple, List

"""Augmentation utilities for sequences."""

def _generate_smooth_noise(n: int, scale: float = 1.0, smoothness: int = 15,
                           seed: Optional[int] = None) -> np.ndarray:
    """
    Generate temporally-correlated noise (drift-like, not white noise).
    Uses Gaussian filtering of random samples to produce smooth perturbations.

    Args:
        n: Number of samples
        scale: Standard deviation of the output noise
        smoothness: Kernel size for Gaussian smoothing (higher = smoother)
        seed: Random seed for reproducibility

    Returns:
        (n,) array of smooth noise values"""
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(n)
    smooth = gaussian_filter1d(raw, sigma=smoothness)
    # Normalize to desired scale
    if smooth.std() > 1e-8:
        smooth = smooth / smooth.std() * scale
    return smooth

def _generate_multiband_noise(n: int, scale: float, fps: float = 30.0,
                              seed: Optional[int] = None,
                              fast_weight: float = 0.30) -> np.ndarray:
    """
    Generate realistic drone position noise with two frequency bands:
      - Slow drift (wind / GPS wander): very smooth, ~3-5s period  (dominant)
      - Fast micro-vibration (motor vibration): ~0.1-0.3s period   (minor)

    The slow drift dominates (1 - fast_weight), giving a smooth GPS-drift look.
    A small fast component adds motor micro-jitter to the platform position.

    NOTE: Do NOT use this for rotation noise — use _generate_smooth_noise
    instead, because a gimbal absorbs all high-frequency rotation.

    Args:
        n: Number of samples
        scale: Overall standard deviation of the combined signal
        fps: Frame rate (for frequency scaling)
        seed: Random seed
        fast_weight: Weight of the fast vibration band [0, 1]. Default 0.30
                     means 70% slow drift + 30% fast vibration.

    Returns:
        (n,) array of multi-band noise"""
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.default_rng(seed)

    # Band 1: Slow drift — sigma = 3-4 seconds worth of frames
    # This is very gradual, like GPS wander or sustained wind.
    drift_sigma = max(5, int(3.5 * fps))  # ~3.5s at given fps
    drift = gaussian_filter1d(rng.standard_normal(n), sigma=drift_sigma)
    if drift.std() > 1e-8:
        drift = drift / drift.std()

    # Band 2: Fast micro-vibration — sigma = 2-4 frames
    # Motor vibration transmitted through the frame to GPS/IMU.
    vib_sigma = max(1, int(0.1 * fps))  # ~0.1s = 3 frames at 30fps
    vib = gaussian_filter1d(rng.standard_normal(n), sigma=vib_sigma)
    if vib.std() > 1e-8:
        vib = vib / vib.std()

    # Mix: mostly slow drift + small fast vibration
    combined = (1.0 - fast_weight) * drift + fast_weight * vib
    # Renormalize to desired scale
    if combined.std() > 1e-8:
        combined = combined / combined.std() * scale
    return combined

def perturb_trajectory(data: dict, position_noise_m: float = 0.15,
                       rotation_noise_deg: float = 0.03,
                       roll_noise_deg: float = 0.05,
                       bank_on_turns: bool = True,
                       bank_factor: float = 0.15,
                       seed: Optional[int] = None) -> dict:
    """
    Add realistic noise to a Google Earth Studio trajectory.

    Simulates real UAV behavior:
      - Position: mostly slow GPS/wind drift + small fast motor micro-vibration
        Uses _generate_multiband_noise (70% slow + 30% fast).
      - Rotation: SLOW gimbal drift ONLY — no fast vibration.
        A stabilized gimbal absorbs all vibration above ~1 Hz. Only slow
        thermal/pressure drift passes through (~0.01-0.05° over seconds).
        Uses _generate_smooth_noise with sigma ~3s.
      - Roll: Same slow-only gimbal drift. Yaw-rate banking added on top.

    Recommended values for realistic gimbal-stabilized drone footage:
      - position_noise_m: 0.10-0.20 m  (GPS drift + wind push)
      - rotation_noise_deg: 0.02-0.05°  (gimbal thermal drift — very small)
      - roll_noise_deg: 0.03-0.08°       (gimbal roll drift, slightly larger)

    Args:
        data: The full JSON dict (with 'cameraFrames')
        position_noise_m: Std-dev of position noise in meters (XY).
                          Z noise is 2/3 of this value.
        rotation_noise_deg: Std-dev of pitch/yaw gimbal drift in degrees.
                            Keep very small (0.02-0.05°) for realistic gimbal.
        roll_noise_deg: Std-dev of roll gimbal drift in degrees (base drift).
        bank_on_turns: If True, add roll proportional to yaw rate (banking)
        bank_factor: Multiplier for turn-banking (degrees roll per deg/frame yaw rate)
        seed: Random seed for reproducibility

    Returns:
        Modified data dict with perturbed cameraFrames"""
    import copy
    data = copy.deepcopy(data)
    frames = data['cameraFrames']
    n = len(frames)
    if n < 2:
        return data

    rng_base = seed if seed is not None else random.randint(0, 2**31)
    fps = data.get('frameRate', 30)

    # --- Position noise (multi-band: 70% slow GPS drift + 30% fast motor vibe) ---
    noise_x = _generate_multiband_noise(n, position_noise_m, fps, seed=rng_base, fast_weight=0.30)
    noise_y = _generate_multiband_noise(n, position_noise_m, fps, seed=rng_base + 1, fast_weight=0.30)
    # Z noise target: ~2/3 of XY so that at XY=0.15m → Z≈0.10m
    noise_z = _generate_multiband_noise(n, position_noise_m * (2.0 / 3.0), fps, seed=rng_base + 2, fast_weight=0.30)

    # Convert position noise from meters to approximate lat/lon degrees
    ref_lat = frames[0]['coordinate']['latitude']
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(ref_lat))

    for i, frame in enumerate(frames):
        frame['coordinate']['latitude'] += noise_y[i] / m_per_deg_lat
        frame['coordinate']['longitude'] += noise_x[i] / m_per_deg_lon
        frame['coordinate']['altitude'] += noise_z[i]

    # --- Rotation noise: SLOW DRIFT ONLY (no fast vibration — gimbal model) ---
    # A stabilized gimbal absorbs all vibration above ~1 Hz. The only rotation
    # perturbation that reaches the camera is very slow thermal/pressure drift.
    rot_sigma_frames = max(10, int(3.0 * fps))  # ~3s period at given fps
    noise_pitch = _generate_smooth_noise(n, rotation_noise_deg, smoothness=rot_sigma_frames, seed=rng_base + 3)
    noise_yaw = _generate_smooth_noise(n, rotation_noise_deg, smoothness=rot_sigma_frames, seed=rng_base + 4)

    for i, frame in enumerate(frames):
        frame['rotation']['x'] += noise_pitch[i]  # pitch
        frame['rotation']['y'] += noise_yaw[i]     # yaw

    # --- Roll drift: slow gimbal drift only (same model as pitch/yaw) ---
    roll_sigma_frames = max(10, int(3.0 * fps))  # ~3s period
    noise_roll = _generate_smooth_noise(n, roll_noise_deg, smoothness=roll_sigma_frames, seed=rng_base + 5)

    # Banking on turns: detect yaw rate and add proportional roll
    if bank_on_turns:
        yaw_values = np.array([f['rotation']['y'] for f in frames])
        yaw_rate = np.diff(yaw_values, prepend=yaw_values[0])
        yaw_rate = np.where(yaw_rate > 180, yaw_rate - 360, yaw_rate)
        yaw_rate = np.where(yaw_rate < -180, yaw_rate + 360, yaw_rate)
        from scipy.ndimage import gaussian_filter1d
        yaw_rate_smooth = gaussian_filter1d(yaw_rate, sigma=5)
        bank_roll = yaw_rate_smooth * bank_factor
    else:
        bank_roll = np.zeros(n)

    for i, frame in enumerate(frames):
        frame['rotation']['z'] += noise_roll[i] + bank_roll[i]

    print(f"   Trajectory perturbation applied:")
    print(f"     Position noise: ±{position_noise_m:.2f}m XY, ±{position_noise_m*2/3:.2f}m Z (70% slow drift, 30% fast motor vibe)")
    print(f"     Rotation noise: ±{rotation_noise_deg:.3f}° pitch/yaw (slow gimbal drift, sigma~3s — NO fast shake)")
    print(f"     Roll noise: ±{roll_noise_deg:.3f}° drift (slow gimbal)" +
          (f" + banking (factor={bank_factor})" if bank_on_turns else ""))

    return data

def apply_motion_blur(frames_dir: Path, poses: List[dict],
                      blur_strength: float = 1.0,
                      speed_threshold_ms: float = 15.0,
                      max_kernel_size: int = 9,
                      frame_rate: float = 30.0) -> int:
    """
    Apply realistic directional motion blur to rendered frames based on
    inter-frame camera velocity (m/s).

    Only frames with HIGH velocity get blur. Blur triggers above speed_threshold_ms
    (default 15.0 m/s ≈ 54 km/h) — only fast sport/racing drone maneuvers.

    The blur kernel size is computed as:
        ksize = clip((speed_ms - threshold) * blur_strength * 0.25 + 3, 3, max_kernel_size)
    Only the excess speed above threshold contributes to kernel size.
    Max kernel is 9 px to keep blur subtle (not smeared).

    Args:
        frames_dir: Directory containing frame_XXXX.{png,jpg} files
        poses: List of camera frame dicts (with 'coordinate')
        blur_strength: Global multiplier for blur intensity (1.0 = subtle realistic)
        speed_threshold_ms: Minimum speed in m/s below which no blur is applied.
                            Default 8.0 means only fast maneuvers get blur.
        max_kernel_size: Maximum blur kernel size in pixels (default: 21)
        frame_rate: Frame rate of the sequence (used to convert per-frame to per-sec)

    Returns:
        Number of frames that received motion blur"""
    frame_files = sorted(frames_dir.glob("frame_*.jpg")) or sorted(frames_dir.glob("frame_*.png"))
    if not frame_files or len(poses) < 2:
        return 0

    n = min(len(frame_files), len(poses))

    # Compute per-frame displacements in meters
    positions = np.array([
        [p['coordinate']['longitude'], p['coordinate']['latitude'], p['coordinate']['altitude']]
        for p in poses[:n]
    ])
    ref_lat = positions[0, 1]
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(ref_lat))
    pos_m = np.column_stack([
        positions[:, 0] * m_per_deg_lon,
        positions[:, 1] * m_per_deg_lat,
        positions[:, 2]
    ])

    # Per-frame displacement (meters/frame)
    disp = np.zeros((n, 3))
    disp[1:] = np.diff(pos_m, axis=0)
    if n > 1:
        disp[0] = disp[1]

    # Convert to speed in m/s
    speeds_ms = np.linalg.norm(disp, axis=1) * frame_rate

    # Rotation rates (deg/frame -> deg/s)
    rotations = np.array([
        [p['rotation']['x'], p['rotation']['y'], p['rotation']['z']]
        for p in poses[:n]
    ])
    rot_disp = np.zeros((n, 3))
    rot_disp[1:] = np.diff(rotations, axis=0)
    if n > 1:
        rot_disp[0] = rot_disp[1]
    angular_speed_ds = np.linalg.norm(rot_disp, axis=1) * frame_rate  # deg/s

    # Combined motion in m/s (angular contribution: 1 deg/s ~ 0.1 m/s equivalent)
    motion_ms = speeds_ms + angular_speed_ds * 0.1

    print(f"   Motion stats (m/s): min={motion_ms.min():.2f}, mean={motion_ms.mean():.2f}, "
          f"max={motion_ms.max():.2f}, threshold={speed_threshold_ms:.2f}")
    print(f"   Frames above threshold: {(motion_ms > speed_threshold_ms).sum()}/{n}")

    blurred_count = 0
    for i in range(n):
        if motion_ms[i] < speed_threshold_ms:
            continue

        frame_path = frame_files[i]
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        # Kernel size from excess speed above threshold only
        # thresh=15: 18 m/s → ksize=3+0.75=3, 27 m/s → ksize=3+3=6, max→9
        excess = motion_ms[i] - speed_threshold_ms
        ksize = int(np.clip(excess * blur_strength * 0.25 + 3, 3, max_kernel_size))
        if ksize % 2 == 0:
            ksize += 1  # Must be odd

        # Blur direction from XY velocity
        vx, vy = disp[i, 0], disp[i, 1]
        angle = np.degrees(np.arctan2(vy, vx)) if (abs(vx) + abs(vy)) > 1e-6 else 0.0

        # Create directional motion blur kernel
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        center = ksize // 2
        cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        for j in range(ksize):
            offset = j - center
            x = int(round(center + offset * cos_a))
            y = int(round(center + offset * sin_a))
            if 0 <= x < ksize and 0 <= y < ksize:
                kernel[y, x] = 1.0
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum

        blurred = cv2.filter2D(img, -1, kernel)
        # Preserve original format (jpg/png)
        cv2.imwrite(str(frame_path), blurred)
        blurred_count += 1

    return blurred_count

def apply_sudden_jitter(data: dict, jitter_frame: Optional[int] = None,
                        jitter_probability: float = 0.2,
                        jitter_magnitude_deg: float = 0.8,
                        jitter_duration_frames: int = 4,
                        seed: Optional[int] = None) -> Tuple[dict, Optional[int]]:
    """
    Simulate a sudden single-frame jitter event (e.g. wind gust, motor glitch).

    Real drones occasionally experience brief, sharp rotational disturbances
    that last 2-4 frames before the gimbal/controller corrects. This adds
    a spike-then-recover pattern to rotation channels.

    The jitter is biased toward the Z-axis (roll), since real wind gusts and
    motor glitches primarily cause roll disturbances — the gimbal stabilizes
    pitch/yaw more effectively than roll.

    Args:
        data: Trajectory dict with 'cameraFrames'
        jitter_frame: Specific frame index to place jitter. If None, placed
                      randomly (with probability jitter_probability).
        jitter_probability: Probability of jitter occurring (0-1). Only used
                            when jitter_frame is None. Default: 0.2 (20%)
        jitter_magnitude_deg: Peak rotation displacement in degrees. Default: 0.8
        jitter_duration_frames: Number of frames for the spike-recover pattern.
                                Default: 4 (1 frame spike + 3 frame smooth recovery)
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (modified data dict, jitter_frame_index or None if no jitter applied)"""
    import copy
    rng = np.random.default_rng(seed)
    frames = data['cameraFrames']
    n = len(frames)

    if n < jitter_duration_frames + 4:
        return data, None

    # Decide whether jitter occurs
    if jitter_frame is None:
        if rng.random() > jitter_probability:
            print("   Sudden jitter: skipped (probability roll)")
            return data, None
        # Random frame, avoiding first/last 10 frames
        margin = max(10, jitter_duration_frames + 2)
        jitter_frame = rng.integers(margin, max(margin + 1, n - margin))

    if jitter_frame < 0 or jitter_frame >= n - jitter_duration_frames:
        print(f"   Sudden jitter: frame {jitter_frame} out of range, skipping")
        return data, None

    data = copy.deepcopy(data)
    frames = data['cameraFrames']

    # Create gentle spike-then-recover pattern: moderate spike, smooth exponential decay
    # e.g. for duration=4: [0.6, -0.25, 0.08, -0.02] (gentle overshoot then settle)
    pattern = np.zeros(jitter_duration_frames)
    pattern[0] = 0.6  # main spike (gentle)
    for k in range(1, jitter_duration_frames):
        pattern[k] = -0.35 * (0.25 ** (k - 1))  # softer damped oscillation

    # Z-axis (roll) dominated jitter — real wind gusts primarily affect roll
    # since gimbal stabilizers handle pitch/yaw better than roll.
    # Roll gets full magnitude, pitch/yaw get reduced (0.3x).
    jitter_roll = rng.uniform(-1, 1) * jitter_magnitude_deg           # full magnitude
    jitter_pitch = rng.uniform(-1, 1) * jitter_magnitude_deg * 0.3    # reduced
    jitter_yaw = rng.uniform(-1, 1) * jitter_magnitude_deg * 0.3      # reduced

    for k in range(jitter_duration_frames):
        idx = jitter_frame + k
        if idx < n:
            frames[idx]['rotation']['x'] += jitter_pitch * pattern[k]
            frames[idx]['rotation']['y'] += jitter_yaw * pattern[k]
            frames[idx]['rotation']['z'] += jitter_roll * pattern[k]

    print(f"   Sudden jitter applied at frame {jitter_frame} "
          f"(magnitude={jitter_magnitude_deg:.1f}°, Z-biased, duration={jitter_duration_frames} frames)")

    return data, jitter_frame

def apply_wind_gust_episodes(data: dict, probability: float = 0.4,
                              n_episodes_max: int = 2,
                              episode_duration_s: float = 4.0,
                              gust_magnitude_deg: float = 0.35,
                              seed: Optional[int] = None) -> Tuple[dict, Optional[List[dict]]]:
    """
    Simulate episodic wind gust events: short bursts of strong fast camera shake
    that appear occasionally in a sequence (not for the full duration).

    Unlike apply_sudden_jitter (4-frame spike), these are sustained episodes of
    ~2-6 seconds of fast high-frequency vibration — simulating a strong wind gust
    that hits the drone and then subsides. Each episode fades in and out smoothly
    using a Hann window envelope so there are no abrupt transitions.

    Only activates with the given probability. When active, places 1-3 non-
    overlapping episodes at random positions in the trajectory.

    The fast vibration (sigma~0.1s = 3 frames) is identical to the old jitter
    model but CONFINED to the gust window. Outside gust windows, the normal slow
    gimbal drift continues unaffected.

    Args:
        data: Trajectory dict with 'cameraFrames'
        probability: Probability that any gusts occur at all. Default: 0.4 (40%)
        n_episodes_max: Maximum number of gust episodes. Actual count: 1-n_episodes_max.
                        Default: 2
        episode_duration_s: Duration of each gust (seconds, excl. ramps). Default: 4.0
        gust_magnitude_deg: Std-dev of rotation noise during gust (degrees). Default: 0.35
        seed: Random seed.

    Returns:
        Tuple of (modified data dict, list of episode dicts or None if no gusts)
        Episode dicts: {'start_frame': int, 'end_frame': int, 'duration_s': float}"""
    from scipy.ndimage import gaussian_filter1d
    import copy
    rng = np.random.default_rng(seed)
    frames = data['cameraFrames']
    n = len(frames)
    fps = data.get('frameRate', 30)

    # Decide whether gusts occur
    if rng.random() > probability:
        print("   Wind gust episodes: skipped (probability roll)")
        return data, None

    episode_frames = max(10, int(episode_duration_s * fps))
    ramp_frames = max(5, int(1.0 * fps))   # 1-second fade-in / fade-out
    full_episode_width = ramp_frames + episode_frames + ramp_frames

    # Must have room for at least one episode plus safety margins
    margin = int(2.0 * fps)
    usable_range = n - 2 * margin - full_episode_width
    if usable_range < 1:
        print(f"   Wind gust episodes: sequence too short ({n} frames), skipping")
        return data, None

    # Choose number of episodes
    n_episodes = rng.integers(1, n_episodes_max + 1)

    # Pick non-overlapping start positions
    episode_starts = []
    min_gap = full_episode_width + int(fps)  # at least 1s gap between gusts
    for _ in range(n_episodes * 10):  # try up to 10x to find valid starts
        if len(episode_starts) >= n_episodes:
            break
        candidate = rng.integers(margin, max(margin + 1, n - margin - full_episode_width))
        # Check overlap with existing episodes
        if all(abs(candidate - s) >= min_gap for s in episode_starts):
            episode_starts.append(int(candidate))

    if not episode_starts:
        print("   Wind gust episodes: could not place non-overlapping episodes, skipping")
        return data, None

    data = copy.deepcopy(data)
    frames = data['cameraFrames']

    # Build per-frame envelope (0 = no gust, 1 = full gust)
    envelope = np.zeros(n, dtype=np.float64)
    episodes_info = []
    for start in episode_starts:
        # Ramp up (Hann-half)
        for k in range(ramp_frames):
            idx = start + k
            if idx < n:
                t = k / ramp_frames
                envelope[idx] = max(envelope[idx], 0.5 * (1 - np.cos(np.pi * t)))
        # Full gust
        for k in range(episode_frames):
            idx = start + ramp_frames + k
            if idx < n:
                envelope[idx] = max(envelope[idx], 1.0)
        # Ramp down
        for k in range(ramp_frames):
            idx = start + ramp_frames + episode_frames + k
            if idx < n:
                t = k / ramp_frames
                envelope[idx] = max(envelope[idx], 0.5 * (1 + np.cos(np.pi * t)))
        end_frame = min(n - 1, start + full_episode_width)
        episodes_info.append({'start_frame': start, 'end_frame': end_frame,
                               'duration_s': episode_duration_s})

    # Generate fast vibration noise (sigma = 0.1s ≈ 3 frames)
    vib_sigma = max(1, int(0.1 * fps))
    def _fast_noise(s):
        raw = rng.standard_normal(n)
        v = gaussian_filter1d(raw, sigma=vib_sigma)
        return v / v.std() * gust_magnitude_deg if v.std() > 1e-8 else v

    gust_pitch = _fast_noise(0)
    gust_yaw   = _fast_noise(1)
    gust_roll  = _fast_noise(2)  # roll gets full magnitude (gimbal handles pitch/yaw better)

    for i, frame in enumerate(frames):
        e = envelope[i]
        if e > 1e-6:
            frame['rotation']['x'] += gust_pitch[i] * e * 0.4   # pitch: 40% of gust
            frame['rotation']['y'] += gust_yaw[i]   * e * 0.4   # yaw:   40%
            frame['rotation']['z'] += gust_roll[i]  * e          # roll:  100%

    ep_summary = ", ".join(f"frame {ep['start_frame']}-{ep['end_frame']}" for ep in episodes_info)
    print(f"   Wind gust episodes: {len(episodes_info)} episode(s) applied ({ep_summary}), "
          f"magnitude={gust_magnitude_deg:.2f}°, duration={episode_duration_s:.1f}s each")

    return data, episodes_info

def randomize_sun_direction(seed: Optional[int] = None) -> Tuple[List[float], float, float]:
    """
    Generate a random sun direction simulating different times of day and seasons.

    Returns:
        Tuple of (sun_direction_vector, azimuth_deg, elevation_deg)
        - sun_direction: [x, y, z] unit vector pointing TOWARD the sun
        - azimuth_deg: 0=North, 90=East, etc.
        - elevation_deg: angle above horizon (20-70°)"""
    rng = np.random.default_rng(seed)

    # Azimuth: uniform 0-360° (any compass direction)
    azimuth = rng.uniform(0, 360)
    # Elevation: 20-70° above horizon (avoids extreme low/high sun)
    # Bias toward 30-50° (most common)
    elevation = rng.uniform(20, 70)

    az_rad = np.deg2rad(azimuth)
    el_rad = np.deg2rad(elevation)

    # Convert to direction vector (pointing toward sun, in scene coords)
    # Scene coords: X=East, Y=North, Z=Up → sun_dir points FROM scene TOWARD sun
    # For lighting, Open3D wants direction of light rays (FROM sun TO scene) → negate Z
    sun_dir = [
        np.sin(az_rad) * np.cos(el_rad),   # East component
        np.cos(az_rad) * np.cos(el_rad),   # North component
        -np.sin(el_rad)                     # Downward (light rays go down)
    ]

    return sun_dir, azimuth, elevation

