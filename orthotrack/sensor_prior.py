"""
Simulated GPS + IMU sensor prior for the OrthoTrack pipeline.

Generates noisy position and orientation priors from GT poses,
modelling realistic DJI-class UAV sensor characteristics:

  GPS: horizontal ~3m, vertical ~5m (consumer, no RTK)
  IMU: roll/pitch ~1°, yaw (compass) ~4°"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

import numpy as np
from scipy.spatial.transform import Rotation


# ── Realistic noise profiles ──────────────────────────────────────────

GPS_NOISE_DEFAULTS = {
    'horizontal_sigma': 3.0,   # metres (DJI consumer, no RTK)
    'vertical_sigma': 5.0,     # metres (barometric + GPS fusion)
}

IMU_NOISE_DEFAULTS = {
    'roll_sigma': 1.0,     # degrees (accelerometer/gyro fusion)
    'pitch_sigma': 1.0,    # degrees
    'yaw_sigma': 4.0,      # degrees (magnetometer / compass)
}


@dataclass
class SensorPrior:
    """Per-frame noisy GPS + IMU prior, pre-computed from GT.

    Attributes
    ----------
    positions : dict[int, np.ndarray]
        frame_id -> noisy (x, y, z) in UTM.
    euler_angles : dict[int, np.ndarray]
        frame_id -> noisy (roll, pitch, yaw) in degrees."""
    positions: Dict[int, np.ndarray] = field(default_factory=dict)
    euler_angles: Dict[int, np.ndarray] = field(default_factory=dict)

    # Noise parameters used (for logging / reproducibility)
    gps_horizontal_sigma: float = GPS_NOISE_DEFAULTS['horizontal_sigma']
    gps_vertical_sigma: float = GPS_NOISE_DEFAULTS['vertical_sigma']
    imu_roll_sigma: float = IMU_NOISE_DEFAULTS['roll_sigma']
    imu_pitch_sigma: float = IMU_NOISE_DEFAULTS['pitch_sigma']
    imu_yaw_sigma: float = IMU_NOISE_DEFAULTS['yaw_sigma']

    def get_position(self, frame_id: int) -> Optional[np.ndarray]:
        """Return noisy GPS position (3,) or None if not available."""
        return self.positions.get(frame_id)

    def get_euler(self, frame_id: int) -> Optional[np.ndarray]:
        """Return noisy IMU euler angles (roll, pitch, yaw) in degrees, or None."""
        return self.euler_angles.get(frame_id)

    def get(self, frame_id: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (noisy_position, noisy_euler) for a frame."""
        return self.get_position(frame_id), self.get_euler(frame_id)

    @staticmethod
    def from_gt_reader(
        gt_reader,
        gps_horizontal_sigma: float = GPS_NOISE_DEFAULTS['horizontal_sigma'],
        gps_vertical_sigma: float = GPS_NOISE_DEFAULTS['vertical_sigma'],
        imu_roll_sigma: float = IMU_NOISE_DEFAULTS['roll_sigma'],
        imu_pitch_sigma: float = IMU_NOISE_DEFAULTS['pitch_sigma'],
        imu_yaw_sigma: float = IMU_NOISE_DEFAULTS['yaw_sigma'],
        seed: int = 42,
    ) -> 'SensorPrior':
        """Create a SensorPrior by adding realistic noise to GT poses.

        Parameters
        ----------
        gt_reader : PoseLoader or CSVPoseLoader
            Ground-truth pose loader.
        gps_horizontal_sigma : float
            Std-dev of horizontal GPS noise in metres.
        gps_vertical_sigma : float
            Std-dev of vertical GPS noise in metres.
        imu_roll_sigma, imu_pitch_sigma, imu_yaw_sigma : float
            Std-dev of IMU euler angle noise in degrees.
        seed : int
            Random seed for reproducibility."""
        rng = np.random.RandomState(seed)
        positions = {}
        euler_angles = {}

        for pose in gt_reader.poses:
            fid = pose.frame_id
            gt_pos = pose.position  # (3,)

            # GPS noise
            dx = rng.normal(0, gps_horizontal_sigma)
            dy = rng.normal(0, gps_horizontal_sigma)
            dz = rng.normal(0, gps_vertical_sigma)
            positions[fid] = gt_pos + np.array([dx, dy, dz])

            # IMU noise — add to GT euler angles
            gt_euler = np.array(pose.euler_angles)  # (roll, pitch, yaw) degrees
            d_roll = rng.normal(0, imu_roll_sigma)
            d_pitch = rng.normal(0, imu_pitch_sigma)
            d_yaw = rng.normal(0, imu_yaw_sigma)
            euler_angles[fid] = gt_euler + np.array([d_roll, d_pitch, d_yaw])

        return SensorPrior(
            positions=positions,
            euler_angles=euler_angles,
            gps_horizontal_sigma=gps_horizontal_sigma,
            gps_vertical_sigma=gps_vertical_sigma,
            imu_roll_sigma=imu_roll_sigma,
            imu_pitch_sigma=imu_pitch_sigma,
            imu_yaw_sigma=imu_yaw_sigma,
        )

    def get_rotation_matrix(self, frame_id: int) -> Optional[np.ndarray]:
        """Return noisy C2W rotation matrix (3x3) from IMU euler angles.
        
        The stored euler angles come from the C2W quaternion decomposition
        (CameraPose.euler_angles decomposes the C2W quaternion stored in
        poses.csv), so the rotation reconstructed here is already C2W."""
        euler = self.get_euler(frame_id)
        if euler is None:
            return None
        # Euler angles represent R_c2w → reconstruct directly
        R_c2w = Rotation.from_euler('xyz', euler, degrees=True).as_matrix()
        return R_c2w

    def summary(self) -> str:
        """Return a summary string of the noise configuration."""
        return (
            f"SensorPrior(GPS: horiz={self.gps_horizontal_sigma:.1f}m, "
            f"vert={self.gps_vertical_sigma:.1f}m | "
            f"IMU: roll={self.imu_roll_sigma:.1f}°, pitch={self.imu_pitch_sigma:.1f}°, "
            f"yaw={self.imu_yaw_sigma:.1f}°)"
        )
