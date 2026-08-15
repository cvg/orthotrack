import torch
import torch.nn.functional as F
import numpy as np
import json
import csv
import cv2
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.spatial.transform import Rotation
import pyproj
from pathlib import Path


@dataclass
class CameraIntrinsics:
    """Camera intrinsics (independent of pose).

    Attributes:
        fov_vertical: Vertical field of view in degrees.
        width: Image width in pixels.
        height: Image height in pixels.
        fx: Focal length x (pixels). If None, computed from fov_vertical.
        fy: Focal length y (pixels). If None, computed from fov_vertical.
        cx: Principal point x (pixels). If None, uses width / 2.
        cy: Principal point y (pixels). If None, uses height / 2."""
    fov_vertical: float = 0.0
    width: int = 0
    height: int = 0
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    # OpenCV-style radial/tangential distortion: [k1, k2, p1, p2, k3].
    # ``None`` means "not known / not applied" (frames are treated as undistorted).
    dist_coef: Optional[np.ndarray] = None

    @property
    def K(self) -> np.ndarray:
        """Return 3x3 intrinsics matrix."""
        h, w = self.height, self.width
        _fy = self.fy if self.fy is not None else h / (2 * np.tan(np.radians(self.fov_vertical) / 2))
        _fx = self.fx if self.fx is not None else _fy
        _cx = self.cx if self.cx is not None else w / 2
        _cy = self.cy if self.cy is not None else h / 2
        return np.array([[_fx, 0, _cx], [0, _fy, _cy], [0, 0, 1]], dtype=np.float64)


    @staticmethod
    def from_meta(sequence_dir: str) -> 'CameraIntrinsics':
        """Load intrinsics from intrinsics.json / meta.json in a sequence dir."""
        seq_path = Path(sequence_dir)
        intr = None
        fov = 0.0
        width, height = 0, 0

        # Try intrinsics.json first (has fx/fy/cx/cy)
        intr_path = seq_path / 'intrinsics.json'
        if intr_path.exists():
            import json as _json
            with open(intr_path) as f:
                intr = _json.load(f)
            width = intr.get('width', 0)
            height = intr.get('height', 0)

        # Try meta.json for fov_vertical and dimensions
        meta_path = seq_path / 'meta.json'
        if meta_path.exists():
            import json as _json
            with open(meta_path) as f:
                meta = _json.load(f)
            fov = meta.get('fov_vertical', fov)
            if width == 0:
                width = meta.get('width', 0)
            if height == 0:
                height = meta.get('height', 0)
            # meta.json may also have intrinsics dict
            if intr is None:
                intr = meta.get('intrinsics', None)

        # Fall back: read fov_vertical from first row of poses.csv
        if fov <= 0:
            poses_path = seq_path / 'poses.csv'
            if poses_path.exists():
                with open(poses_path) as f:
                    reader = csv.DictReader(f)
                    row = next(reader, None)
                    if row and 'fov_vertical' in row:
                        fov = float(row['fov_vertical'])

        fx = intr.get('fx') if intr else None
        fy = intr.get('fy') if intr else None
        cx = intr.get('cx') if intr else None
        cy = intr.get('cy') if intr else None

        # Optional distortion (OpenCV convention). Accept either an explicit
        # ``dist`` list ([k1, k2, p1, p2, k3]) or scalar fields (k1/k2/p1/p2/k3).
        dist_arr = None
        if intr is not None:
            raw_dist = intr.get('dist') or intr.get('distortion') or intr.get('dist_coef')
            if raw_dist is not None:
                try:
                    dist_arr = np.asarray(list(raw_dist), dtype=np.float64).reshape(-1)
                except Exception:
                    dist_arr = None
            if dist_arr is None:
                k1 = intr.get('k1'); k2 = intr.get('k2')
                p1 = intr.get('p1'); p2 = intr.get('p2'); k3 = intr.get('k3')
                if any(v is not None for v in (k1, k2, p1, p2, k3)):
                    dist_arr = np.array([
                        float(k1) if k1 is not None else 0.0,
                        float(k2) if k2 is not None else 0.0,
                        float(p1) if p1 is not None else 0.0,
                        float(p2) if p2 is not None else 0.0,
                        float(k3) if k3 is not None else 0.0,
                    ], dtype=np.float64)
            if dist_arr is not None and not np.any(np.abs(dist_arr) > 1e-9):
                dist_arr = None  # all zeros = no-op

        # If fov_vertical was not loaded explicitly, derive it from fy + height.
        # This ensures footprint / altitude estimates work correctly when only
        # fx/fy/cx/cy are provided (e.g. from a COLMAP-calibrated intrinsics.json).
        if fov <= 0 and fy is not None and height > 0:
            import math as _math
            fov = 2.0 * _math.degrees(_math.atan(height / (2.0 * float(fy))))

        return CameraIntrinsics(
            fov_vertical=float(fov),
            width=int(width), height=int(height),
            fx=float(fx) if fx is not None else None,
            fy=float(fy) if fy is not None else None,
            cx=float(cx) if cx is not None else None,
            cy=float(cy) if cy is not None else None,
            dist_coef=dist_arr,
        )


@dataclass
class CameraPose:
    """Represents a camera pose (translation + rotation) in the local UTM coordinate system."""
    frame_id: int
    
    # Position in UTM coordinates
    x: float  # UTM easting
    y: float  # UTM northing
    z: float  # Altitude above ground level
    
    # Rotation as quaternion (w, x, y, z)
    qw: float
    qx: float
    qy: float
    qz: float
    
    @property
    def position(self) -> np.ndarray:
        """Return position as numpy array."""
        return np.array([self.x, self.y, self.z])
    
    @property
    def quaternion(self) -> np.ndarray:
        """Return quaternion as numpy array (w, x, y, z)."""
        return np.array([self.qw, self.qx, self.qy, self.qz])
    
    @property
    def rotation_matrix(self) -> np.ndarray:
        """Return 3x3 rotation matrix (cached after first computation)."""
        if not hasattr(self, '_rotation_matrix_cache'):
            r = Rotation.from_quat([self.qx, self.qy, self.qz, self.qw])
            self._rotation_matrix_cache = r.as_matrix()
        return self._rotation_matrix_cache
    
    @property
    def euler_angles(self) -> Tuple[float, float, float]:
        """Return Euler angles (roll, pitch, yaw) in degrees."""
        r = Rotation.from_quat([self.qx, self.qy, self.qz, self.qw])
        return tuple(r.as_euler('xyz', degrees=True))
    


def get_camera_center_from_w2c(w2c):
    """
    Compute camera center (C) from world-to-camera matrix (W2C).
    C = -R^T * t"""
    if torch.is_tensor(w2c):
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        return -R.t() @ t
    else:
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        return -R.T @ t


def decompose_pose(pose):
    """
    Extract rotation matrix R and translation vector t from 4x4 pose matrix."""
    if torch.is_tensor(pose):
        # Support batching
        if pose.ndim == 3:
            return pose[:, :3, :3], pose[:, :3, 3]
        return pose[:3, :3], pose[:3, 3]
    else:
        return pose[:3, :3], pose[:3, 3]


def compute_w2c_translation(R, C):
    """
    Compute W2C translation vector t from rotation R and camera center C.
    t = -R * C"""
    if torch.is_tensor(R):
        if R.dtype != C.dtype:
            C = C.to(R.dtype)
        if R.ndim == 3:
            return -torch.bmm(R, C.unsqueeze(-1)).squeeze(-1)
        return -R @ C
    else:
        return -R @ C


def quat_pos_to_w2c(x: float, y: float, z: float,
                    qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """
    Build a 4x4 world-to-camera matrix from position and quaternion.

    Args:
        x, y, z: Camera position in world coordinates.
        qw, qx, qy, qz: Camera-to-world rotation quaternion (scalar-first convention,
                         but scipy expects [x, y, z, w]).

    Returns:
        (4, 4) world-to-camera matrix."""
    rot_c2w = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    c2w = np.eye(4)
    c2w[:3, :3] = rot_c2w
    c2w[:3, 3] = [x, y, z]
    return np.linalg.inv(c2w)














def compute_pose_error(
    R_pred: torch.Tensor,
    t_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute rotation and translation errors (Torch version).
    
    Args:
        R_pred: (B, 3, 3) predicted rotation
        t_pred: (B, 3, 1) predicted translation
        R_gt: (B, 3, 3) ground truth rotation
        t_gt: (B, 3, 1) ground truth translation
    
    Returns:
        rot_error: (B,) rotation error in degrees
        trans_error: (B,) translation error in meters"""
    # Rotation error
    R_diff = torch.bmm(R_pred, R_gt.transpose(1, 2))
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_angle = (trace - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1, 1)
    rot_error = torch.acos(cos_angle) * 180 / torch.pi  # degrees
    
    # Translation error
    trans_error = (t_pred - t_gt).norm(dim=1).squeeze(-1)  # meters
    
    return rot_error, trans_error


def parse_google_earth_rotation(rotation_dict: Dict, ecef_pos: Dict, ecef_transformer=None) -> Rotation:
    """
    Verified convention from convert_dataset.py:
    1. Create rotation from JSON (XYZ extrinsic).
    2. Get lat/lon from ECEF position (reference for the rotation).
    3. Transform from ECEF to local ENU.
    4. Compute final Camera-to-World rotation in ENU.
    
    Args:
        rotation_dict: Dict with 'x', 'y', 'z' rotation angles
        ecef_pos: Dict with 'x', 'y', 'z' ECEF position
        ecef_transformer: Optional pre-created pyproj.Transformer (EPSG:4978->4326).
                          If None, one is created (expensive per call)."""
    # 1. Rotation from JSON
    rx, ry, rz = rotation_dict['x'], rotation_dict['y'], rotation_dict['z']
    R_json = Rotation.from_euler('XYZ', [rx, ry, rz], degrees=True).as_matrix()
    
    # 2. Reference lat/lon from ECEF position
    if ecef_transformer is None:
        ecef_transformer = pyproj.Transformer.from_crs('EPSG:4978', 'EPSG:4326', always_xy=True)
    lon_ecef, lat_ecef, _ = ecef_transformer.transform(ecef_pos['x'], ecef_pos['y'], ecef_pos['z'])
    
    # 3. Transform from ECEF to local ENU at the ECEF-derived location
    lat_rad, lon_rad = np.deg2rad(lat_ecef), np.deg2rad(lon_ecef)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    
    R_ecef_to_enu = np.array([
        [-sin_lon,           cos_lon,            0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon,  cos_lat * sin_lon,  sin_lat]
    ])
    
    # 4. Final camera rotation in ENU frame (camera-to-world)
    R_cam_to_enu = R_ecef_to_enu @ R_json
    return Rotation.from_matrix(R_cam_to_enu)

class PoseLoader:
    """Loads and manages ground truth poses from Google Earth Studio JSON."""
    
    def __init__(self, json_path: str, utm_zone: int = 33):
        self.json_path = json_path
        self.utm_zone = utm_zone
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.name = self.data['name']
        # NOTE: GES 'numFrames' is the last frame INDEX (0-based),
        # so actual count = len(cameraFrames) = numFrames + 1.
        self.num_frames = len(self.data['cameraFrames'])
        self.width = self.data['width']
        self.height = self.data['height']
        
        # Create transformer for lat/lon to UTM
        # Use provided zone (default 33 for Berlin)
        self.transformer = pyproj.Transformer.from_crs(
            'EPSG:4326', f'EPSG:326{utm_zone}', always_xy=True
        )
        
        # Parse all camera frames
        self.poses = self._parse_poses()
        
        # Per-frame FOV and lat/lon (metadata, not stored in CameraPose)
        frames = self.data['cameraFrames']
        self.fov_verticals = [f['fovVertical'] for f in frames]
        self.latitudes = [f['coordinate']['latitude'] for f in frames]
        self.longitudes = [f['coordinate']['longitude'] for f in frames]
        
        # Reference point (first frame)
        first_frame = self.data['cameraFrames'][0]
        self.ref_lat = first_frame['coordinate']['latitude']
        self.ref_lon = first_frame['coordinate']['longitude']
        self.ref_alt = first_frame['coordinate']['altitude']
    
    def _parse_poses(self) -> List[CameraPose]:
        """Parse all camera poses from the JSON data."""
        poses = []
        
        # Create ECEF transformer ONCE (expensive to construct)
        ecef_transformer = pyproj.Transformer.from_crs('EPSG:4978', 'EPSG:4326', always_xy=True)
        
        # Vectorize lat/lon to UTM transform
        frames = self.data['cameraFrames']
        lons = np.array([f['coordinate']['longitude'] for f in frames])
        lats = np.array([f['coordinate']['latitude'] for f in frames])
        xs, ys = self.transformer.transform(lons, lats)
        
        for i, frame in enumerate(frames):
            coord = frame['coordinate']
            rotation = frame['rotation']
            
            lon, lat = coord['longitude'], coord['latitude']
            x, y = xs[i], ys[i]
            z = coord['altitude']
            
            # Parse rotation using scipy (with cached transformer)
            r = parse_google_earth_rotation(rotation, frame['position'], ecef_transformer=ecef_transformer)
            quat = r.as_quat()  # Returns (x, y, z, w)
            
            pose = CameraPose(
                frame_id=i,
                x=x, y=y, z=z,
                qw=quat[3], qx=quat[0], qy=quat[1], qz=quat[2],
            )
            poses.append(pose)
        
        return poses
    
    def get_pose(self, frame_id: int) -> CameraPose:
        """Get pose for a specific frame."""
        return self.poses[frame_id]
    
    def get_fov_vertical(self, frame_id: int) -> float:
        """Get per-frame vertical FOV in degrees."""
        return self.fov_verticals[frame_id]
    
    def get_all_positions_utm(self) -> np.ndarray:
        """Get all positions as (N, 3) array in UTM coordinates."""
        return np.array([[p.x, p.y, p.z] for p in self.poses])
    
    def get_trajectory_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box of the trajectory in UTM coordinates."""
        positions = self.get_all_positions_utm()
        x_min, y_min = positions[:, 0].min(), positions[:, 1].min()
        x_max, y_max = positions[:, 0].max(), positions[:, 1].max()
        return x_min, y_min, x_max, y_max
    
    def compute_pose_error(self, estimated_pose: CameraPose, 
                           gt_pose: CameraPose) -> Dict[str, float]:
        """
        Compute error metrics between estimated and ground truth pose.
        
        Returns:
            Dictionary with error metrics"""
        # Position error
        pos_error = np.linalg.norm(estimated_pose.position - gt_pose.position)
        
        # Horizontal error (XY only)
        horizontal_error = np.linalg.norm(
            estimated_pose.position[:2] - gt_pose.position[:2]
        )
        
        # Vertical error
        vertical_error = abs(estimated_pose.z - gt_pose.z)
        
        # Rotation error
        R_est = estimated_pose.rotation_matrix
        R_gt = gt_pose.rotation_matrix
        R_diff = R_est @ R_gt.T
        angle_error = np.rad2deg(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
        
        return {
            'position_error': pos_error,
            'horizontal_error': horizontal_error,
            'vertical_error': vertical_error,
            'rotation_error': angle_error
        }


class CSVPoseLoader:
    """Loads and manages ground truth poses from CSV file (compact format)."""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.poses = self._parse_poses()
        self.num_frames = len(self.poses)
        
        # Get dimensions from first pose (default if not available)
        self.width = 512
        self.height = 512
        
    def _parse_poses(self) -> List[CameraPose]:
        """Parse all camera poses from the CSV file."""
        poses = []
        
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pose = CameraPose(
                    frame_id=int(row['frame_id']),
                    x=float(row['x']),
                    y=float(row['y']),
                    z=float(row['z']),
                    qw=float(row['qw']),
                    qx=float(row['qx']),
                    qy=float(row['qy']),
                    qz=float(row['qz']),
                )
                poses.append(pose)
        
        return poses
    
    def get_pose(self, frame_id: int) -> CameraPose:
        """Get pose for a specific frame."""
        return self.poses[frame_id]
    
    def get_all_positions_utm(self) -> np.ndarray:
        """Get all positions as (N, 3) array in UTM coordinates."""
        return np.array([[p.x, p.y, p.z] for p in self.poses])
    
    def get_trajectory_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box of the trajectory in UTM coordinates."""
        positions = self.get_all_positions_utm()
        x_min, y_min = positions[:, 0].min(), positions[:, 1].min()
        x_max, y_max = positions[:, 0].max(), positions[:, 1].max()
        return x_min, y_min, x_max, y_max


def compute_intrinsics(image_width: int, image_height: int, 
                       fov_vertical_deg: float) -> np.ndarray:
    """
    Compute camera intrinsics from vertical FOV.
    
    Args:
        image_width, image_height: Image dimensions
        fov_vertical_deg: Vertical field of view in degrees
        
    Returns:
        3x3 intrinsic matrix K"""
    fov_rad = np.deg2rad(fov_vertical_deg)
    fy = image_height / (2 * np.tan(fov_rad / 2))
    fx = fy  # Assume square pixels
    cx = image_width / 2
    cy = image_height / 2
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return K

def load_intrinsics(intrinsics_path: str) -> Dict[str, float]:
    """Load camera intrinsics from file.
    
    Args:
        intrinsics_path: Path to intrinsics.txt file
        
    Returns:
        Dictionary with fx, fy, cx, cy, width, height, fov_vertical"""
    intrinsics = {}
    with open(intrinsics_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            if '=' in line:
                key, value = line.split('=')
                intrinsics[key.strip()] = float(value.strip())
    return intrinsics


def rotation_to_quat(R_c2w):
    """Convert rotation matrix to (qx, qy, qz, qw) or None."""
    if R_c2w is None:
        return None
    try:
        return Rotation.from_matrix(R_c2w).as_quat()  # scipy: x,y,z,w
    except Exception:
        return None
def build_intrinsics_matrix(gt_data: dict) -> np.ndarray:
    """Build 3×3 intrinsics matrix from GT data."""
    intr = gt_data['intrinsics']
    return np.array([
        [intr['fx'], 0, intr['cx']],
        [0, intr['fy'], intr['cy']],
        [0, 0, 1],
    ], dtype=np.float32)

