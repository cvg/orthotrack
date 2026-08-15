import os
import json
import urllib.request
import urllib.error
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms.functional as TF
import random
from utils.depth import depth_to_normals_np, load_depth_npz, load_depth_png
from typing import Optional, List, Union, Tuple, Dict, Any

from utils.geo import (
    get_visible_footprint,
    dsm_to_xyz,
    unproject_depth_to_world,
)
from utils.pose import get_camera_center_from_w2c, decompose_pose, compute_w2c_translation, quat_pos_to_w2c
from utils.data_utils import augment_data, normalize_data
from utils.tensor_ops import normalize

# Prevent OpenCV from spawning multiple threads in dataloader workers
cv2.setNumThreads(0)


class MovingDrone(Dataset):
    """
    Unified MovingDrone Dataset supporting both single-frame and sequence modes.

    
    Usage Modes:
        - Single-frame mode (sequence_length=1): Returns samples without temporal dimension,
          compatible with OrthoLocDataset for OrthoPose training.
        - Sequence mode (sequence_length>1): Returns samples with temporal dimension (L, ...),
          suitable for video-based tracking networks.
    
    Split Configuration:
        Looks for splits.json in dataset_dir with format:
            {"train": [...], "val": [...], "test_inPlace": [...], "test_outPlace": [...]}
        Supported split values: 'train', 'val', 'test_inPlace', 'test_outPlace',
        'test' (union of test_inPlace + test_outPlace), or 'all'.
        If splits.json not found, uses default 70/15/15 split based on sorted sequence names.
    
    Sub-sampling:
        Use stride parameter to sub-sample frames within sequences (stride=10 uses every 10th frame).
        For training, stride=10 is recommended to reduce temporal redundancy.

    Returns a dict with:
        Shared (no time dim): dop, dsm, geodata_mask, intrinsics, intrinsics_raw,
            lod_vertices/faces/labels/lod_dop, lidar_world, crop_bounds, scale, gsd, lod_level
        Per-frame (time dim L if sequence_length>1): query, depth, point_map, query_mask,
            extrinsics, extrinsics_raw, R, t, lod_uav, lidar_uav, matches,
            correspondence maps (if return_matching), sample_id

    Args:
        dataset_dir: Path to MovingDrone dataset root (containing scenes/, splits.json).
        split: 'train', 'val', 'test_inPlace', 'test_outPlace', 'test', or 'all'.
        stride: Frame sub-sampling stride (1=all frames, 10=every 10th frame).
        sequence_length: Controls temporal behavior.
            - int: fixed length (1 = single frame, squeezed for OrthoLoc compatibility)
            - tuple (min, max): random uniform from range each access
            - list [a, b, c]: random choice from list each access"""
    # Default GSD limits (used when not specified in constructor)
    DEFAULT_MIN_GSD = 0.1  # 10 cm/px - prevents overly zoomed-in crops
    DEFAULT_MAX_GSD = 2.0  # 2 m/px - prevents overly large crops from oblique views

    # Keys whose tensors may vary in size across samples (collated as lists, not stacked)
    VARIABLE_SIZE_KEYS = {'dop_fullres', 'dsm_fullres'}

    @staticmethod
    def collate_fn(batch):
        """Custom collate that keeps variable-size tensors as lists instead of stacking.
        
        Use this when return_fullres_dop or return_fullres_dsm is True with batch_size > 1:
            DataLoader(dataset, batch_size=B, collate_fn=MovingDrone.collate_fn)"""
        from torch.utils.data.dataloader import default_collate
        
        # Separate variable-size keys from stackable ones
        var_keys = MovingDrone.VARIABLE_SIZE_KEYS
        elem = batch[0]
        
        # For each key: if it's a variable-size key with tensors, keep as list; 
        # otherwise use default collate behavior
        result = {}
        stackable = {}
        for key in elem:
            vals = [d[key] for d in batch]
            if key in var_keys and isinstance(vals[0], torch.Tensor):
                result[key] = vals  # List of tensors (varying sizes)
            else:
                stackable[key] = vals
        
        # Use default collate for everything else
        if stackable:
            # Build a list of dicts with only stackable keys
            stackable_batch = [{k: d[k] for k in stackable if k in d} for d in batch]
            try:
                stacked = default_collate(stackable_batch)
                result.update(stacked)
            except (RuntimeError, TypeError):
                # Fallback: collate each key individually
                for key in stackable:
                    vals = [d[key] for d in batch if key in d]
                    if not vals:
                        continue
                    try:
                        result[key] = default_collate(vals)
                    except (RuntimeError, TypeError):
                        result[key] = vals  # Keep as list
        
        return result

    def __init__(self, dataset_dir=None, split=None, sequences=None, predownload=False,
                 size=(512, 512), normalization='01', xyz_normalization='none',
                 augment=False, force_augment=False, stride=1, lod_level='2', load_lidar=False,
                 load_dop=True, load_dsm=True, load_lod=True, load_depth=True, load_normals=False,
                 crop_multiplier_range=(1.2, 1.5), ensure_coverage=True,
                 return_matching: bool = False, min_gsd: Optional[float] = None,
                 max_gsd: Optional[float] = None,
                 sequence_length: Union[int, Tuple[int, int], List[int]] = 1,
                 enable_noise: bool = False, enable_flip: bool = False, enable_rotation: bool = True,
                 crop_mode: str = 'pointmap',
                 preserve_dsm_gsd: bool = False,
                 return_fullres_dop: bool = False,
                 return_fullres_dsm: bool = False,
                 return_resized_dop: bool = True,
                 return_resized_dsm: bool = True,
                 dop_year: Union[str, int] = 'last'):
        """
        Initialize MovingDrone dataset.
        
        Args:
            dataset_dir: Path to MovingDrone root (must contain scenes/ directory). If None, uses MOVINGDRONE_DIR env var.
            split: 'train', 'val', 'test', or 'all'. See class docstring for split config.
            sequences: List of specific sequence names to load (mutually exclusive with split).
            predownload: If True, downloads all required sequences during init. If False, downloads on the fly.
            size: Output image size (H, W). Default: (512, 512).
            normalization: Image normalization: '01' (0-1 range) or 'imagenet'.
            xyz_normalization: DSM/point_map normalization: 'none', 'mean_std', 'minmax_01', 'minmax_11'.
            augment: Enable data augmentation (flips, color jitter, rotation).
            force_augment: Force all augmentations to be applied.
            stride: Frame sub-sampling stride. Use stride=10 for training to reduce redundancy.
            sequence_length: 1 for single-frame mode (OrthoPose), >1 for sequences (tracking).
            load_dop: Load DOP orthophoto tiles.
            load_dsm: Load DSM elevation tiles.
            load_depth: Load pre-rendered depth maps.
            load_normals: Load (or synthesise) surface normals.
                If the pre-rendered normals/ folder exists for a frame, it is used directly.
                Otherwise normals are computed on-the-fly from the depth map via
                cross-product back-projection (mean angular error ~0.5° vs rendered GT).
                Default: False (normals are no longer stored by default in new sequences).
            load_lod: Load LoD building meshes.
            load_lidar: Load LiDAR point clouds.
            return_matching: Compute dense correspondence maps (query_in_dop, dop_in_query).
            crop_mode: How to determine DOP/DSM crop bounds:
                - 'pointmap': Use actual point_map bounds (tighter, better for training).
                - 'footprint': Use camera footprint projection (larger context, better for eval).
            min_gsd: Minimum GSD (m/px) - caps maximum resolution (prevents overly zoomed-in crops).
                     If None, uses DEFAULT_MIN_GSD (0.1m).
            max_gsd: Maximum GSD (m/px) - caps minimum resolution (prevents overly zoomed-out crops).
                     If None, uses DEFAULT_MAX_GSD (2.0m).
            preserve_dsm_gsd: If True, the DSM is returned at native GSD resolution without
                     resizing, avoiding interpolation artifacts at building edges. The DOP
                     and query are still resized to `size` for the matcher. The DSM may have
                     different spatial dimensions than DOP — this works because the model's
                     warp_to_3d uses normalized grid_sample coords. Best for evaluation/inference."""
        super().__init__()
        
        if sequences is not None and split is not None:
            raise ValueError("Cannot define both 'split' and 'sequences' at the same time.")
        if sequences is None and split is None:
            split = 'train'
            
        if dataset_dir is None:
            dataset_dir = os.environ.get("MOVINGDRONE_DIR", os.path.expanduser("~/.cache/movingdrone"))
            
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self._target_sequences = sequences
        self.predownload = predownload
        self.sequences = []
        self._video_caps = {}  # {video_path_str: (cap, last_fid)}

        self.size = size
        self.normalization = normalization
        self.xyz_normalization = xyz_normalization
        self.augment = augment
        self.force_augment = force_augment
        self.stride = stride
        self.lod_level = lod_level
        self.load_lidar = load_lidar
        self.load_dop = load_dop
        self.load_dsm = load_dsm
        self.load_lod = load_lod
        self.load_depth = load_depth
        self.load_normals = load_normals
        self.crop_multiplier_range = crop_multiplier_range
        self.ensure_coverage = ensure_coverage
        self.return_matching = return_matching
        self.crop_mode = crop_mode
        # Store user-provided values (may be None to use per-sequence defaults)
        self._user_min_gsd = min_gsd
        self._user_max_gsd = max_gsd
        self.sequence_length = sequence_length
        self.enable_noise = enable_noise
        self.enable_flip = enable_flip
        self.enable_rotation = enable_rotation
        self.preserve_dsm_gsd = preserve_dsm_gsd
        self.return_fullres_dop = return_fullres_dop
        self.return_fullres_dsm = return_fullres_dsm
        self.return_resized_dop = return_resized_dop
        self.return_resized_dsm = return_resized_dsm
        self.dop_year = dop_year

        self.samples = []

        # Performance Caches
        self._pose_cache = {}        # {seq_path: pd.DataFrame}
        self._intr_cache = {}        # {seq_path: K}
        self._lod_cache = {}         # {seq_path: (v, f, l, dop_proj)}
        self._lidar_cache = {}       # {seq_path: np.ndarray}
        self._pm_cache = {}          # {depth_path: (point_map, depth)}
        self._preprocessed_cache = {} # {seq_path: (dop_img, dop_meta, dsm_h, dsm_meta)}

        self._load_sequences()
        self._create_samples()

    # ------------------------------------------------------------------
    # Sequence loading and splits
    # ------------------------------------------------------------------

    # Default train/val/test split ratios when splits.json is not available
    DEFAULT_SPLIT_RATIOS = {'train': 0.7, 'val': 0.15, 'test': 0.15}

    def _load_sequences(self):
        """Load sequences filtered by split or sequences list. Supports splits.json auto-download."""
        seq_dir = self.dataset_dir / "scenes"
        seq_dir.mkdir(parents=True, exist_ok=True)
        
        # Download splits.json if not present
        splits_path = self.dataset_dir / "splits.json"
        if not splits_path.exists():
            print("[MovingDrone] Downloading splits.json...")
            try:
                urllib.request.urlretrieve("https://cvg.cit.tum.de/webshare/g/papers/Dhaouadi/MovingDrone/splits.json", splits_path)
            except Exception as e:
                print(f"[MovingDrone] Failed to download splits.json: {e}")
                
        if splits_path.exists():
            with open(splits_path, 'r') as f:
                splits_config = json.load(f)
        else:
            splits_config = {}

        if self._target_sequences is not None:
            split_names = self._target_sequences
        elif self.split == 'all':
            split_names = set()
            for v in splits_config.values():
                split_names.update(v)
            split_names = list(split_names)
        elif self.split == 'test' and 'test' not in splits_config:
            split_names = set(splits_config.get('test_inPlace', []))
            split_names |= set(splits_config.get('test_outPlace', []))
            split_names = list(split_names)
        else:
            split_names = splits_config.get(self.split, [])

        self.sequences = [seq_dir / s for s in split_names]
        
        if self.predownload:
            try:
                from tqdm import tqdm
                seq_iter = tqdm(self.sequences, desc="Pre-downloading sequences")
            except ImportError:
                print(f"[MovingDrone] Pre-downloading {len(self.sequences)} sequences...")
                seq_iter = self.sequences
                
            for seq_path in seq_iter:
                self._ensure_sequence_downloaded(seq_path.name)

    def _ensure_sequence_downloaded(self, seq_name):
        # Local layout uses scenes/<name>/; the academic server hosts the same
        # files under sequences/<name>/ (remote path name differs on purpose).
        seq_dir = self.dataset_dir / "scenes" / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)
        base_url = f"https://cvg.cit.tum.de/webshare/g/papers/Dhaouadi/MovingDrone/sequences/{seq_name}"
        
        def download_file(rel_path):
            local_path = seq_dir / rel_path
            if local_path.exists() and local_path.stat().st_size > 0:
                return True
            local_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"{base_url}/{rel_path}"
            try:
                urllib.request.urlretrieve(url, local_path)
                return True
            except Exception as e:
                if local_path.exists():
                    local_path.unlink()  # Remove empty/failed file
                return False

        # Basic files
        if not download_file("meta.json"):
            return
            
        with open(seq_dir / "meta.json", "r") as f:
            meta = json.load(f)
            
        download_file("poses.csv")
        download_file("intrinsics.json")
        download_file("video.mp4")
        
        # Geodata
        if self.load_dsm:
            download_file("dsm.npz")
        if self.load_lidar:
            download_file("lidar.npz")
        if self.load_lod:
            download_file("lod1.npz")
            download_file("lod2.npz")
            
        # DOP
        if self.load_dop:
            dops = meta.get("dops", {})
            for yr, entry in dops.items():
                if 'file' in entry:
                    download_file(entry['file'])
                else:
                    download_file(f"dop_{entry['year']}.jpg")
                    download_file("dop.jpg")
                    
        # Frames / Depth / Normals
        num_frames = meta.get("num_frames", 0)
        
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kwargs: x
        
        if not (seq_dir / "video.mp4").exists():
            for i in tqdm(range(num_frames), desc=f"{seq_name} frames", leave=False):
                download_file(f"rendering/frame_{i:04d}.jpg")
                
        if self.load_depth:
            for i in tqdm(range(num_frames), desc=f"{seq_name} depth", leave=False):
                download_file(f"depth/depth_{i:04d}.png")
                download_file(f"depth/depth_{i:04d}.json")
        
        if self.load_normals:
            for i in tqdm(range(num_frames), desc=f"{seq_name} normals", leave=False):
                download_file(f"normals/normal_{i:04d}.npz")

    def _generate_default_split(self, all_seqs, split):
        """Generate train/val/test split using deterministic hashing."""
        if not all_seqs:
            return []

        # Use deterministic approach: sort and split by ratio
        n = len(all_seqs)
        train_end = max(1, int(n * self.DEFAULT_SPLIT_RATIOS['train'])) if n > 0 else 0
        val_end = train_end + int(n * self.DEFAULT_SPLIT_RATIOS['val'])

        if split == 'train':
            return all_seqs[:train_end]
        elif split == 'val':
            return all_seqs[train_end:val_end]
        elif split == 'test':
            return all_seqs[val_end:]
        else:
            print(f"[MovingDrone] Unknown split '{split}', using all sequences")
            return all_seqs


    # ------------------------------------------------------------------
    # Video reading
    # ------------------------------------------------------------------
    # Priority:
    #   1. rendering/frame_{:04d}.jpg  — individual JPEG files (legacy, fast)
    #   2. PyAV (fast random seek on H.264 -bf 0 video)
    #   3. cv2.VideoCapture (sequential-optimised fallback)
    # ------------------------------------------------------------------

    def _get_video_cap(self, video_path):
        """Return a cached [cv2.VideoCapture, last_frame_id] for fallback use."""
        v_str = str(video_path)
        CACHE_LIMIT = 256
        if v_str in self._video_caps:
            val = self._video_caps.pop(v_str)
            self._video_caps[v_str] = val
            return val
        if len(self._video_caps) >= CACHE_LIMIT:
            old_vd, (old_cap, _) = next(iter(self._video_caps.items()))
            if old_cap: old_cap.release()
            del self._video_caps[old_vd]
        cap = cv2.VideoCapture(v_str)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self._video_caps[v_str] = [cap, -1]
        return self._video_caps[v_str]

    def _read_frame_pyav(self, video_path: str, frame_id: int) -> np.ndarray:
        """
        Decode a single frame from a H.264 video using PyAV.

        Works reliably on videos encoded with ``-bf 0`` (no B-frames).

        Returns:
            np.ndarray of shape (H, W, 3) uint8 RGB, or raises RuntimeError."""
        import av
        container = av.open(video_path)
        video_stream = container.streams.video[0]

        # Target presentation timestamp for frame_id
        fps = float(video_stream.average_rate)
        time_base = video_stream.time_base
        # PTS = frame_id / fps / time_base
        target_pts = int(frame_id / fps / time_base)

        # Seek to just before the target frame
        container.seek(target_pts, stream=video_stream)

        for packet in container.demux(video_stream):
            for decoded_frame in packet.decode():
                if decoded_frame.pts is None or decoded_frame.pts >= target_pts:
                    arr = decoded_frame.to_ndarray(format='rgb24')   # (H, W, 3) uint8
                    container.close()
                    return arr
        container.close()
        raise RuntimeError(f"PyAV: frame {frame_id} not found in {video_path}")

    def _get_video_frame(self, video_path, frame_id):
        """Read a specific frame. Tries rendering dir → PyAV → cv2 fallback."""
        video_path = Path(video_path)

        # --- Priority 1: pre-rendered JPEG files ---
        render_dir = video_path.parent / "rendering"
        if render_dir.exists():
            for ext in ('jpg', 'jpeg', 'png'):
                frame_path = render_dir / f"frame_{frame_id:04d}.{ext}"
                if frame_path.exists():
                    frame = cv2.imread(str(frame_path))
                    if frame is not None:
                        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Priority 2: PyAV (fast random seek) ---
        try:
            return self._read_frame_pyav(str(video_path), frame_id)
        except Exception:
            pass  # Fall through to cv2

        # --- Priority 3: cv2 fallback (sequential-optimised) ---
        cache_entry = self._get_video_cap(video_path)
        cap = cache_entry[0]
        last_fid = cache_entry[1]
        if frame_id == last_fid + 1:
            ret, frame = cap.read()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
        if ret:
            cache_entry[1] = frame_id
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raise RuntimeError(f"Failed to read frame {frame_id} from {video_path}")

    def _get_video_frames_range(self, video_path, frame_ids: list) -> list:
        """Read multiple frames from video."""
        return [self._get_video_frame(video_path, fid) for fid in frame_ids]

    def close(self):
        """Close all cached video captures."""
        for v_str in list(self._video_caps.keys()):
            cap, _ = self._video_caps.pop(v_str)
            if cap: cap.release()

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def _load_intrinsics(self, seq_path, meta):
        """Load camera intrinsics from intrinsics.json."""
        json_path = seq_path / "intrinsics.json"
        K = np.eye(3)
        if json_path.exists():
            with open(json_path, 'r') as f:
                d = json.load(f)
            K[0, 0] = d.get('fx', 0)
            K[1, 1] = d.get('fy', 0)
            K[0, 2] = d.get('cx', 0)
            K[1, 2] = d.get('cy', 0)
        return K

    # ------------------------------------------------------------------
    # Geodata loading
    # ------------------------------------------------------------------

    # Max sequences to cache in memory per worker.
    # Each decompressed DOP+DSM is ~280MB, so 8 sequences ≈ 2.2GB per worker.
    _GEODATA_CACHE_LIMIT = 8

    def _load_preprocessed_geodata(self, seq_path):
        """Load preprocessed DOP/DSM/LoD/LiDAR if available. LRU-bounded cache."""
        if seq_path in self._preprocessed_cache:
            # Move to end (LRU refresh)
            val = self._preprocessed_cache.pop(seq_path)
            self._preprocessed_cache[seq_path] = val
            return val

        # Evict oldest if cache is full
        while len(self._preprocessed_cache) >= self._GEODATA_CACHE_LIMIT:
            oldest_key = next(iter(self._preprocessed_cache))
            del self._preprocessed_cache[oldest_key]

        # Paths
        meta_path = seq_path / "meta.json"
        dsm_path = seq_path / "dsm.npz"

        if not (meta_path.exists() and dsm_path.exists()):
            return None

        try:
            # Load metadata
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            # Resolve DOP(s): discover all available years, then select based on self.dop_year
            # Each entry: (year_int, dop_path, dop_meta_dict)
            available_dops = []

            dops = meta.get('dops', {})
            if dops:
                for yr_str in sorted(dops.keys(), reverse=True):
                    entry = dops[yr_str]
                    candidate_path = seq_path / entry['file']
                    if candidate_path.exists():
                        available_dops.append((entry.get('year', int(yr_str.split('_')[-1])), candidate_path, entry))
                        continue
                    legacy_name = f"dop_{entry['year']}.jpg"
                    legacy_candidate = seq_path / legacy_name
                    if legacy_candidate.exists():
                        available_dops.append((entry['year'], legacy_candidate, entry))

            # Legacy fallback: dop key with dop.jpg
            if not available_dops and meta.get('dop'):
                legacy_path = seq_path / "dop.jpg"
                if legacy_path.exists():
                    dm = meta['dop']
                    available_dops.append((dm.get('year', 0), legacy_path, dm))

            # Fallback: scan dop/<year>.jpg files on disk
            # bounds in dsm.npz are already in utm_offset-local coordinates
            if not available_dops:
                dop_subdir = seq_path / 'dop'
                if dop_subdir.is_dir():
                    year_jpgs = sorted(dop_subdir.glob('*.jpg'), reverse=True)
                    for jpg in year_jpgs:
                        try:
                            yr = int(jpg.stem)
                        except ValueError:
                            continue
                        with np.load(dsm_path) as dsm_data:
                            raw_bounds = list(dsm_data['bounds'])
                            dm = {
                                'gsd': float(dsm_data['gsd']),
                                'bounds': raw_bounds,
                                'year': yr,
                                'file': f'dop/{jpg.name}',
                            }
                        available_dops.append((yr, jpg, dm))
                # Also check legacy dop_<year>.jpg in sequence root
                if not available_dops:
                    for jpg in sorted(seq_path.glob('dop_*.jpg'), reverse=True):
                        try:
                            yr = int(jpg.stem.split('_', 1)[1])
                        except (ValueError, IndexError):
                            continue
                        with np.load(dsm_path) as dsm_data:
                            raw_bounds = list(dsm_data['bounds'])
                            dm = {
                                'gsd': float(dsm_data['gsd']),
                                'bounds': raw_bounds,
                                'year': yr,
                                'file': jpg.name,
                            }
                        available_dops.append((yr, jpg, dm))

            if not available_dops:
                return None

            # Sort by year descending (most recent first)
            available_dops.sort(key=lambda x: x[0], reverse=True)

            # Select primary DOP based on self.dop_year
            dop_meta = None
            dop_path = None
            dop_all = None  # Only populated when dop_year='all'

            if isinstance(self.dop_year, int):
                # Specific year requested
                for yr, p, m in available_dops:
                    if yr == self.dop_year:
                        dop_path, dop_meta = p, m
                        break
                if dop_meta is None:
                    avail_yrs = [y for y, _, _ in available_dops]
                    print(f"Warning: DOP year {self.dop_year} not found in {seq_path}. "
                          f"Available: {avail_yrs}. Falling back to most recent.")
                    dop_path, dop_meta = available_dops[0][1], available_dops[0][2]
            elif self.dop_year == 'all':
                # Load ALL DOPs; primary = most recent
                dop_path, dop_meta = available_dops[0][1], available_dops[0][2]
                dop_all = {}
                for yr, p, m in available_dops:
                    img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                    dop_all[yr] = (img, m)
            else:
                # 'last' (default) — most recent
                dop_path, dop_meta = available_dops[0][1], available_dops[0][2]

            if dop_meta is None or dop_path is None:
                return None

            # Load DOP image
            dop_img = cv2.cvtColor(cv2.imread(str(dop_path)), cv2.COLOR_BGR2RGB)

            # Load DSM (bounds in npz are utm_offset-local coordinates)
            with np.load(dsm_path) as data:
                dsm_h = data['height']
                raw_b = np.array(data['bounds'], dtype=np.float64)
                dsm_meta = {
                    'bounds': list(raw_b),
                    'gsd': float(data['gsd'])
                }

            # Load LiDAR (optional)
            lidar_path = seq_path / "lidar.npz"
            lidar_data = None
            if lidar_path.exists():
                with np.load(lidar_path) as data:
                    lidar_data = {k: data[k] for k in data.files}

            # Load LoD (optional)
            lod_data = {}
            for lvl in [1, 2]:
                lod_p = seq_path / f"lod{lvl}.npz"
                if lod_p.exists():
                    with np.load(lod_p) as data:
                         lod_data[f"lod{lvl}"] = {k: data[k] for k in data.files}

            cache_entry = {
                'dop': (dop_img, dop_meta),
                'dop_all': dop_all,  # dict {year: (img, meta)} or None
                'dsm': (dsm_h, dsm_meta),
                'lidar': lidar_data,
                'lod': lod_data
            }
            self._preprocessed_cache[seq_path] = cache_entry
            return cache_entry

        except Exception as e:
            print(f"Warning: Failed to load preprocessed geodata for {seq_path}: {e}")
            return None

    def _get_max_sequence_length(self):
        sl = self.sequence_length
        if isinstance(sl, int):
            return sl
        elif isinstance(sl, tuple):
            return sl[1]
        elif isinstance(sl, list):
            return max(sl)
        return 1

    def _resolve_sequence_length(self, sample):
        sl = self.sequence_length
        if isinstance(sl, int):
            if sl < 0:
                return sample.get('max_length', 1)
            return sl
        elif isinstance(sl, tuple):
            min_len, max_len = sl
            available = sample.get('max_length', max_len)
            return random.randint(min_len, min(max_len, available))
        elif isinstance(sl, list):
            chosen = random.choice(sl)
            available = sample.get('max_length', chosen)
            return min(chosen, available)
        return 1

    def _create_samples(self):
        max_sl = self._get_max_sequence_length()
        mode = "single-frame" if max_sl == 1 else f"sequence (length up to {max_sl})"
        print(f"[MovingDrone] Loading {len(self.sequences)} sequences for split='{self.split}', mode={mode}, stride={self.stride}")
        for seq_path in self.sequences:
            if not self.predownload:
                self._ensure_sequence_downloaded(seq_path.name)
            try:
                with open(seq_path / "meta.json", 'r') as f:
                    meta = json.load(f)

                num_frames = meta['num_frames']
                has_geo = any(
                    (seq_path / entry['file']).exists() or
                    (seq_path / f"dop_{entry['year']}.jpg").exists()  # legacy flat path
                    for entry in meta.get('dops', {}).values()
                ) or (seq_path / "dop.jpg").exists()  # legacy fallback
                # Fallback: scan dop/<year>.jpg directory
                if not has_geo:
                    dop_subdir = seq_path / 'dop'
                    if dop_subdir.is_dir():
                        has_geo = any(
                            f.suffix == '.jpg' and f.stem.isdigit()
                            for f in dop_subdir.iterdir()
                        )
                has_geo = has_geo and (seq_path / "dsm.npz").exists()

                if max_sl == 1:
                    for i in range(0, num_frames, self.stride):
                        self.samples.append({
                            'seq_path': seq_path,
                            'frame_id': i,
                            'meta': meta,
                            'has_geo': has_geo,
                        })
                elif max_sl < 0:
                     # Full sequence mode: one sample per sequence covering the whole video
                     self.samples.append({
                            'seq_path': seq_path,
                            'start_frame': 0,
                            'max_length': num_frames,
                            'meta': meta,
                            'has_geo': has_geo,
                        })
                else:
                    min_sl = self.sequence_length if isinstance(self.sequence_length, int) else (
                        self.sequence_length[0] if isinstance(self.sequence_length, tuple) else min(self.sequence_length)
                    )
                    for i in range(0, num_frames - min_sl + 1, self.stride):
                        max_avail = num_frames - i
                        self.samples.append({
                            'seq_path': seq_path,
                            'start_frame': i,
                            'max_length': max_avail,
                            'meta': meta,
                            'has_geo': has_geo,
                        })
            except Exception as e:
                print(f"Error loading sequence {seq_path}: {e}")

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_depth_pointmap(self, seq_path, meta, frame_id, K, K_scaled, pose_xyz, seq_key):
        """Load depth NPZ and compute point map for a single frame."""
        x, y, z, qw, qx, qy, qz = pose_xyz
        point_map = torch.zeros((3, self.size[0], self.size[1]))
        image_depth = torch.zeros((1, self.size[0], self.size[1]))

        depth_path = seq_path / "depth" / f"depth_{frame_id:04d}.png"
        depth_json_path = seq_path / "depth" / f"depth_{frame_id:04d}.json"
        if depth_path in self._pm_cache:
            return self._pm_cache[depth_path]

        if self.load_depth and depth_path.exists() and depth_json_path.exists():
            try:
                depth_np = load_depth_png(depth_path, depth_json_path)   # (H, W) float32 ray-dist, format-agnostic
                H_d, W_d = depth_np.shape
                v_grid, u_grid = np.mgrid[0:H_d, 0:W_d]
                fx_raw, fy_raw = K[0, 0], K[1, 1]
                cx_raw, cy_raw = K[0, 2], K[1, 2]

                u_cam = (u_grid - cx_raw) / fx_raw
                v_cam = (v_grid - cy_raw) / fy_raw
                ray_norm = np.sqrt(u_cam**2 + v_cam**2 + 1)
                depth_planar = (depth_np / ray_norm).astype(np.float32)

                depth_tensor = torch.from_numpy(depth_planar).unsqueeze(0)
                image_depth = TF.resize(depth_tensor, self.size, interpolation=TF.InterpolationMode.NEAREST)
                depth_map_resized = image_depth[0].numpy()

                # render_origin: offset between depth rendering origin and local coord origin
                render_origin = np.array(meta.get('render_origin', [0, 0, 0]))

                local_x = x - render_origin[0]
                local_y = y - render_origin[1]
                local_z = z - render_origin[2]
                w2c_local = quat_pos_to_w2c(local_x, local_y, local_z, qw, qx, qy, qz)
                pm_local = unproject_depth_to_world(depth_map_resized, K_scaled, w2c_local)

                point_map = torch.from_numpy(pm_local)
                for c in range(3):
                    point_map[c] += render_origin[c]
                
                # Cache successful load
                if len(self._pm_cache) > 200: # Simple LRU-ish
                    self._pm_cache.pop(next(iter(self._pm_cache)))
                self._pm_cache[depth_path] = (point_map, image_depth)

            except Exception as e:
                print(f"Point map from depth failed: {e}")

        return point_map, image_depth

    def _load_normals(self, seq_path, frame_id):
        """Load or synthesise surface normals for a single frame.

        Strategy (in priority order):
          1. Pre-rendered file: ``normals/normal_{frame_id:04d}.npz`` — loaded directly.
          2. Depth-based synthesis: normals computed on-the-fly from the depth map using
             cross-product back-projection (utils.depth.depth_to_normals_np).
             Validated accuracy: mean angular error ~0.5°, 99% of pixels < 5°.
          3. Zero tensor fallback if neither file is available.

        Open3D primitive_normals are stored OUTWARD-facing (away from camera).
        The synthesised normals are returned in the same convention (negated, outward)
        so callers see a consistent sign regardless of which path was taken."""
        normals = torch.zeros((3, self.size[0], self.size[1]))

        if not self.load_normals:
            return normals

        # --- Path 1: pre-rendered file ---
        normal_path = seq_path / "normals" / f"normal_{frame_id:04d}.npz"
        if normal_path.exists():
            try:
                normals_np = np.load(normal_path)['normals']  # (H, W, 3)
                normals_tensor = torch.from_numpy(normals_np).permute(2, 0, 1)  # (3, H, W)
                normals = TF.resize(normals_tensor, self.size, interpolation=TF.InterpolationMode.NEAREST)
                return normals
            except Exception as e:
                print(f"[MovingDrone] Normals file load failed ({normal_path.name}): {e}")

        # --- Path 2: synthesise from depth ---
        depth_path = seq_path / "depth" / f"depth_{frame_id:04d}.png"
        depth_json_path = seq_path / "depth" / f"depth_{frame_id:04d}.json"
        if depth_path.exists() and depth_json_path.exists():
            try:
                # Load per-sequence intrinsics (cached in _intr_cache)
                K = self._intr_cache.get(seq_path)
                if K is None:
                    K = self._load_intrinsics(seq_path, {})
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

                t_hit = load_depth_png(depth_path, depth_json_path)  # (H_orig, W_orig) ray-distance, format-agnostic

                # Compute camera-space normals (pointing toward camera)
                # Negate to match Open3D outward-facing convention
                normals_np = -depth_to_normals_np(t_hit, fx, fy, cx, cy)  # (H, W, 3)

                normals_tensor = torch.from_numpy(normals_np).permute(2, 0, 1)  # (3, H, W)
                normals = TF.resize(normals_tensor, self.size, interpolation=TF.InterpolationMode.NEAREST)
            except Exception as e:
                print(f"[MovingDrone] Normals synthesis from depth failed (frame {frame_id}): {e}")

        return normals

    def _project_to_uav(self, points_3d, w2c, K_scaled, depth_map=None, tolerance=20.0, max_points=500000):
        """Project 3D world points to UAV image coordinates with optional occlusion check.
        Returns (N, 3) array of (u, v, z_cam) for visible points.
        
        Args:
            max_points: If > 0 and len(points_3d) exceeds this, randomly subsample
                        before projection for speed (default 500K)."""
        if points_3d.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # Subsample large point clouds for speed (uniform stride, O(1))
        if max_points > 0 and points_3d.shape[0] > max_points:
            step = max(1, points_3d.shape[0] // max_points)
            points_3d = points_3d[::step]

        ones = np.ones((points_3d.shape[0], 1))
        pts_h = np.concatenate([points_3d, ones], axis=1)
        pts_cam = (w2c @ pts_h.T).T
        pts_cam3 = pts_cam[:, :3]

        valid_z = pts_cam3[:, 2] > 0.1
        pts_cam3 = pts_cam3[valid_z]
        if pts_cam3.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)

        pts_proj = (K_scaled @ pts_cam3.T).T
        u = pts_proj[:, 0] / (pts_proj[:, 2] + 1e-8)
        v = pts_proj[:, 1] / (pts_proj[:, 2] + 1e-8)

        H, W = self.size
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        if not in_img.any():
            return np.zeros((0, 3), dtype=np.float32)

        u_in, v_in, z_in = u[in_img], v[in_img], pts_cam3[in_img, 2]

        if depth_map is not None:
            u_idx = np.clip(u_in.astype(int), 0, W - 1)
            v_idx = np.clip(v_in.astype(int), 0, H - 1)
            ref_depth = depth_map[0].numpy()[v_idx, u_idx]
            visible = (ref_depth == 0) | (z_in < (ref_depth + tolerance))
            if not visible.any():
                return np.zeros((0, 3), dtype=np.float32)
            u_in, v_in, z_in = u_in[visible], v_in[visible], z_in[visible]

        return np.stack([u_in, v_in, z_in], axis=1).astype(np.float32)

    def _compute_correspondence_maps(self, data):
        """Compute dense correspondence maps between query and DOP views.
        Modifies data dict in-place, adding query_in_dop, dop_in_query and masks."""
        H, W = self.size
        crop_bounds = data.get('crop_bounds')

        pm = data.get('point_map')
        mask_query_in_dop = torch.zeros((1, H, W))
        query_in_dop = torch.zeros((2, H, W))
        grid_q2d = torch.zeros((1, H, W, 2))
        mask_query_in_dop_candidate = torch.zeros((1, H, W))

        if pm is not None and crop_bounds is not None:
            c_min_x, c_min_y, c_max_x, c_max_y = crop_bounds
            u_dop = (pm[0] - c_min_x) / (c_max_x - c_min_x) * 2.0 - 1.0
            v_dop = (c_max_y - pm[1]) / (c_max_y - c_min_y) * 2.0 - 1.0
            query_in_dop = torch.stack([u_dop, v_dop], dim=0)
            grid_q2d = query_in_dop.permute(1, 2, 0).unsqueeze(0)

            within_dop = (u_dop >= -1) & (u_dop <= 1) & (v_dop >= -1) & (v_dop <= 1)
            dop_mask_sampled = torch.nn.functional.grid_sample(
                data['geodata_mask'].unsqueeze(0), grid_q2d, mode='nearest', align_corners=False
            ).squeeze(0)

            dsm_h_raw = data['dsm'][2].unsqueeze(0).unsqueeze(0)
            dsm_pooled = torch.nn.functional.max_pool2d(dsm_h_raw, kernel_size=3, stride=1, padding=1)
            dsm_sampled = torch.nn.functional.grid_sample(
                dsm_pooled, grid_q2d, mode='nearest', align_corners=False
            ).squeeze()
            visible_in_dop = (dsm_sampled <= pm[2] + 10.0)

            mask_query_in_dop = data['query_mask'] * within_dop.float().unsqueeze(0) * dop_mask_sampled * visible_in_dop.float().unsqueeze(0)
            mask_query_in_dop_candidate = data['query_mask'] * within_dop.float().unsqueeze(0) * dop_mask_sampled

        dsm = data.get('dsm')
        mask_dop_in_query = torch.zeros((1, H, W))
        dop_in_query = torch.zeros((2, H, W))
        grid_d2q = torch.zeros((1, H, W, 2))
        mask_dop_in_query_candidate = torch.zeros((1, H, W))

        if dsm is not None:
            # DSM may have different spatial dims when preserve_dsm_gsd=True
            H_d, W_d = dsm.shape[1], dsm.shape[2]
            ones_d = torch.ones((1, H_d, W_d))
            dsm_h = torch.cat([dsm, ones_d], dim=0).view(4, -1)
            pts_cam = data['extrinsics'] @ dsm_h
            pts_cam3 = pts_cam[:3, :]
            pts_proj = data['intrinsics'] @ pts_cam3

            u_uav = (pts_proj[0] / (pts_proj[2] + 1e-8) / (W - 1)) * 2.0 - 1.0
            v_uav = (pts_proj[1] / (pts_proj[2] + 1e-8) / (H - 1)) * 2.0 - 1.0
            dop_in_query = torch.stack([u_uav, v_uav], dim=0).view(2, H_d, W_d)
            grid_d2q = dop_in_query.permute(1, 2, 0).unsqueeze(0)

            within_uav = (u_uav >= -1) & (u_uav <= 1) & (v_uav >= -1) & (v_uav <= 1)
            within_uav = within_uav.view(H_d, W_d)
            z_in_cam = pts_cam3[2, :].view(H_d, W_d)
            uav_depth_sampled = torch.nn.functional.grid_sample(
                data['depth'].unsqueeze(0), grid_d2q, mode='nearest', align_corners=False
            ).squeeze(0).squeeze(0)
            is_visible = (uav_depth_sampled == 0) | (z_in_cam < (uav_depth_sampled + 10.0))

            # geodata_mask is at self.size; resize to DSM dims if needed for masking
            geo_mask_d = data['geodata_mask']
            if (H_d, W_d) != (H, W):
                geo_mask_d = TF.resize(geo_mask_d, (H_d, W_d),
                                       interpolation=TF.InterpolationMode.NEAREST)
            mask_dop_in_query = geo_mask_d * within_uav.float().unsqueeze(0) * is_visible.float().unsqueeze(0)
            mask_dop_in_query_candidate = geo_mask_d * within_uav.float().unsqueeze(0)

            # Resize dop_in_query and masks back to (H, W) for consistent output
            if (H_d, W_d) != (H, W):
                dop_in_query = TF.resize(dop_in_query, (H, W),
                                         interpolation=TF.InterpolationMode.BILINEAR)
                grid_d2q = dop_in_query.permute(1, 2, 0).unsqueeze(0)
                mask_dop_in_query = TF.resize(mask_dop_in_query, (H, W),
                                              interpolation=TF.InterpolationMode.NEAREST)
                mask_dop_in_query_candidate = TF.resize(mask_dop_in_query_candidate, (H, W),
                                                        interpolation=TF.InterpolationMode.NEAREST)

        # Bidirectional Consistency
        q_back_in_q = torch.nn.functional.grid_sample(
            dop_in_query.unsqueeze(0), grid_q2d, mode='bilinear', align_corners=False
        ).squeeze(0)
        uav_grid_ident = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij'
        ), dim=0)[[1, 0]]
        dist_q = torch.norm(q_back_in_q - uav_grid_ident, dim=0)
        consistent_q = (dist_q < 0.5)

        d_back_in_d = torch.nn.functional.grid_sample(
            query_in_dop.unsqueeze(0), grid_d2q, mode='bilinear', align_corners=False
        ).squeeze(0)
        dop_grid_ident = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij'
        ), dim=0)[[1, 0]]
        dist_d = torch.norm(d_back_in_d - dop_grid_ident, dim=0)
        consistent_d = (dist_d < 0.5)

        m_d_sampled_in_q = torch.nn.functional.grid_sample(
            mask_dop_in_query.unsqueeze(0), grid_q2d, mode='nearest', align_corners=False
        ).squeeze(0)
        m_q_sampled_in_d = torch.nn.functional.grid_sample(
            mask_query_in_dop.unsqueeze(0), grid_d2q, mode='nearest', align_corners=False
        ).squeeze(0)

        mask_query_in_dop_mutual = mask_query_in_dop * m_d_sampled_in_q * consistent_q.float().unsqueeze(0)
        mask_dop_in_query_mutual = mask_dop_in_query * m_q_sampled_in_d * consistent_d.float().unsqueeze(0)

        data['query_in_dop'] = query_in_dop
        data['mask_query_in_dop'] = (mask_query_in_dop_mutual > 0.5).float()
        data['mask_query_in_dop_oneway'] = (mask_query_in_dop_candidate > 0.5).float()
        data['dop_in_query'] = dop_in_query
        data['mask_dop_in_query'] = (mask_dop_in_query_mutual > 0.5).float()
        data['mask_dop_in_query_oneway'] = (mask_dop_in_query_candidate > 0.5).float()

    # ------------------------------------------------------------------
    # Main __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq_path = sample['seq_path']
        meta = sample['meta']
        max_sl = self._get_max_sequence_length()

        # ============================================================
        # Phase 1: Resolve frame IDs
        # ============================================================
        if max_sl == 1:
            frame_ids = [sample['frame_id']]
        else:
            L = self._resolve_sequence_length(sample)
            start = sample['start_frame']
            frame_ids = list(range(start, start + L))
        L = len(frame_ids)

        # ============================================================
        # Phase 2: Load poses and intrinsics (intrinsics are shared)
        # ============================================================
        if seq_path not in self._pose_cache:
            self._pose_cache[seq_path] = pd.read_csv(seq_path / "poses.csv")
        df_poses = self._pose_cache[seq_path]

        if seq_path not in self._intr_cache:
            self._intr_cache[seq_path] = self._load_intrinsics(seq_path, meta)
        K = self._intr_cache[seq_path]

        sx = self.size[1] / meta['width']
        sy = self.size[0] / meta['height']
        K_scaled = K.copy()
        K_scaled[0, 0] *= sx; K_scaled[0, 2] *= sx
        K_scaled[1, 1] *= sy; K_scaled[1, 2] *= sy

        poses_w2c = []
        poses_xyz = []  # (x, y, z, qw, qx, qy, qz) per frame
        for fid in frame_ids:
            row = df_poses.iloc[fid]
            x, y, z = row['x'], row['y'], row['z']
            qw, qx, qy, qz = row['qw'], row['qx'], row['qy'], row['qz']
            poses_w2c.append(quat_pos_to_w2c(x, y, z, qw, qx, qy, qz))
            poses_xyz.append((x, y, z, qw, qx, qy, qz))

        # ============================================================
        # Phase 3: Load depth / point maps / normals for ALL frames
        # ============================================================
        seq_key = str(seq_path.name)
        point_maps = []
        depths = []
        normals_list = []
        for t, fid in enumerate(frame_ids):
            pm, dep = self._load_depth_pointmap(
                seq_path, meta, fid, K, K_scaled, poses_xyz[t], seq_key
            )
            point_maps.append(pm)
            depths.append(dep)
            if self.load_normals:
                normals_list.append(self._load_normals(seq_path, fid))

        # ============================================================
        # Phase 4: Compute UNION crop bounds across all frames
        # ============================================================
        image_dop = torch.zeros((3, self.size[0], self.size[1]))
        image_dsm_xyz = torch.zeros((3, self.size[0], self.size[1]))
        fullres_dop = None   # Will hold (3, H_native, W_native) tensor if requested
        fullres_dsm = None   # Will hold (3, H_native, W_native) XYZ tensor if requested
        preprocessed = None  # Will hold preprocessed geodata dict if available
        crop_bounds = None
        scale = 1.0
        native_gsd = 0.2  # Default; overridden from preprocessed metadata
        crop_extensions = (0.0, 0.0, 0.0, 0.0)
        resolved_dop_year = None  # Actual DOP year used (set in Phase 5)

        if sample['has_geo']:
            # --- Compute point_map bounds (used for center and optionally for size) ---
            all_valid_x, all_valid_y = [], []
            pm_min_x = pm_min_y = float('inf')
            pm_max_x = pm_max_y = float('-inf')
            for pm in point_maps:
                valid = (pm[2] != 0) & (pm[2] > -1000)
                if valid.any():
                    all_valid_x.append(pm[0][valid])
                    all_valid_y.append(pm[1][valid])
                    pm_min_x = min(pm_min_x, float(pm[0][valid].min()))
                    pm_max_x = max(pm_max_x, float(pm[0][valid].max()))
                    pm_min_y = min(pm_min_y, float(pm[1][valid].min()))
                    pm_max_y = max(pm_max_y, float(pm[1][valid].max()))

            if all_valid_x:
                combined_x = torch.cat(all_valid_x)
                combined_y = torch.cat(all_valid_y)
                center_x = float(torch.median(combined_x))
                center_y = float(torch.median(combined_y))
                has_pm_bounds = (pm_min_x != float('inf'))
            else:
                # Fallback: centroid of all camera positions
                cx_sum = sum(p[0] for p in poses_xyz)
                cy_sum = sum(p[1] for p in poses_xyz)
                center_x, center_y = cx_sum / L, cy_sum / L
                has_pm_bounds = False

            # --- Compute footprint bounds (used when crop_mode='footprint') ---
            union_min_x = union_min_y = float('inf')
            union_max_x = union_max_y = float('-inf')
            for w2c in poses_w2c:
                fmin_x, fmin_y, fmax_x, fmax_y = get_visible_footprint(
                    w2c, K, meta['height'], meta['width'], plane_z=35.0
                )
                if not np.isnan(fmin_x):
                    union_min_x = min(union_min_x, fmin_x)
                    union_min_y = min(union_min_y, fmin_y)
                    union_max_x = max(union_max_x, fmax_x)
                    union_max_y = max(union_max_y, fmax_y)
            has_fp_bounds = (union_min_x != float('inf'))

            # --- Choose bounds based on crop_mode ---
            if self.crop_mode == 'pointmap' and has_pm_bounds:
                # Use point_map bounds: tighter crops matching what UAV actually sees
                metric_w_base = pm_max_x - pm_min_x
                metric_h_base = pm_max_y - pm_min_y
                # Ensure minimum size (at least 20m in each direction)
                metric_w_base = max(metric_w_base, 20.0)
                metric_h_base = max(metric_h_base, 20.0)
            elif self.ensure_coverage and has_fp_bounds:
                # Use footprint bounds: larger context
                metric_w_base = union_max_x - union_min_x
                metric_h_base = union_max_y - union_min_y
            else:
                # Fallback: default size
                metric_w_base = self.size[1] * 0.2
                metric_h_base = self.size[0] * 0.2

            mult = random.uniform(*self.crop_multiplier_range)
            metric_w = metric_w_base * mult
            metric_h = metric_h_base * mult

            scale = max(metric_w / self.size[1], metric_h / self.size[0])
            
            # Resolve GSD limits: user override > class defaults
            min_gsd = self._user_min_gsd if self._user_min_gsd is not None else self.DEFAULT_MIN_GSD
            max_gsd = self._user_max_gsd if self._user_max_gsd is not None else self.DEFAULT_MAX_GSD
            
            scale = max(scale, min_gsd)  # Floor: prevent too high resolution
            scale = min(scale, max_gsd)  # Ceiling: prevent too low resolution
            metric_w = scale * self.size[1]
            metric_h = scale * self.size[0]

            half_w = metric_w / 2
            half_h = metric_h / 2
            c_min_x, c_max_x = center_x - half_w, center_x + half_w
            c_min_y, c_max_y = center_y - half_h, center_y + half_h
            crop_bounds = (c_min_x, c_min_y, c_max_x, c_max_y)

            # Crop extensions relative to union footprint (for metadata)
            if has_fp_bounds:
                ext_l = union_min_x - c_min_x
                ext_r = c_max_x - union_max_x
                ext_b = union_min_y - c_min_y
                ext_t = c_max_y - union_max_y
            else:
                ext_l = ext_r = (half_w - self.size[1] * 0.2 / 2)
                ext_b = ext_t = (half_h - self.size[0] * 0.2 / 2)
            crop_extensions = (float(ext_l), float(ext_b), float(ext_r), float(ext_t))

            # ========================================================
            # Phase 5: Geodata Loading (Preprocessed Cache or GeoTIFF)
            # ========================================================
            preprocessed = self._load_preprocessed_geodata(seq_path)

            native_gsd = 0.2
            resolved_dop_year = None  # Will be set to the actual year used
            if preprocessed:
                # For dop_year='all', randomly pick from available years
                if self.dop_year == 'all' and preprocessed.get('dop_all'):
                    yr = random.choice(list(preprocessed['dop_all'].keys()))
                    dop_img, dop_meta = preprocessed['dop_all'][yr]
                    resolved_dop_year = yr
                else:
                    dop_img, dop_meta = preprocessed['dop']
                    resolved_dop_year = dop_meta.get('year')
                # Use preprocessed global arrays (Fast)
                dsm_h, dsm_meta = preprocessed['dsm']
                
                # Assume DOP and DSM have same GSD in preprocessing
                native_gsd = dop_meta['gsd']
                
                px_w = int(round(metric_w / native_gsd))
                px_h = int(round(metric_h / native_gsd))
                
                # Global bounds
                g_min_x, g_min_y, g_max_x, g_max_y = dop_meta['bounds']
                
                # Crop logic (safe with padding)
                # Map UTM to pixel coords in global image (North-Up)
                # (0,0) is (min_x, max_y)
                col_c = (center_x - g_min_x) / native_gsd
                row_c = (g_max_y - center_y) / native_gsd
                
                col_start = int(round(col_c - px_w / 2))
                row_start = int(round(row_c - px_h / 2))
                col_end = col_start + px_w
                row_end = row_start + px_h
                
                # Valid Intersection
                full_h, full_w = dop_img.shape[:2]
                v_col_start = max(0, col_start)
                v_row_start = max(0, row_start)
                v_col_end = min(full_w, col_end)
                v_row_end = min(full_h, row_end)
                
                v_w = v_col_end - v_col_start
                v_h = v_row_end - v_row_start
                
                # Load DOP
                if self.load_dop or self.return_fullres_dop:
                    image_dop_np = np.zeros((px_h, px_w, 3), dtype=np.uint8)
                    if v_w > 0 and v_h > 0:
                        b_col = v_col_start - col_start
                        b_row = v_row_start - row_start
                        image_dop_np[b_row:b_row+v_h, b_col:b_col+v_w] = dop_img[v_row_start:v_row_end, v_col_start:v_col_end]
                    
                    # Full-res DOP: native crop without resize
                    if self.return_fullres_dop:
                        fullres_dop = TF.to_tensor(image_dop_np)  # (3, px_h, px_w)
                    
                    # Resized DOP for matcher
                    if self.load_dop and self.return_resized_dop:
                        image_dop = TF.to_tensor(image_dop_np)
                        image_dop = TF.resize(image_dop, self.size, interpolation=TF.InterpolationMode.BILINEAR)
                    elif self.load_dop:
                        # load_dop=True but return_resized_dop=False: keep zero tensor
                        pass

                # Load DSM
                if self.load_dsm or self.return_fullres_dsm:
                    merged_dsm = np.full((px_h, px_w), -9999.0, dtype=np.float32)
                    if v_w > 0 and v_h > 0:
                        b_col = v_col_start - col_start
                        b_row = v_row_start - row_start
                        merged_dsm[b_row:b_row+v_h, b_col:b_col+v_w] = dsm_h[v_row_start:v_row_end, v_col_start:v_col_end]
                    
                    crop_bounds = (c_min_x, c_min_y, c_max_x, c_max_y)

                    # Full-res DSM: native crop as XYZ without resize
                    if self.return_fullres_dsm:
                        fullres_dsm = torch.from_numpy(
                            dsm_to_xyz(merged_dsm, crop_bounds, (px_h, px_w))
                        )

                    # Resized DSM for model input
                    if self.load_dsm and self.return_resized_dsm:
                        # When preserve_dsm_gsd is set, keep DSM at native resolution
                        # (crop + pad only, no resize). This avoids interpolation artifacts
                        # at building edges and preserves exact GSD for precise PnP.
                        # The model's warp_to_3d uses normalized grid_sample so DSM can
                        # have different spatial dims than DOP.
                        dsm_target_size = (px_h, px_w) if self.preserve_dsm_gsd else self.size
                        image_dsm_xyz = torch.from_numpy(
                            dsm_to_xyz(merged_dsm, crop_bounds, dsm_target_size)
                        )
                    elif self.load_dsm:
                        # load_dsm=True but return_resized_dsm=False: keep zero tensor
                        pass
                
                found_bounds = crop_bounds

            else:
                # No preprocessed geodata available
                if sample['has_geo']:
                    print(f"Warning: Preprocessed geodata missing for {seq_path}. Expected dop_{{year}}.jpg, dsm.npz, etc.")
                found_bounds = crop_bounds

            if found_bounds is not None:
                crop_bounds = found_bounds
                scale = (crop_bounds[2] - crop_bounds[0]) / self.size[1]

        # ============================================================
        # Phase 6: Load shared LoD mesh (single reference)
        # ============================================================
        # Preprocessed LoD/LiDAR npz files are stored in local coords
        # (with utm_offset already subtracted), same as poses and crop_bounds.

        lod_vertices = np.zeros((0, 3), dtype=np.float32)
        lod_faces = np.zeros((0, 3), dtype=np.int32)
        lod_labels = np.array([])
        lod_dop = np.zeros((0, 3), dtype=np.float32)

        if self.load_lod:
            if seq_path in self._lod_cache:
                lod_vertices, lod_faces, lod_labels, lod_dop = self._lod_cache[seq_path]
            else:
                # Actual loading (only if not cached)
                # Preprocessed LoD (Priority 1)
                lod_key = f"lod{self.lod_level}"
                if preprocessed and 'lod' in preprocessed and lod_key in preprocessed['lod']:
                    lod_npz = preprocessed['lod'][lod_key]
                    lod_vertices = lod_npz['vertices'].astype(np.float32)
                    lod_faces = lod_npz['faces']
                    if 'labels' in lod_npz:
                        lod_labels = lod_npz['labels']
                    else:
                        lod_labels = np.zeros(lod_faces.shape[0], dtype=np.int32)
                else:
                    # No fallback to OBJ/CityGML (as requested)
                    if sample['has_geo']:
                         print(f"Warning: Preprocessed LoD missing for {seq_path}. Expected lod1.npz/lod2.npz.")

                if lod_faces.shape[0] > 0 and (lod_labels is None or len(lod_labels) == 0):
                    lod_labels = np.array(['building'] * len(lod_faces))

                # Single DOP projection (shared crop) - temporary for caching
                if lod_vertices.shape[0] > 0 and crop_bounds is not None:
                    cb_min_x, cb_min_y, cb_max_x, cb_max_y = crop_bounds
                    du = (lod_vertices[:, 0] - cb_min_x) / (cb_max_x - cb_min_x) * self.size[1]
                    dv = (cb_max_y - lod_vertices[:, 1]) / (cb_max_y - cb_min_y) * self.size[0]
                    lod_dop = np.stack([du, dv, lod_vertices[:, 2]], axis=1).astype(np.float32)

                # Store in cache
                self._lod_cache[seq_path] = (lod_vertices, lod_faces, lod_labels, lod_dop)

        # ============================================================
        # Phase 7: Load shared LiDAR 3D points
        # ============================================================
        lidar_world = np.zeros((0, 3), dtype=np.float32)  # 3D points in world coords

        if self.load_lidar and sample['has_geo']:
            if seq_path in self._lidar_cache:
                lidar_world = self._lidar_cache[seq_path]
            elif preprocessed and 'lidar' in preprocessed and preprocessed['lidar'] is not None:
                lidar_npz = preprocessed['lidar']
                lidar_world = lidar_npz['points'].astype(np.float32)
                self._lidar_cache[seq_path] = lidar_world
            else:
                # No fallback to LAS tiles (as requested)
                if sample['has_geo']:
                    print(f"Warning: Preprocessed LiDAR missing for {seq_path}. Expected lidar.npz.")
                lidar_world = np.zeros((0, 3), dtype=np.float32)

        # ============================================================
        # Phase 8: Load query images (batch read)
        # ============================================================
        video_path = seq_path / "video.mp4"
        images_np = self._get_video_frames_range(video_path, frame_ids)
        for t in range(L):
            if images_np[t] is None:
                images_np[t] = np.zeros((meta['height'], meta['width'], 3), dtype=np.uint8)

        # ============================================================
        # Phase 9: Build per-frame dicts (with cloned shared data for augmentation)
        # ============================================================
        # Shared spatial tensors (will be cloned into each frame dict for augmentation safety)
        shared_dsm = torch.nan_to_num(image_dsm_xyz, nan=0.0) if self.load_dsm else None
        # Shared LoD projection onto DOP (needs cloning when augmenting to avoid in-place modification issues)
        shared_lod_dop = torch.from_numpy(lod_dop).float() if lod_vertices.shape[0] > 0 else None

        # Geodata mask (shared) — must be at self.size for query-space operations
        if self.load_dop and self.load_dsm and shared_dsm is not None:
            if self.preserve_dsm_gsd and shared_dsm.shape[1:] != image_dop.shape[1:]:
                # DSM at native resolution — resize its mask to DOP size for combination
                dsm_mask_native = (shared_dsm != 0).any(dim=0, keepdim=True).float()
                dsm_mask_resized = TF.resize(dsm_mask_native, self.size,
                                             interpolation=TF.InterpolationMode.NEAREST)
                geodata_mask = ((image_dop != 0).any(dim=0, keepdim=True).float()
                               * dsm_mask_resized)
            else:
                geodata_mask = ((image_dop != 0).any(dim=0, keepdim=True).float()
                               * (shared_dsm != 0).any(dim=0, keepdim=True).float())
        elif self.load_dop:
            geodata_mask = (image_dop != 0).any(dim=0, keepdim=True).float()
        elif self.load_dsm:
            if self.preserve_dsm_gsd:
                # DSM at native resolution — resize mask to self.size
                dsm_mask_native = (shared_dsm != 0).any(dim=0, keepdim=True).float()
                geodata_mask = TF.resize(dsm_mask_native, self.size,
                                         interpolation=TF.InterpolationMode.NEAREST)
            else:
                geodata_mask = (shared_dsm != 0).any(dim=0, keepdim=True).float()
        else:
            geodata_mask = torch.ones((1, self.size[0], self.size[1]))

        frame_dicts = []
        for t, fid in enumerate(frame_ids):
            query_img = TF.to_tensor(images_np[t])
            query_img = TF.resize(query_img, self.size)
            w2c = poses_w2c[t]

            # Per-frame point_map and depth
            pm = point_maps[t]
            dep = depths[t]

            # Per-frame query mask
            if self.load_depth:
                pm_clean = torch.nan_to_num(pm, nan=0.0)
                query_mask = (pm_clean != 0).any(dim=0, keepdim=True).float()
            else:
                pm_clean = pm
                query_mask = torch.ones((1, self.size[0], self.size[1]))

            # Generate per-frame matches
            matches = []
            if self.load_dsm and shared_dsm is not None and crop_bounds is not None:
                H_dsm, W_dsm = shared_dsm.shape[1], shared_dsm.shape[2]
                dsm_flat = shared_dsm.view(3, -1).numpy()
                ones = np.ones((1, dsm_flat.shape[1]))
                xyz_h = np.concatenate([dsm_flat, ones], axis=0)
                xyz_cam = w2c @ xyz_h
                pts_proj = K_scaled @ xyz_cam[:3, :]
                us = pts_proj[0] / (pts_proj[2] + 1e-8)
                vs = pts_proj[1] / (pts_proj[2] + 1e-8)
                zs = xyz_cam[2]
                dsm_zs = dsm_flat[2, :]

                H_t, W_t = self.size
                inside = (us >= 0) & (us < W_t) & (vs >= 0) & (vs < H_t)
                valid_z = (zs > 0)
                valid_dsm = (dsm_zs > -1000) & (dsm_zs != 0)
                valid_idx = np.where(inside & valid_z & valid_dsm)[0]
                if len(valid_idx) > 0:
                    choice = np.random.choice(valid_idx, min(len(valid_idx), 5), replace=False)
                    for ci in choice:
                        matches.append({
                            'uav': (float(us[ci]), float(vs[ci])),
                            'dop': (float(ci % W_dsm), float(ci // W_dsm))
                        })

            # Per-frame LoD UAV projection (must project ALL vertices to preserve face indices)
            lod_uav = np.zeros((0, 3), dtype=np.float32)
            if self.load_lod and lod_vertices.shape[0] > 0:
                ones = np.ones((lod_vertices.shape[0], 1))
                pts_h = np.concatenate([lod_vertices, ones], axis=1)
                pts_cam = (w2c @ pts_h.T).T
                pts_cam3 = pts_cam[:, :3]
                pts_proj = (K_scaled @ pts_cam3.T).T
                u = pts_proj[:, 0] / (pts_proj[:, 2] + 1e-8)
                v = pts_proj[:, 1] / (pts_proj[:, 2] + 1e-8)
                lod_uav = np.stack([u, v, pts_cam3[:, 2]], axis=1).astype(np.float32)

            # Per-frame LiDAR UAV projection
            lidar_uav = np.zeros((0, 3), dtype=np.float32)
            if self.load_lidar and lidar_world.shape[0] > 0:
                lidar_uav = self._project_to_uav(lidar_world, w2c, K_scaled, depth_map=dep)

            # Build frame dict — clone shared spatial data so augment_data can flip safely
            fd = {
                'sample_id': f"{seq_path.name}_{fid:04d}",
                # Per-frame
                'query': query_img,
                'extrinsics': torch.from_numpy(w2c[:3, :]).float(),
                'extrinsics_raw': torch.from_numpy(w2c[:3, :]).float(),
                'query_mask': query_mask,
                # Shared (cloned for augmentation safety)
                'dop': image_dop.clone() if self.augment else image_dop,
                'geodata_mask': geodata_mask.clone() if self.augment else geodata_mask,
                'intrinsics': torch.from_numpy(K_scaled).float(),
                'intrinsics_raw': torch.from_numpy(K).float(),
                # Non-spatial metadata
                'lod_level': self.lod_level,
                'scale': float(scale),
                'gsd': float(scale),
                'crop_extensions': crop_extensions,
                'crop_bounds': crop_bounds,
                # Per-frame projections
                'lod_uav': torch.from_numpy(lod_uav).float(),
                'lidar_uav': torch.from_numpy(lidar_uav).float(),
                # Shared mesh data (lod_dop cloned when augmenting to prevent in-place modification issues)
                'lod_vertices': lod_vertices,
                'lod_faces': lod_faces,
                'lod_labels': lod_labels,
                'lod_dop': shared_lod_dop.clone() if (self.augment and shared_lod_dop is not None) else (shared_lod_dop if shared_lod_dop is not None else lod_dop),
            }

            if self.load_depth:
                fd['depth'] = dep
                fd['point_map'] = pm_clean

            if self.load_normals:
                fd['normals'] = normals_list[t]

            if self.load_dsm:
                fd['dsm'] = shared_dsm.clone() if self.augment else shared_dsm
                fd['matches'] = matches

            if self.load_lidar:
                fd['lidar_world'] = lidar_world

            # Correspondence maps (computed on raw data before augmentation)
            if self.return_matching:
                self._compute_correspondence_maps(fd)

            frame_dicts.append(fd)

        # ============================================================
        # Phase 10: Augmentation (consistent decisions across all frames)
        # ============================================================
        if self.augment:
            from utils.data_utils import (sample_color_jitter_params, 
                                          sample_noise_augmentation_decisions,
                                          sample_noise_augmentation_params)
            # Respect enable_flip flag for H/V flips
            do_hflip = (random.random() > 0.5) if self.enable_flip else False
            do_vflip = (random.random() > 0.5) if self.enable_flip else False
            do_color = random.random() > 0.5
            # Pre-sample rotation ONCE for consistency across all frames
            # Respect enable_rotation flag
            if self.enable_rotation:
                rot_k = random.choice([0, 1, 2, 3])  # 0=none, 1=90°, 2=180°, 3=270°
            else:
                rot_k = 0  # no rotation
            # Pre-sample color jitter params ONCE for consistency across all frames
            color_params = sample_color_jitter_params() if do_color else None
            # Pre-sample noise augmentation decisions ONCE for consistency
            # Respect enable_noise flag
            if self.enable_noise:
                noise_decisions = sample_noise_augmentation_decisions(force_all=self.force_augment)
            else:
                noise_decisions = {'do_noise': False, 'do_erase': False, 'do_geo_drop': False, 'do_partial': False}
            # Pre-sample noise params using first frame dimensions
            fd0 = frame_dicts[0]
            img_H, img_W = fd0['query'].shape[-2:]
            geo_H, geo_W = fd0.get('dop', fd0.get('dsm', fd0['query'])).shape[-2:]
            noise_params = sample_noise_augmentation_params(img_H, img_W, geo_H, geo_W)
            
            if self.force_augment:
                do_hflip = do_vflip = self.enable_flip  # Only force if enabled
                do_color = True
                rot_k = random.choice([1, 2, 3]) if self.enable_rotation else 0
                color_params = sample_color_jitter_params()
            for i in range(len(frame_dicts)):
                frame_dicts[i] = augment_data(
                    frame_dicts[i],
                    force_hflip=do_hflip,
                    force_vflip=do_vflip,
                    force_color=do_color,
                    color_jitter_params=color_params,
                    force_rotation=rot_k,
                    force_noise=noise_decisions['do_noise'],
                    force_erase=noise_decisions['do_erase'],
                    force_geo_drop=noise_decisions['do_geo_drop'],
                    force_partial=noise_decisions['do_partial'],
                    noise_params=noise_params,
                )

        # ============================================================
        # Phase 11: Normalization (shared DSM stats across all frames)
        # ============================================================
        xyz_norm = self.xyz_normalization
        if xyz_norm in [None, 'none', 'None', False]:
            xyz_norm = None

        if xyz_norm is not None and L > 1:
            # Compute stats across ALL frames' DSMs combined
            all_dsm_valid = []
            for fd in frame_dicts:
                dsm = fd.get('dsm')
                if dsm is not None:
                    valid_mask = (dsm != 0).all(dim=0)
                    if valid_mask.sum() > 0:
                        all_dsm_valid.append(dsm[:, valid_mask])
            if all_dsm_valid:
                combined = torch.cat(all_dsm_valid, dim=1)
                if xyz_norm in ('minmax_01', 'minmax_11'):
                    dsm_offset = combined.min(dim=1).values
                    dsm_scale = (combined.max(dim=1).values - dsm_offset) + 1e-8
                else:
                    dsm_offset = combined.mean(dim=1)
                    dsm_scale = combined.std(dim=1) + 1e-8
                for fd in frame_dicts:
                    fd['dsm_offset'] = dsm_offset
                    fd['dsm_scale'] = dsm_scale

        for i in range(len(frame_dicts)):
            frame_dicts[i] = normalize_data(frame_dicts[i], self.normalization, xyz_norm)

        # ============================================================
        # Phase 12: Build return dict (shared + per-frame split)
        # ============================================================
        result = {}

        # --- Per-frame tensors: stacked with torch.stack → (L, ...) ---
        per_frame_tensor_keys = [
            'query', 'depth', 'point_map', 'query_mask', 'normals',
            'extrinsics', 'extrinsics_raw', 'R', 't',
            'query_in_dop', 'mask_query_in_dop', 'mask_query_in_dop_oneway',
            'dop_in_query', 'mask_dop_in_query', 'mask_dop_in_query_oneway',
        ]
        for key in per_frame_tensor_keys:
            vals = [fd[key] for fd in frame_dicts if key in fd]
            if len(vals) == L:
                result[key] = torch.stack(vals, dim=0)

        # --- Per-frame lists ---
        result['sample_id'] = [fd['sample_id'] for fd in frame_dicts]
        result['lod_uav'] = [fd['lod_uav'] for fd in frame_dicts]
        if self.load_lidar:
            result['lidar_uav'] = [fd['lidar_uav'] for fd in frame_dicts]
        if self.load_dsm:
            result['matches'] = [fd.get('matches', []) for fd in frame_dicts]

        # --- Shared (no time dim) — take from frame 0 ---
        fd0 = frame_dicts[0]

        # Shared spatial tensors (already augmented identically in all frames)
        if self.load_dop and self.return_resized_dop:
            result['dop'] = fd0['dop']
        if self.load_dsm and self.return_resized_dsm:
            result['dsm'] = fd0['dsm']
        result['geodata_mask'] = fd0['geodata_mask']
        result['intrinsics'] = fd0['intrinsics']
        result['intrinsics_raw'] = fd0['intrinsics_raw']

        # Shared LoD mesh (lod_dop from augmented frame dict, vertices/faces/labels unchanged)
        if self.load_lod:
            result['lod_vertices'] = lod_vertices
            result['lod_faces'] = lod_faces
            result['lod_labels'] = lod_labels
            result['lod_dop'] = fd0['lod_dop']  # Use augmented lod_dop from frame dict

        # Shared LiDAR 3D points
        if self.load_lidar:
            result['lidar_world'] = lidar_world

        # Shared metadata
        result['lod_level'] = self.lod_level
        result['crop_bounds'] = crop_bounds
        result['crop_extensions'] = crop_extensions
        result['scale'] = float(scale)
        result['gsd'] = float(scale)

        # When preserving DSM GSD, report the actual DSM resolution info
        if self.preserve_dsm_gsd and self.load_dsm and 'dsm' in result:
            dsm_shape = result['dsm'].shape  # (3, H_native, W_native)
            result['native_dsm_size'] = (int(dsm_shape[1]), int(dsm_shape[2]))
            result['native_dsm_gsd'] = float(native_gsd) if preprocessed else float(scale)
            result['dsm_gsd'] = result['native_dsm_gsd']
        else:
            result['dsm_gsd'] = float(scale)

        # Full-resolution outputs (varying sizes — DataLoader collates as lists)
        result['native_gsd'] = float(native_gsd) if preprocessed else float(scale)
        if resolved_dop_year is not None:
            result['dop_year'] = resolved_dop_year
        if self.return_fullres_dop and fullres_dop is not None:
            result['dop_fullres'] = fullres_dop  # (3, H_native, W_native) at native GSD
        if self.return_fullres_dsm and fullres_dsm is not None:
            result['dsm_fullres'] = fullres_dsm  # (3, H_native, W_native) XYZ at native GSD

        # Normalization metadata (shared)
        for key in ['dsm_offset', 'dsm_scale', 'norm_type']:
            if key in fd0:
                result[key] = fd0[key]

        # Augmentation flags (shared across all frames in sequence)
        for key in ['aug_hflip', 'aug_vflip', 'aug_color', 'aug_rotation',
                    'aug_noise', 'aug_erase', 'aug_geodata_dropout', 'aug_partial_overlap']:
            if key in fd0:
                result[key] = fd0[key]

        result['sequence_length'] = L

        # ============================================================
        # Phase 13: DOP projection params (affine world→pixel mapping)
        # Only needed for visualization scripts, skip during training
        # to avoid expensive np.linalg.lstsq per sample.
        # ============================================================
        if self.return_matching and self.load_dsm and 'dsm' in result and result['dsm'].shape[0] == 3:
            dsm_final = result['dsm']
            _, Hd, Wd = dsm_final.shape
            y_grid, x_grid = torch.meshgrid(
                torch.arange(Hd), torch.arange(Wd), indexing='ij')
            m = (dsm_final[0] != 0).cpu().numpy()
            if m.sum() > 100:
                X = dsm_final[0].cpu().numpy()[m]
                Y = dsm_final[1].cpu().numpy()[m]
                U = x_grid.cpu().numpy()[m]
                V = y_grid.cpu().numpy()[m]
                try:
                    A_mat = np.stack([X, Y, np.ones_like(X)], axis=1)
                    B_mat = np.stack([U, V], axis=1)
                    M = np.linalg.lstsq(A_mat, B_mat, rcond=None)[0]
                    result['dop_proj_params'] = torch.tensor(
                        [M[0, 0], M[1, 0], M[2, 0], M[0, 1], M[1, 1], M[2, 1]],
                        dtype=torch.float32)
                except Exception:
                    pass

        # ============================================================
        # Phase 14: Squeeze temporal dim for single-frame (training-ready)
        # ============================================================
        if L == 1:
            # Per-frame tensors: (1, ...) → (...)
            for key in per_frame_tensor_keys:
                if key in result and isinstance(result[key], torch.Tensor):
                    result[key] = result[key].squeeze(0)

            # Per-frame lists: unwrap single element
            if isinstance(result.get('sample_id'), list):
                result['sample_id'] = result['sample_id'][0]
            for key in ['lod_uav', 'lidar_uav', 'matches']:
                if key in result and isinstance(result[key], list) and len(result[key]) == 1:
                    result[key] = result[key][0]

        return result
