"""
Foundation Model Tracker — uses multi-view 3D reconstruction models (DA3, VGGT,
Pi3, Pi3X, MapAnything, MASt3R, DUSt3R, etc.) as an alternative to optical flow
for inter-frame camera pose estimation.

Key idea: given a keyframe with known absolute pose (from RoMaV2 matching against
orthographic geodata), we use a foundation model to estimate the relative pose of
subsequent frames. The relative pose is then transformed to absolute UTM coordinates
using the known keyframe pose — no Sim(3) alignment needed.

Workflow:
  1. Pipeline establishes a keyframe via RoMaV2 + PnP → known absolute pose
  2. For each subsequent frame, run the foundation model on (keyframe, current)
  3. Extract relative pose from the model's output
  4. Apply relative pose to the known keyframe pose → absolute pose in UTM

This replaces optical flow (LK / WAFT / ptlflow) for inter-frame tracking."""

import numpy as np
import cv2
import torch
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


def _ensure_vggt_importable():
    """Ensure the vggt package is importable by adding the torch hub cache to sys.path.
    VGGT is not pip-installable; it's downloaded via torch.hub or git clone."""
    try:
        import vggt  # noqa: F401
        return
    except ImportError:
        pass
    # Typical torch hub cache location
    hub_dir = Path.home() / '.cache' / 'torch' / 'hub' / 'facebookresearch_vggt_main'
    if hub_dir.exists() and str(hub_dir) not in sys.path:
        sys.path.insert(0, str(hub_dir))
        return
    # Fallback: download via torch.hub (this also puts it on sys.path internally)
    import torch as _torch
    _torch.hub.load('facebookresearch/vggt', 'vggt_1b', trust_repo=True,
                     force_reload=False, source='github')
    if hub_dir.exists() and str(hub_dir) not in sys.path:
        sys.path.insert(0, str(hub_dir))


@dataclass
class FoundationTrackerResult:
    """Result from foundation model pose estimation."""
    position: Optional[np.ndarray]       # Camera center in world (UTM) coords (3,)
    rotation_c2w: Optional[np.ndarray]   # C2W rotation matrix (3, 3)
    confidence: float                    # Quality metric [0, 1]
    inference_time: float                # Seconds


# ===================================================================== #
#  Supported models                                                     #
# ===================================================================== #

_FOUNDATION_MODELS: Dict[str, dict] = {
    # DA3 family (depth_anything_3 package)
    'da3':         {'da3_model': 'da3-large',               'backend': 'da3'},
    'da3_nested':  {'da3_model': 'da3nested-giant-large',   'backend': 'da3'},
    # VGGT (vggt package via torch hub)
    'vggt':            {'backend': 'vggt'},
    'vggt_commercial': {'backend': 'vggt_commercial'},
    # Pi3 / Pi3X (pi3 package)
    'pi3':         {'backend': 'pi3'},
    'pi3x':        {'backend': 'pi3x'},
    # MapAnything (mapanything package)
    'mapanything':  {'backend': 'mapanything'},
    # DUSt3R / MASt3R family (dust3r-family)
    'dust3r':      {'backend': 'dust3r'},
    'mast3r':      {'backend': 'mast3r'},
    'pow3r':       {'backend': 'pow3r'},
}




# ===================================================================== #
#  Tracker class                                                        #
# ===================================================================== #

class FoundationModelTracker:
    """
    Tracks camera poses between keyframes using 3D reconstruction foundation
    models instead of optical flow.

    The tracker stores a keyframe with its known absolute pose, then for each
    new frame runs the foundation model on the (keyframe, current) image pair
    to estimate a relative transform. The absolute pose of the current frame
    is obtained by composing the relative transform with the known keyframe pose."""

    def __init__(self, model_name: str = 'da3_nested', device: str = 'cuda'):
        if model_name not in _FOUNDATION_MODELS:
            raise ValueError(
                f"Unknown foundation model: {model_name}. "
                f"Available: {list(_FOUNDATION_MODELS.keys())}"
            )
        self.model_name = model_name
        self.model_config = _FOUNDATION_MODELS[model_name]
        self.backend = self.model_config['backend']
        self.device = device
        self.model = None  # lazy init

        # Keyframe state
        self.keyframe_id: int = -1
        self.keyframe_image: Optional[np.ndarray] = None
        self.keyframe_image_path: Optional[str] = None   # DA3 needs file paths
        self.keyframe_c2w_world: Optional[np.ndarray] = None  # known absolute C2W (4, 4)
        self.keyframe_position: Optional[np.ndarray] = None   # camera centre in UTM

        # Scale estimation across keyframe segments
        self.estimated_scale: float = 1.0
        self._scale_history: list = []

        # Temp dir for images (DA3 needs file paths)
        # Use model-specific subdir to avoid conflicts between parallel runs
        self.tmp_dir = Path(f'./tmp/fm_tracker/{self.model_name}')
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline-compatibility fields
        self.initial_num_pts: int = 200
        self.tracked_pts_2d: Optional[np.ndarray] = None
        self.tracked_pts_3d: Optional[np.ndarray] = None
        self.tracked_confs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    #  Model loading (lazy)                                                #
    # ------------------------------------------------------------------ #

    def _load_model(self):
        """Lazy-load the foundation model on first use."""
        if self.model is not None:
            return

        backend = self.backend
        device = self.device

        if backend == 'da3':
            from depth_anything_3.api import DepthAnything3
            da3_variant = self.model_config['da3_model']
            _HF_IDS = {
                'da3-large':             'depth-anything/DA3-LARGE-1.1',
                'da3nested-giant-large': 'depth-anything/DA3NESTED-GIANT-LARGE-1.1',
            }
            hf_id = _HF_IDS.get(da3_variant,
                                 f'depth-anything/{da3_variant.upper()}-1.1')
            print(f"  Loading DA3 '{da3_variant}' from '{hf_id}' ...")
            self.model = DepthAnything3.from_pretrained(hf_id)
            self.model.device = device
            self.model = self.model.to(device).eval()

        elif backend == 'vggt':
            _ensure_vggt_importable()
            from vggt.models.vggt import VGGT as _VGGTModel
            print("  Loading VGGT-1B ...")
            self.model = _VGGTModel.from_pretrained("facebook/VGGT-1B")
            self.model = self.model.to(device).eval()

        elif backend == 'vggt_commercial':
            _ensure_vggt_importable()
            from vggt.models.vggt import VGGT as _VGGTModel
            ckpt_path = Path(__file__).resolve().parent.parent / 'checkpoints' / 'vggt_1B_commercial.pt'
            if ckpt_path.exists():
                print(f"  Loading VGGT-1B commercial from {ckpt_path} ...")
                self.model = _VGGTModel()
                state_dict = torch.load(str(ckpt_path), map_location='cpu')
                self.model.load_state_dict(state_dict, strict=True)
            else:
                print("  Loading VGGT-1B (fallback -- commercial ckpt not found) ...")
                self.model = _VGGTModel.from_pretrained("facebook/VGGT-1B")
            self.model = self.model.to(device).eval()

        elif backend == 'pi3':
            from pi3.models.pi3 import Pi3
            print("  Loading Pi3 ...")
            self.model = Pi3.from_pretrained("yyfz233/Pi3")
            self.model = self.model.to(device).eval()

        elif backend == 'pi3x':
            from pi3.models.pi3x import Pi3X
            print("  Loading Pi3X ...")
            self.model = Pi3X.from_pretrained("yyfz233/Pi3X")
            self.model = self.model.to(device).eval()

        elif backend == 'mapanything':
            self._load_mapanything_model('mapanything')

        elif backend == 'dust3r':
            self._load_dust3r_family('dust3r')

        elif backend == 'mast3r':
            self._load_dust3r_family('mast3r')

        elif backend == 'pow3r':
            self._load_dust3r_family('pow3r')

        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _load_mapanything_model(self, model_name: str):
        """Load a MapAnything model via Hydra config."""
        import os
        config_dir = Path(os.environ.get(
            'MAPANYTHING_CONFIG_DIR',
            os.path.expanduser('~/Projects/map-anything/configs'),
        ))
        ckpt_dir = os.environ.get(
            'CHECKPOINTS_DIR',
            str(Path(__file__).resolve().parent.parent / 'checkpoints'),
        )
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
            overrides = [
                f"model={model_name}",
                f"root_pretrained_checkpoints_dir={ckpt_dir}",
            ]
            cfg = compose(config_name="train", overrides=overrides)
        from mapanything.models import init_model
        print(f"  Loading {model_name} via MapAnything ...")
        self.model = init_model(cfg, self.device)
        self.model = self.model.eval()

    def _load_dust3r_family(self, model_name: str):
        """Load DUSt3R / MASt3R / Pow3R model."""
        ckpt_dir = Path(__file__).resolve().parent.parent / 'checkpoints'
        _CKPT_MAP = {
            'dust3r': 'DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth',
            'mast3r': 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
            'pow3r':  'Pow3R_ViTLarge_BaseDecoder_512_linear.pth',
        }
        ckpt_name = _CKPT_MAP.get(model_name)
        if ckpt_name:
            ckpt_path = ckpt_dir / ckpt_name
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        # Try loading via MapAnything factory (most robust)
        try:
            self._load_mapanything_model(model_name)
        except Exception:
            # Fallback: direct loading
            print(f"  MapAnything loading failed, trying direct import for {model_name} ...")
            if model_name == 'dust3r':
                from dust3r.model import AsymmetricCroCo3DStereo
                self.model = AsymmetricCroCo3DStereo.from_pretrained(
                    str(ckpt_dir / _CKPT_MAP['dust3r'])
                ).to(self.device).eval()
            elif model_name == 'mast3r':
                from mast3r.model import AsymmetricMASt3R
                self.model = AsymmetricMASt3R.from_pretrained(
                    str(ckpt_dir / _CKPT_MAP['mast3r'])
                ).to(self.device).eval()
            else:
                raise

    # ------------------------------------------------------------------ #
    #  Keyframe management                                                 #
    # ------------------------------------------------------------------ #

    def set_keyframe(self, frame_id: int, image: np.ndarray,
                     absolute_position: np.ndarray,
                     absolute_rotation_c2w: Optional[np.ndarray] = None):
        """
        Set a new keyframe with known absolute pose.

        Args:
            frame_id: Frame ID.
            image: Keyframe image (RGB, uint8, H×W×3).
            absolute_position: Camera centre in world (UTM) coordinates (3,).
            absolute_rotation_c2w: C2W rotation matrix (3, 3). If None, identity
                rotation is assumed."""
        self.keyframe_id = frame_id
        self.keyframe_image = image.copy()
        self.keyframe_position = absolute_position.copy()

        # Build 4×4 C2W
        self.keyframe_c2w_world = np.eye(4, dtype=np.float64)
        if absolute_rotation_c2w is not None:
            self.keyframe_c2w_world[:3, :3] = absolute_rotation_c2w
        self.keyframe_c2w_world[:3, 3] = absolute_position

        # DA3 needs file paths
        if self.backend == 'da3':
            self.keyframe_image_path = self._save_image(image, f"keyframe_{frame_id}")

        # Reset compatibility fields
        self.initial_num_pts = 200
        self._set_compat_points(200)

    def _set_compat_points(self, n: int):
        """Set dummy 2D/3D points for pipeline compatibility."""
        self.tracked_pts_2d = np.zeros((n, 2), dtype=np.float32)
        self.tracked_pts_3d = np.zeros((n, 3), dtype=np.float64)
        self.tracked_confs = np.ones(n, dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Pose estimation                                                     #
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    #  Relative → absolute helper                                          #
    # ------------------------------------------------------------------ #



    # ------------------------------------------------------------------ #
    #  DA3 backend                                                         #
    # ------------------------------------------------------------------ #

    def _save_image(self, image: np.ndarray, name: str) -> str:
        path = self.tmp_dir / f"{name}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                     [cv2.IMWRITE_JPEG_QUALITY, 95])
        return str(path)


    # ------------------------------------------------------------------ #
    #  VGGT backend                                                        #
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    #  Pi3 / Pi3X backend                                                  #
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    #  MapAnything / DUSt3R / MASt3R / Pow3R backend                      #
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    #  GPU memory management                                               #
    # ------------------------------------------------------------------ #

    def offload_to_cpu(self):
        """Move model to CPU to free GPU memory for keyframe matching."""
        if self.model is not None and hasattr(self.model, 'cpu'):
            try:
                self.model.cpu()
            except Exception:
                pass
            torch.cuda.empty_cache()

    def reload_to_gpu(self):
        """Move model back to GPU after keyframe matching."""
        if self.model is not None and hasattr(self.model, 'to'):
            try:
                self.model.to(self.device)
            except Exception:
                pass

    def offload_waft(self):
        """Compatibility alias — offloads foundation model to CPU."""
        self.offload_to_cpu()

    def reload_waft(self):
        """Compatibility alias — reloads foundation model to GPU."""
        self.reload_to_gpu()
