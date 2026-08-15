"""
Run foundation model inference on extracted MovingDrone frames.

Supported models:
  MapAnything family (via MapAnything's unified model factory):
  - mapanything       : MapAnything (images only)
  - mapanything_K     : MapAnything + known camera intrinsics  (--provide_intrinsics)
  - mapanything_K+D   : MapAnything + intrinsics + depth       (--provide_intrinsics --provide_depth)
  - mapanything_K+D+P : MapAnything + intrinsics + depth + GT poses (oracle upper bound)
  - vggt              : VGGT (loads pretrained weights via torch hub)
  - vggt_commercial   : VGGT commercial license variant (needs checkpoints/vggt_1B_commercial.pt)
  - dust3r            : DUSt3R with global BA  (needs checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth)
  - mast3r            : MASt3R with sparse GA  (needs checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth)
  - metric_dust3r     : DUSt3R backbone fine-tuned for metric depth/scale (same ckpt as MASt3R)
  - must3r            : MUSt3R scalable matcher  (needs checkpoints/MUSt3R_512.pth + retrieval ckpt)
  - pow3r             : Pow3R feed-forward  (needs checkpoints/Pow3R_ViTLarge_BaseDecoder_512_linear.pth)
  - pow3r_ba          : Pow3R + global BA  (same ckpt as pow3r)
  - pi3               : π³ (loads pretrained weights automatically)
  - pi3x              : π³X (upgraded Pi3 with ConvHead; pip install -e ~/Projects/thirdparty_pi3)
  - moge_1            : MoGe v1 monocular geometry  (downloads Ruicheng/moge-vitl)
  - moge_2            : MoGe v2 monocular geometry  (downloads Ruicheng/moge-2-vitl)
  - modular_dust3r    : Modular DUSt3R with DPT head  (auto-download)
  - anycalib          : AnyCalib monocular intrinsics prediction (pip install anycalib)

  Depth Anything 3 (native, via depth_anything_3 package):
  - da3               : DA3-Large (fast, good quality; pip install -e ~/Projects/thirdparty_da3)
  - da3_nested        : DA3-Nested-Giant-Large (best quality, ~2x slower than da3)

  SLAM-based systems (subprocess runners — require separate repo + conda env):
  - vggt_slam_v1      : VGGT-SLAM v1 (NeurIPS 2025; branch version1.0) — conda env: vggt-slam
  - vggt_slam_v2      : VGGT-SLAM v2 (main branch) — conda env: vggt-slam
  - vggt_long         : VGGT-Long (chunk-based long-seq SLAM) — conda env: vggt-long

  SLAM model setup:
    # VGGT-SLAM v1
    git clone -b version1.0 https://github.com/MIT-SPARK/VGGT-SLAM.git ~/Projects/VGGT-SLAM-v1
    conda create -n vggt-slam ...  # see repo README
    export VGGT_SLAM_V1_DIR=~/Projects/VGGT-SLAM-v1

    # VGGT-SLAM v2
    git clone https://github.com/MIT-SPARK/VGGT-SLAM.git ~/Projects/VGGT-SLAM
    export VGGT_SLAM_V2_DIR=~/Projects/VGGT-SLAM

    # VGGT-Long
    git clone https://github.com/DengKaiCQ/VGGT-Long.git ~/Projects/VGGT-Long
    conda create -n vggt-long ...  # see repo README
    export VGGT_LONG_DIR=~/Projects/VGGT-Long

MapAnything input modes: The same single model supports 12+ reconstruction tasks depending on
what geometric inputs you provide alongside images:
  - images only          → uncalibrated multi-view SfM
  - + intrinsics (K)     → calibrated multi-view SfM
  - + depth (D)          → multi-view stereo with depth
  - + poses (P)          → camera-pose-conditioned reconstruction
The K/D/P suffixes in output names indicate what was provided.

Checkpoint paths (for models that need local files) are resolved via:
  ~/Projects/map-anything/configs/machine/default.yaml → root_pretrained_checkpoints_dir"""

import argparse
import json
import time
import numpy as np
import torch
import os
from pathlib import Path
from PIL import Image
from utils.pose import build_intrinsics_matrix
from utils.depth import load_depth
from utils.tensor_ops import _move_to_device
from utils.data_utils import load_ground_truth

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# --------------------------------------------------------------------------- #
#  Model resolution / normalization requirements                               #
#  Drawn from each model's configs/model/*.yaml and MapAnything README.        #
# --------------------------------------------------------------------------- #
#
#  resolution  : target image resolution passed to load_images / preprocess_inputs
#  norm_type   : 'dinov2' (ImageNet), 'dust3r' (DUSt3R), 'identity' (no norm)
#  patch_size  : ViT patch size (14 for DINOv2-based, 16 for CroCo-based)
#
MODEL_CONFIGS = {
    # ---- MapAnything (single model, multiple input modes) -----------------
    'mapanything': {'resolution': 518, 'norm_type': 'dinov2', 'patch_size': 14},
    # ---- Feed-forward pose & geometry estimation -------------------------
    'vggt':        {'resolution': 518, 'norm_type': 'identity', 'patch_size': 14},
    'vggt_commercial': {'resolution': 518, 'norm_type': 'identity', 'patch_size': 14},
    'pi3':         {'resolution': 518, 'norm_type': 'identity', 'patch_size': 14},
    # Pi3X: upgraded Pi3 with ConvHead, better confidence, metric-scale support.
    # Requires: pip install -e ~/Projects/thirdparty_pi3  (pi3 package with Pi3X class)
    'pi3x':        {'resolution': 518, 'norm_type': 'identity', 'patch_size': 14, 'custom': True},
    'moge_1':      {'resolution': 518, 'norm_type': 'identity', 'patch_size': 14, 'monocular': True},
    'moge_2':      {'resolution': 518, 'norm_type': 'identity', 'patch_size': 14, 'monocular': True},
    # ---- DUSt3R / MASt3R family (CroCo backbone) -------------------------
    'dust3r':      {'resolution': 512, 'norm_type': 'dust3r', 'patch_size': 16},
    'mast3r':      {'resolution': 512, 'norm_type': 'dust3r', 'patch_size': 16},
    'metric_dust3r': {'resolution': 512, 'norm_type': 'dust3r', 'patch_size': 16},
    'must3r':      {'resolution': 512, 'norm_type': 'dust3r', 'patch_size': 16},
    # pow3r (bare) is PAIRWISE-ONLY (exactly 2 views). For N-view eval use pow3r_ba.
    'pow3r':       {'resolution': 512, 'norm_type': 'dust3r', 'patch_size': 16, 'pairwise': True},
    'pow3r_ba':     {'resolution': 512, 'norm_type': 'dust3r', 'patch_size': 16},
    # Modular DUSt3R — Hydra config is named modular_dust3r_512_dpt.yaml
    'modular_dust3r': {'resolution': 512, 'norm_type': 'dust3r', 'patch_size': 16,
                       'config_override': 'modular_dust3r_512_dpt'},
    # ---- Monocular calibration (intrinsics-only) -------------------------
    # AnyCalib predicts camera intrinsics from a single image.
    # Outputs intrinsics + ray_directions (no poses or 3D points).
    'anycalib':    {'resolution': 518, 'norm_type': 'identity', 'patch_size': 14,
                    'monocular': True, 'calibration_only': True},
    # ---- Depth Anything 3 (DA3) ----------------------------------------
    # DA3 takes direct image file paths (not normalised tensors).
    # Two variants: da3 (da3-large, fast) and da3_nested (da3nested-giant-large, best quality).
    # Requires: pip install -e ~/Projects/thirdparty_da3  (depth_anything_3 package)
    'da3':         {'resolution': 504, 'norm_type': 'identity', 'patch_size': 14,
                    'custom': True, 'da3_model': 'da3-large'},
    'da3_nested':  {'resolution': 504, 'norm_type': 'identity', 'patch_size': 14,
                    'custom': True, 'da3_model': 'da3nested-giant-large'},
    # ---- SLAM-based systems (subprocess runners) ------------------------
    # These require separate cloned repos and conda environments.
    # See setup instructions printed when you first try to run them.
    'vggt_slam_v1': {'resolution': 518, 'norm_type': 'identity', 'patch_size': 14,
                     'subprocess': True,
                     'subprocess_repo_env': 'VGGT_SLAM_V1_DIR',
                     'subprocess_repo_default': 'thirdparty/VGGT-SLAM-v1',
                     'subprocess_conda_env': 'vggt-slam-v1',
                     'subprocess_script': 'main.py',
                     'subprocess_extra_args': [],
                     'subprocess_output_format': 'slam_tum'},  # frame_id x y z qx qy qz qw
    'vggt_slam_v2': {'resolution': 518, 'norm_type': 'identity', 'patch_size': 14,
                     'subprocess': True,
                     'subprocess_repo_env': 'VGGT_SLAM_V2_DIR',
                     'subprocess_repo_default': 'thirdparty/VGGT-SLAM-v2',
                     'subprocess_conda_env': 'vggt-slam-v2',
                     'subprocess_script': 'main.py',
                     'subprocess_extra_args': [],
                     'subprocess_output_format': 'slam_tum'},  # frame_id x y z qx qy qz qw
    'vggt_long':   {'resolution': 518, 'norm_type': 'identity', 'patch_size': 14,
                    'subprocess': True,
                    'subprocess_repo_env': 'VGGT_LONG_DIR',
                    'subprocess_repo_default': 'thirdparty/VGGT-Long',
                    'subprocess_conda_env': 'vggt-long',
                    'subprocess_script': 'vggt_long.py',
                    'subprocess_extra_args': [],
                    'subprocess_output_format': 'vggt_long_txt'},  # 16 floats (4x4 C2W flat)
}








def prepare_views(
    gt_data: dict,
    data_dir: Path,
    sequence: str,
    model_config: dict,
    model_name: str = '',
    provide_intrinsics: bool = False,
    provide_depth: bool = False,
    provide_poses: bool = False,
    device: str = 'cuda',
) -> list:
    """
    Prepare input views in MapAnything's expected format.

    Image-only: uses load_images (fast, handles resize+norm internally).
    Multi-modal: uses preprocess_inputs (handles consistent resize of K, depth, poses)."""
    image_paths = [
        str(data_dir / sequence / frame['image_path'])
        for frame in gt_data['frames']
    ]
    needs_multimodal = provide_intrinsics or provide_depth or provide_poses

    if not needs_multimodal:
        from mapanything.utils.image import load_images
        views = load_images(
            folder_or_list=image_paths,
            resolution_set=model_config['resolution'],
            norm_type=model_config['norm_type'],
            patch_size=model_config['patch_size'],
        )
    else:
        from mapanything.utils.image import preprocess_inputs
        from PIL import Image as PILImage

        K_orig = build_intrinsics_matrix(gt_data)
        raw_views = []
        for i, frame in enumerate(gt_data['frames']):
            img = np.array(PILImage.open(image_paths[i]).convert('RGB'))
            view = {'img': img}
            if provide_intrinsics:
                view['intrinsics'] = K_orig.copy()
            if provide_depth and frame.get('depth_path'):
                depth = load_depth(frame['depth_path'], data_dir, sequence)
                if depth is not None:
                    view['depth_z'] = depth
                    view['is_metric_scale'] = torch.tensor([True])
            if provide_poses:
                view['camera_poses'] = np.array(frame['c2w'], dtype=np.float32)
            raw_views.append(view)

        views = preprocess_inputs(
            raw_views,
            resolution_set=model_config['resolution'],
            norm_type=model_config['norm_type'],
            patch_size=model_config.get('patch_size', 14),
        )

    # Add label/instance keys required by MASt3R / DUSt3R wrappers.
    # MapAnything explicitly rejects 'label' — only add it for dust3r-family models.
    _LABEL_MODELS = {'mast3r', 'must3r', 'metric_dust3r', 'dust3r', 'pow3r', 'pow3r_ba'}
    for i, view in enumerate(views):
        view.setdefault('instance', [str(i)])  # safe: in mapanything's allowed list
        if model_name in _LABEL_MODELS:
            view.setdefault('label', [sequence])
    return views


def _normalize_view_keys(views: list) -> list:
    """
    Normalize view dict keys to satisfy multiple model wrappers.
    MapAnything preprocess_inputs uses: 'intrinsics', 'depth_z', 'camera_poses'
    Pow3R/Pow3RBA wrapper expects:      'camera_intrinsics', 'depthmap', 'camera_pose'

    When values are unavailable (images-only mode), we provide placeholder tensors
    so that Pow3R's loss_of_one_batch can call .to(device) on them.
    The actual values don't matter: pow3r_ba (images_only) uses overall_prob=0,
    so none of these conditioning inputs are ever used by the model."""
    for view in views:
        B, C, H, W = view["img"].shape
        # Intrinsics: preprocess_inputs → 'intrinsics'; Pow3R expects 'camera_intrinsics'
        if 'camera_intrinsics' not in view:
            intr = view.get('intrinsics', None)
            if intr is None:
                # Rough placeholder: diagonal focal matrix
                f = float(max(H, W))
                intr = torch.tensor([[[f, 0, W/2.], [0, f, H/2.], [0, 0, 1.]]])  # (1,3,3)
            view['camera_intrinsics'] = intr
        # Depth: preprocess_inputs → 'depth_z' (1,H,W); Pow3R expects 'depthmap' (H,W) or (1,H,W)
        if 'depthmap' not in view:
            dz = view.get('depth_z', None)
            if dz is not None:
                view['depthmap'] = dz[0] if dz.ndim == 3 else dz
            else:
                view['depthmap'] = torch.ones(1, H, W)  # placeholder all-ones depth
        # Pose: preprocess_inputs → 'camera_poses'; Pow3R expects 'camera_pose'
        if 'camera_pose' not in view:
            cp = view.get('camera_poses', None)
            if cp is None:
                cp = torch.eye(4, dtype=torch.float32).unsqueeze(0)  # (1,4,4) identity
            view['camera_pose'] = cp
    return views




def run_inference(model, views: list, model_name: str, memory_efficient: bool = True,
                  device: str = 'cuda') -> list:
    """
    Run inference.  MapAnything ('mapanything') uses the high-level infer() API.
    All external models use forward() directly (no bf16 autocast — causes dtype
    issues in MASt3R/MUSt3R reciprocal NN matching).
    Monocular models (MoGe) require one view at a time — run frame-by-frame."""
    if hasattr(model, 'infer'):
        return model.infer(
            views,
            memory_efficient_inference=memory_efficient,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
        )
    else:
        # Determine device: try model parameters first, then fall back to supplied device
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            # MUSt3R stores encoder/decoder in a plain tuple — no registered params
            model_device = torch.device(device)
        views = _move_to_device(views, model_device)
        model.eval()
        with torch.no_grad():
            # Monocular models (e.g. MoGe) only accept 1 view — iterate frames
            try:
                return model(views)
            except AssertionError as e:
                if 'only supports 1' in str(e) or 'input view' in str(e).lower():
                    print(f"  Monocular model detected — running frame-by-frame ({len(views)} frames)")
                    results = []
                    for i, v in enumerate(views):
                        with torch.no_grad():
                            out = model([v])
                        results.extend(out)
                    return results
                raise
            except ValueError as e:
                # MASt3R sparse_global_alignment can fail with "not enough values
                # to unpack" when zero feature matches are found (e.g. repetitive
                # aerial textures). Return None so caller can handle gracefully.
                if 'not enough values to unpack' in str(e):
                    print(f"  [ERROR] {model_name}: zero matches in sparse alignment — skipping ({e})")
                    return None
                raise


def extract_results(predictions: list, model_name: str) -> dict:
    """Extract camera poses, dense outputs, and intrinsics from unified prediction dicts."""
    positions, rotations, depths, pts3d_list, intrinsics_list = [], [], [], [], []

    for pred in predictions:
        # ---- Camera pose ------------------------------------------------- #
        if 'camera_poses' in pred and pred['camera_poses'] is not None:
            c2w = pred['camera_poses'].cpu().numpy()
            if c2w.ndim == 3:
                c2w = c2w[0]
            positions.append(c2w[:3, 3])
            rotations.append(c2w[:3, :3])
        elif 'cam_trans' in pred and pred['cam_trans'] is not None:
            pos = pred['cam_trans'].cpu().numpy().ravel()
            positions.append(pos[:3])
            if 'cam_quats' in pred and pred['cam_quats'] is not None:
                from scipy.spatial.transform import Rotation
                q = pred['cam_quats'].cpu().numpy().ravel()
                rotations.append(Rotation.from_quat(q).as_matrix())
            else:
                rotations.append(np.eye(3))
        else:
            positions.append(np.zeros(3))
            rotations.append(np.eye(3))

        # ---- Depth ------------------------------------------------------- #
        if 'depth_z' in pred and pred['depth_z'] is not None:
            d = pred['depth_z'].cpu().numpy()
            depths.append(d[0] if d.ndim > 2 else d)

        # ---- Dense 3-D points -------------------------------------------- #
        if 'pts3d' in pred and pred['pts3d'] is not None:
            p = pred['pts3d'].cpu().numpy()
            pts3d_list.append(p[0] if p.ndim > 3 else p)

        # ---- Intrinsics (e.g. from AnyCalib / MoGe) --------------------- #
        if 'intrinsics' in pred and pred['intrinsics'] is not None:
            intr = pred['intrinsics']
            if isinstance(intr, torch.Tensor):
                intr = intr.cpu().numpy()
            if intr.ndim == 3:
                intr = intr[0]  # (B, 3, 3) -> (3, 3)
            intrinsics_list.append(intr)

    return {
        'model': model_name,
        'n_views': len(predictions),
        'positions':  np.array(positions)  if positions  else np.zeros((0, 3)),
        'rotations':  np.array(rotations)  if rotations  else np.zeros((0, 3, 3)),
        'depths':     depths     or None,
        'pts3d':      pts3d_list or None,
        'intrinsics': intrinsics_list or None,
    }


def save_results(results: dict, output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'positions': results['positions'],
        'rotations': results['rotations'],
    }
    if results.get('frame_indices') is not None:
        save_dict['frame_indices'] = results['frame_indices']
    np.savez(output_path / 'predictions.npz', **save_dict)

    if results.get('depths'):
        for i, d in enumerate(results['depths']):
            np.savez_compressed(output_path / f'depth_{i:04d}.npz', depth=d)
    if results.get('pts3d'):
        for i, p in enumerate(results['pts3d']):
            np.savez_compressed(output_path / f'pts3d_{i:04d}.npz', pts3d=p)
    if results.get('intrinsics'):
        intr_arr = np.array(results['intrinsics'])  # (N, 3, 3)
        np.savez(output_path / 'intrinsics.npz', intrinsics=intr_arr)

    meta = {
        'model': results['model'],
        'n_views': results['n_views'],
        'has_depths': results.get('depths') is not None,
        'has_pts3d':  results.get('pts3d') is not None,
        'has_intrinsics': results.get('intrinsics') is not None,
        'inference_time': results.get('inference_time'),
    }
    with open(output_path / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Results saved to {output_path}")


def get_suffix(provide_intrinsics, provide_depth, provide_poses):
    parts = (['K'] if provide_intrinsics else []) \
          + (['D'] if provide_depth else []) \
          + (['P'] if provide_poses else [])
    return ('_' + '+'.join(parts)) if parts else ''


# Path to MapAnything Hydra configs.  Override with env var MAPANYTHING_CONFIG_DIR.
_MA_CONFIG_DIR = Path(os.environ.get(
    'MAPANYTHING_CONFIG_DIR',
    os.path.expanduser('~/Projects/map-anything/configs'),
))

# Project root is the directory containing this script.
_PROJECT_ROOT = Path(__file__).resolve().parent

# Directory containing pretrained model checkpoints (for DUSt3R, MASt3R, MUSt3R,
# Pow3R, VGGT-commercial, etc.).  Override with env var CHECKPOINTS_DIR.
_CHECKPOINTS_DIR = os.environ.get(
    'CHECKPOINTS_DIR',
    str(_PROJECT_ROOT / 'checkpoints'),
)

# Module-level model cache: keeps models loaded between sequence/window calls
# within the same subprocess.  Key: (model_name, device).  This avoids the
# ~27s reload overhead when running windowed inference across many windows.
_model_cache: dict = {}


# --------------------------------------------------------------------------- #
#  SwiGLU replacement (MapAnything HF checkpoint workaround)                  #
# --------------------------------------------------------------------------- #

class _SwiGLUFFNFused(torch.nn.Module):
    """
    Replace xformers.ops.SwiGLUFFNFused for environments without xformers.
    Uses w12 (fused gate+value) and w3 (output) parameter names to match
    the facebook/map-anything HuggingFace checkpoint."""
    def __init__(self, in_features, hidden_features, out_features=None, bias=True):
        super().__init__()
        out_features = out_features or in_features
        self.w12 = torch.nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3  = torch.nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(torch.nn.functional.silu(x1) * x2)


def _replace_mlp_with_swiglu(model):
    if not hasattr(model, 'info_sharing'):
        return
    blocks = getattr(model.info_sharing, 'self_attention_blocks', [])
    n = 0
    for block in blocks:
        old = getattr(block, 'mlp', None)
        if old is None or not hasattr(old, 'fc1'):
            continue
        in_f = old.fc1.in_features
        hidden_old = old.fc1.out_features
        out_f = old.fc2.out_features
        bias = old.fc1.bias is not None
        hidden_swiglu = int(2 * hidden_old / 3)
        block.mlp = _SwiGLUFFNFused(in_f, hidden_swiglu, out_f, bias)
        n += 1
    if n:
        print(f"  Replaced {n} Mlp modules with SwiGLU")


# --------------------------------------------------------------------------- #
#  Pi3X wrapper (loaded directly from the pi3 package, not via MapAnything)   #
# --------------------------------------------------------------------------- #

class _Pi3XWrapper(torch.nn.Module):
    """
    Pi3X inference wrapper with the same output format as MapAnything's Pi3Wrapper.

    Pi3X is the upgraded variant of Pi3 (Pi3 repo, class pi3.models.pi3x.Pi3X).
    Install:  pip install -e ~/Projects/thirdparty_pi3

    Differences from Pi3: ConvHead output (smoother), better confidence, metric-scale support,
    optional conditioning (poses / intrinsics / depth).

    Output per view: same dict as Pi3Wrapper → cam_trans, cam_quats, pts3d, pts3d_cam."""
    def __init__(self, torch_hub_force_reload: bool = False):
        super().__init__()
        from pi3.models.pi3x import Pi3X
        print("Loading Pi3X from HuggingFace cache ...")
        self.model = Pi3X.from_pretrained(
            "yyfz233/Pi3X",
            force_download=torch_hub_force_reload,
        )
        self.dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

    def forward(self, views: list) -> list:
        """
        Args:
            views: list of dicts with keys 'img' (B, C, H, W) in [0, 1] range, 'data_norm_type'.
        Returns:
            list of dicts, one per view, with keys: cam_trans, cam_quats, pts3d, pts3d_cam."""
        # Validate normalisation
        assert views[0].get('data_norm_type', ['identity'])[0] == 'identity', (
            "Pi3X expects identity normalisation"
        )
        num_views = len(views)
        images = torch.stack([v["img"] for v in views], dim=1)  # (B, N, C, H, W)

        with torch.autocast("cuda", dtype=self.dtype):
            results = self.model(images)

        from scipy.spatial.transform import Rotation as _Rot
        res = []
        with torch.no_grad():
            for vi in range(num_views):
                c2w = results["camera_poses"][:, vi, ...]  # (B, 4, 4)
                cam_trans = c2w[..., :3, 3]               # (B, 3)
                # Rotation matrix → quaternion (x, y, z, w) — one sample at a time
                rot_np = c2w[0, :3, :3].float().cpu().numpy()  # (3, 3)
                q = _Rot.from_matrix(rot_np).as_quat()         # [x, y, z, w]
                cam_quats = torch.tensor(q, dtype=cam_trans.dtype, device=cam_trans.device
                                         ).unsqueeze(0)         # (1, 4)
                pts3d     = results["points"][:, vi, ...]
                pts3d_cam = results.get("local_points", results["points"])[:, vi, ...]
                depth     = torch.norm(pts3d_cam.float(), dim=-1, keepdim=True)
                res.append({
                    "cam_trans":      cam_trans,
                    "cam_quats":      cam_quats,
                    "pts3d":          pts3d,
                    "pts3d_cam":      pts3d_cam,
                    "depth_along_ray": depth,
                })
        return res


# --------------------------------------------------------------------------- #
#  DA3 (Depth Anything 3) inference helpers                                    #
# --------------------------------------------------------------------------- #

def _load_da3_model(da3_model: str = 'da3-large', device: str = 'cuda'):
    """
    Load a Depth Anything 3 model.

    Requires: pip install -e ~/Projects/thirdparty_da3  (depth_anything_3 package)

    Args:
        da3_model: Internal model variant name, e.g. 'da3-large', 'da3nested-giant-large'.
            Full list: 'da3-base','da3-small','da3-large','da3-giant','da3metric-large',
                       'da3mono-large','da3nested-giant-large'
        device: Target device string."""
    from depth_anything_3.api import DepthAnything3
    # from_pretrained uses PyTorchModelHubMixin; model_name is stored in the HF config.json.
    # The HF repo IDs follow the naming convention DATECASED-VARIANT.
    # Map internal variant → HF repo ID (release 1.1):
    _HF_IDS = {
        'da3-base':              'depth-anything/DA3-BASE-1.1',
        'da3-small':             'depth-anything/DA3-SMALL-1.1',
        'da3-large':             'depth-anything/DA3-LARGE-1.1',
        'da3-giant':             'depth-anything/DA3-GIANT-1.1',
        'da3metric-large':       'depth-anything/DA3METRIC-LARGE-1.1',
        'da3mono-large':         'depth-anything/DA3MONO-LARGE-1.1',
        'da3nested-giant-large': 'depth-anything/DA3NESTED-GIANT-LARGE-1.1',
    }
    hf_id = _HF_IDS.get(da3_model, f'depth-anything/{da3_model.upper().replace("-", "-")}-1.1')
    print(f"  Loading DA3 model '{da3_model}' from HF repo '{hf_id}' ...")
    model = DepthAnything3.from_pretrained(hf_id)
    model.device = device
    return model.to(device).eval()


def _run_da3_inference(model, gt_data: dict, data_dir: 'Path', sequence: str) -> dict:
    """
    Run DA3 inference using raw image file paths.

    DA3 takes a list of file-path strings (or PIL Images) and returns a Prediction with:
        prediction.extrinsics : np.ndarray (N, 4, 4)  W2C camera matrices (OpenCV format)
        prediction.intrinsics : np.ndarray (N, 3, 3)

    We invert the W2C matrices to get C2W, then extract positions and rotations
    consistent with the rest of the pipeline."""
    image_paths = [
        str(data_dir / sequence / frame['image_path'])
        for frame in gt_data['frames']
    ]
    prediction = model.inference(image_paths)

    positions, rotations = [], []
    if prediction.extrinsics is not None:
        for ext in prediction.extrinsics:   # (3, 4) or (4, 4) W2C
            ext = np.array(ext, dtype=np.float64)
            if ext.shape == (4, 4):
                c2w = np.linalg.inv(ext)
            else:
                # (3, 4) top-3 rows of W2C: [R | t]
                # C2W analytically: R_c2w = R^T, t_c2w = -R^T @ t
                R = ext[:3, :3]
                t = ext[:3, 3]
                c2w = np.eye(4, dtype=np.float64)
                c2w[:3, :3] = R.T
                c2w[:3,  3] = -R.T @ t
            positions.append(c2w[:3, 3].astype(np.float32))
            rotations.append(c2w[:3, :3].astype(np.float32))
    else:
        n = len(gt_data['frames'])
        positions = [np.zeros(3, dtype=np.float32)] * n
        rotations = [np.eye(3, dtype=np.float32)] * n

    return {
        'positions': np.array(positions),
        'rotations': np.array(rotations),
    }


# --------------------------------------------------------------------------- #
#  Subprocess runners for SLAM-based systems (VGGT-SLAM v1/v2, VGGT-Long)     #
# --------------------------------------------------------------------------- #


def _resolve_slam_repo(model_config: dict) -> 'tuple[Path | None, str]':
    """
    Return (repo_path_or_None, error_message).
    Reads repo dir from an env var, or falls back to the default path.  Relative
    default paths are resolved against the project root (script directory)."""
    env_var = model_config.get('subprocess_repo_env', '')
    raw_default = model_config.get('subprocess_repo_default', '')
    raw_dir = os.environ.get(env_var, raw_default) if env_var else raw_default
    if not raw_dir:
        return None, 'No repo path configured.'
    expanded = os.path.expanduser(raw_dir)
    # Resolve relative paths against the project root so that
    # 'thirdparty/VGGT-SLAM-v1' works regardless of cwd.
    repo_p = Path(expanded) if os.path.isabs(expanded) else _PROJECT_ROOT / expanded
    if not repo_p.exists():
        return None, (
            f"Repository not found at '{repo_p}'.\n"
            f"  • Clone the repo there (or set env var {env_var}=<path>)\n"
            f"    default: {repo_p}"
        )
    return repo_p, ''


def _check_conda_env(conda_env: str) -> bool:
    """Return True if the given conda environment name exists."""
    import subprocess
    result = subprocess.run(
        ['conda', 'env', 'list'],
        capture_output=True, text=True,
    )
    return f'\n{conda_env} ' in result.stdout or f'\n{conda_env}\n' in result.stdout


def _parse_slam_tum_poses(poses_file: Path) -> dict:
    """
    Parse VGGT-SLAM pose file (TUM-like format: frame_id x y z qx qy qz qw per line).
    Returns {'positions': np.ndarray (N, 3), 'rotations': np.ndarray (N, 3, 3),
             'frame_ids': np.ndarray (N,) of int}.
    frame_ids are the numeric identifiers extracted from image filenames by VGGT-SLAM
    (e.g. frame_0030.png → 30).  VGGT-SLAM performs keyframe selection internally,
    so N may be smaller than the number of input images."""
    from scipy.spatial.transform import Rotation as _Rot
    positions, rotations, frame_ids = [], [], []
    with open(poses_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = [float(v) for v in line.split()]
            if len(vals) < 8:
                continue
            frame_id = int(float(vals[0]))
            pos = np.array(vals[1:4], dtype=np.float32)   # x y z (C2W translation)
            q   = np.array(vals[4:8], dtype=np.float64)   # qx qy qz qw (scipy order)
            rot = _Rot.from_quat(q).as_matrix().astype(np.float32)
            frame_ids.append(frame_id)
            positions.append(pos)
            rotations.append(rot)
    # Sort by frame id
    order = np.argsort(frame_ids)
    return {
        'positions': np.array(positions)[order],
        'rotations': np.array(rotations)[order],
        'frame_ids': np.array(frame_ids, dtype=int)[order],
    }


def _parse_vggt_long_poses(poses_file: Path) -> dict:
    """
    Parse VGGT-Long pose file (16 floats per line = 4x4 C2W matrix flattened)."""
    positions, rotations = [], []
    with open(poses_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = [float(v) for v in line.split()]
            if len(vals) < 16:
                continue
            c2w = np.array(vals, dtype=np.float32).reshape(4, 4)
            positions.append(c2w[:3, 3])
            rotations.append(c2w[:3, :3])
    return {
        'positions': np.array(positions) if positions else np.zeros((0, 3)),
        'rotations': np.array(rotations) if rotations else np.zeros((0, 3, 3)),
    }


def _run_subprocess_model(
    model_name: str,
    gt_data: dict,
    data_dir: Path,
    sequence: str,
    model_config: dict,
    device: str = 'cuda',
) -> 'dict | None':
    """
    Run a SLAM-based model as a subprocess.

    Steps:
      1. Verify the repo directory and conda env exist.
      2. Copy / symlink extracted frame images to a temp folder.
      3. Run the model's script in the conda env via subprocess.
      4. Parse the output pose file.
      5. Return {'positions': ..., 'rotations': ...}.

    Returns None on setup failure (prints instructions)."""
    import shutil
    import subprocess
    import tempfile

    # ---- 1. Check repo -------------------------------------------------- #
    repo_p, err = _resolve_slam_repo(model_config)
    if repo_p is None:
        conda_env = model_config.get('subprocess_conda_env', '')
        print(f"\n[SETUP REQUIRED] Cannot run '{model_name}':")
        print(f"  {err}")
        print(f"  • After cloning, set up conda env '{conda_env}' following the repo README.")
        print(f"  • Then re-run this script.")
        return None

    conda_env = model_config.get('subprocess_conda_env', '')
    if not _check_conda_env(conda_env):
        print(f"\n[SETUP REQUIRED] Conda env '{conda_env}' not found for '{model_name}'.")
        print(f"  • Follow '{repo_p}/README.md' to create the environment.")
        print(f"  • Expected env name: '{conda_env}'")
        return None

    # ---- 2. Create temp image directory --------------------------------- #
    with tempfile.TemporaryDirectory(prefix=f'orthotrack_{model_name}_') as tmp_dir:
        tmp_img_dir = Path(tmp_dir) / 'images'
        tmp_img_dir.mkdir()

        for frame in gt_data['frames']:
            src = data_dir / sequence / frame['image_path']
            dst = tmp_img_dir / src.name
            shutil.copy2(str(src), str(dst))

        poses_file = Path(tmp_dir) / 'poses.txt'
        script    = str(repo_p / model_config['subprocess_script'])
        extra_args = model_config.get('subprocess_extra_args', [])
        conda_bin  = Path(os.environ.get('CONDA_PREFIX', '')).parent.parent / 'bin' / 'conda'
        if not conda_bin.exists():
            conda_bin = 'conda'

        # ---- 3. Build and run subprocess -------------------------------- #
        if model_name in ('vggt_slam_v1', 'vggt_slam_v2'):
            cmd = [
                str(conda_bin), 'run', '-n', conda_env,
                'python', script,
                '--image_folder', str(tmp_img_dir),
                '--log_results',
                '--log_path', str(poses_file),
                '--skip_dense_log',
            ] + extra_args
        elif model_name == 'vggt_long':
            # VGGT-Long auto-derives its save_dir from image_dir + datetime.
            # We find camera_poses.txt after execution by scanning {repo}/exps/.
            cmd = [
                str(conda_bin), 'run', '-n', conda_env,
                'python', script,
                '--image_dir', str(tmp_img_dir),
            ] + extra_args
            poses_file = None   # determined after subprocess completes
        else:
            print(f"  [ERROR] Unknown subprocess model: {model_name}")
            return None

        print(f"  Running subprocess: {' '.join(cmd[:6])} ...")
        try:
            result = subprocess.run(
                cmd,
                cwd=str(repo_p),
                capture_output=False,   # allow real-time output
                timeout=3600,           # 1h max
            )
            if result.returncode != 0:
                print(f"  [ERROR] Subprocess for '{model_name}' exited with code {result.returncode}")
                return None
        except subprocess.TimeoutExpired:
            print(f"  [ERROR] Subprocess for '{model_name}' timed out.")
            return None
        except FileNotFoundError as e:
            print(f"  [ERROR] Could not launch subprocess: {e}")
            return None

        # ---- 4. Parse output -------------------------------------------- #
        # For VGGT-Long: locate camera_poses.txt from auto-derived save_dir.
        if model_name == 'vggt_long':
            exps_dir = repo_p / 'exps'
            candidates = sorted(exps_dir.rglob('camera_poses.txt'), key=lambda p: p.stat().st_mtime)
            if not candidates:
                print(f"  [ERROR] camera_poses.txt not found under {exps_dir}")
                return None
            poses_file = candidates[-1]   # most recently written
            print(f"  VGGT-Long poses file: {poses_file}")

        if not poses_file or not poses_file.exists():
            print(f"  [ERROR] Expected pose file not found: {poses_file}")
            return None

        fmt = model_config.get('subprocess_output_format', 'slam_tum')
        if fmt == 'slam_tum':
            return _parse_slam_tum_poses(poses_file)
        elif fmt == 'vggt_long_txt':
            return _parse_vggt_long_poses(poses_file)
        else:
            print(f"  [ERROR] Unknown subprocess_output_format: {fmt}")
            return None


# --------------------------------------------------------------------------- #
#  Model loading                                                               #
# --------------------------------------------------------------------------- #

def _load_model(model_name: str, device: str = 'cuda'):
    """
    Load a model by name.  Results are cached in _model_cache so that
    subsequent calls within the same subprocess (e.g. multiple windows)
    return the already-loaded model without reloading weights.

    MapAnything family ('mapanything'):
        Loaded from HuggingFace hub (facebook/map-anything) because
        from_pretrained fails with uniception 0.1.4 due to unsupported encoder
        config keys. We construct via model_factory + load state dict manually.

    External models with a Hydra config (dust3r, mast3r, vggt, …):
        Hydra composes the config (using machine=default to pick up local
        checkpoint paths) and passes it to init_model.

    External models without a Hydra config (fallback):
        Direct model_factory call."""
    cache_key = (model_name, device)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    import hydra
    from mapanything.models import init_model

    model_config = MODEL_CONFIGS.get(model_name, {})

    # ------------------------------------------------------------------ #
    #  Pi3X (custom wrapper — not registered in MapAnything v1.1)        #
    # ------------------------------------------------------------------ #
    if model_name == 'pi3x':
        model = _Pi3XWrapper(torch_hub_force_reload=False).to(device).eval()
        _model_cache[cache_key] = model
        return model

    # ------------------------------------------------------------------ #
    #  DA3 (Depth Anything 3) — loads directly from depth_anything_3     #
    # ------------------------------------------------------------------ #
    if model_name in ('da3', 'da3_nested'):
        da3_variant = model_config.get('da3_model', 'da3-large')
        model = _load_da3_model(da3_model=da3_variant, device=device)
        _model_cache[cache_key] = model
        return model

    # ------------------------------------------------------------------ #
    #  Subprocess models — no model object needed at this stage          #
    # ------------------------------------------------------------------ #
    if model_config.get('subprocess'):
        # Return a sentinel so run_model() knows to use the subprocess path
        return None

    model_config_path = _MA_CONFIG_DIR / 'model' / f'{model_name}.yaml'

    # ------------------------------------------------------------------ #
    #  MapAnything (HuggingFace)                                          #
    # ------------------------------------------------------------------ #
    if model_name in ('mapanything', 'mapanything_ablations'):
        hf_name = 'facebook/map-anything' if model_name == 'mapanything' \
                  else 'facebook/map-anything-apache'

        from huggingface_hub import hf_hub_download
        from mapanything.models import model_factory
        import json as _json

        config_path = hf_hub_download(repo_id=hf_name, filename='config.json')
        with open(config_path) as f:
            hf_config = _json.load(f)

        # Strip encoder keys unsupported by installed uniception version
        for k in ('norm_returned_features', 'keep_first_n_layers', 'torch_hub_force_reload'):
            hf_config.get('encoder_config', {}).pop(k, None)

        # Remove mlp_layer string that uniception expects as a callable
        hf_config.get('info_sharing_config', {}).get('module_args', {}).pop('mlp_layer', None)

        # Keep only accepted MapAnything constructor params
        accepted = {
            'name', 'encoder_config', 'info_sharing_config', 'pred_head_config',
            'geometric_input_config', 'fusion_norm_layer',
            'pretrained_checkpoint_path', 'load_specific_pretrained_submodules',
            'specific_pretrained_submodules',
        }
        hf_config = {k: v for k, v in hf_config.items() if k in accepted}

        print(f"  Building MapAnything from HF config...")
        model = model_factory('mapanything', torch_hub_force_reload=False, **hf_config)
        _replace_mlp_with_swiglu(model)

        for ckpt_name in ('model.safetensors', 'checkpoint.pth'):
            try:
                ckpt_path = hf_hub_download(repo_id=hf_name, filename=ckpt_name)
                if ckpt_name.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    sd = load_file(ckpt_path)
                else:
                    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                    sd = raw.get('model', raw.get('state_dict', raw))
                res = model.load_state_dict(sd, strict=False)
                print(f"  Loaded weights from {hf_name}/{ckpt_name}")
                if res.missing_keys:
                    print(f"    Missing keys: {len(res.missing_keys)}")
                break
            except Exception as e:
                print(f"  Warning: could not load {ckpt_name}: {e}")

        model = model.to(device).eval()
        _model_cache[cache_key] = model
        return model

    # ------------------------------------------------------------------ #
    #  External models resolved via Hydra                                 #
    # ------------------------------------------------------------------ #
    elif model_config_path.exists() or any(
        (_MA_CONFIG_DIR / 'model' / f"{MODEL_CONFIGS.get(model_name, {}).get('config_override', '')}.yaml").exists()
        for _ in [0]
    ):
        # Use config_override name if present (e.g. modular_dust3r → modular_dust3r_512_dpt)
        hydra_model_name = MODEL_CONFIGS.get(model_name, {}).get('config_override', model_name)
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with hydra.initialize_config_dir(version_base=None, config_dir=str(_MA_CONFIG_DIR)):
            cfg = hydra.compose(
                config_name='dense_n_view_benchmark',
                overrides=[
                    f'model={hydra_model_name}',
                    'machine=default',
                    # Provide placeholder values for mandatory machine settings so that
                    # models referencing checkpoint paths don't cause MissingMandatoryValue
                    # errors during OmegaConf resolution.
                    f'root_pretrained_checkpoints_dir={_CHECKPOINTS_DIR}',
                    f'root_uniception_pretrained_checkpoints_dir={_CHECKPOINTS_DIR}',
                    # Some encoder configs reference machine.* directly (not via top-level key).
                    f'machine.root_pretrained_checkpoints_dir={_CHECKPOINTS_DIR}',
                    f'machine.root_uniception_pretrained_checkpoints_dir={_CHECKPOINTS_DIR}',
                    'root_data_dir=/tmp',
                    'mapanything_dataset_metadata_dir=/tmp',
                    'root_experiments_dir=/tmp',
                    'machine.root_data_dir=/tmp',
                    'machine.mapanything_dataset_metadata_dir=/tmp',
                    'machine.root_experiments_dir=/tmp',
                    'machine.external_benchmark_data_root_data_dir=/tmp',
                ],
            )
        print(f"  Loading via Hydra config: {hydra_model_name} (requested: {model_name})")
        model = init_model(
            cfg.model.model_str,
            cfg.model.model_config,
            torch_hub_force_reload=cfg.model.get('torch_hub_force_reload', False),
        )
        model = model.to(device).eval()
        _model_cache[cache_key] = model
        return model

    # ------------------------------------------------------------------ #
    #  Fallback                                                           #
    # ------------------------------------------------------------------ #
    else:
        from mapanything.models import model_factory
        print(f"  Loading via model_factory (no Hydra config found for {model_name})")
        model = model_factory(model_name, name=model_name, torch_hub_force_reload=False)
        model = model.to(device).eval()
        _model_cache[cache_key] = model
        return model


# --------------------------------------------------------------------------- #
#  Per-sequence runner                                                         #
# --------------------------------------------------------------------------- #

def run_model(
    model_name: str,
    gt_data: dict,
    data_dir: Path,
    sequence: str,
    output_dir: Path,
    device: str = 'cuda',
    provide_intrinsics: bool = False,
    provide_depth: bool = False,
    provide_poses: bool = False,
    memory_efficient: bool = True,
    frame_indices: 'np.ndarray | None' = None,
):
    run_name = f"{model_name}{get_suffix(provide_intrinsics, provide_depth, provide_poses)}"
    out_path = output_dir / sequence / run_name

    if (out_path / 'predictions.npz').exists():
        print(f"  Skipping {run_name} (already exists)")
        return

    print(f"\n--- Running {run_name} on {sequence} ---")
    model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['mapanything'])

    # Warn about pairwise-only models (pow3r bare requires exactly 2 views)
    if model_config.get('pairwise') and len(gt_data['frames']) != 2:
        print(f"  [WARNING] {model_name} is pairwise-only (requires exactly 2 views).")
        print(f"  Got {len(gt_data['frames'])} frames. Run with --max_frames 2 or use {model_name}_ba instead.")
        return

    t0 = time.time()

    # ------------------------------------------------------------------ #
    #  Path 1: subprocess SLAM models (vggt_slam_v1/v2, vggt_long)       #
    # ------------------------------------------------------------------ #
    if model_config.get('subprocess'):
        pose_results = _run_subprocess_model(
            model_name, gt_data, data_dir, sequence, model_config, device=device
        )
        if pose_results is None:
            return   # setup error already printed
        inf_time = time.time() - t0

        # --- Map SLAM keyframe frame_ids to correct GT frame_indices --- #
        # SLAM models (VGGT-SLAM) may select keyframes internally, so the
        # number of output poses can be < number of input images.  The
        # returned frame_ids are numbers extracted from image filenames
        # (e.g. frame_0030.png → 30).  We build a mapping from those
        # numbers back to offset positions in gt_data['frames'] and then
        # to entries in frame_indices.
        slam_frame_indices = frame_indices  # default: use full array
        if 'frame_ids' in pose_results:
            import re as _re
            # Build lookup: filename_number → position in gt_data['frames']
            fname_num_to_pos = {}
            for pos_i, frame in enumerate(gt_data['frames']):
                m = _re.search(r'\d+', Path(frame['image_path']).stem)
                if m:
                    fname_num_to_pos[int(m.group())] = pos_i
            # Map each SLAM frame_id to the correct index
            matched_indices = []
            for fid in pose_results['frame_ids']:
                pos_i = fname_num_to_pos.get(int(fid))
                if pos_i is not None:
                    if frame_indices is not None:
                        matched_indices.append(frame_indices[pos_i])
                    else:
                        matched_indices.append(pos_i)
                else:
                    print(f"  [WARNING] SLAM frame_id {fid} not found in gt_data frames")
            if matched_indices:
                slam_frame_indices = np.array(matched_indices, dtype=int)
                print(f"  SLAM keyframes: {len(slam_frame_indices)}/{len(gt_data['frames'])} frames matched")
            else:
                print(f"  [WARNING] No SLAM frame_ids matched gt_data — using full frame_indices")

        results = {
            'model': run_name,
            'n_views': len(gt_data['frames']),
            'positions': pose_results['positions'],
            'rotations': pose_results['rotations'],
            'depths': None,
            'pts3d': None,
            'inference_time': inf_time,
            'frame_indices': slam_frame_indices,
        }
        print(f"  Subprocess completed in {inf_time:.1f}s, {len(gt_data['frames'])} frames")
        save_results(results, out_path)
        return

    # ------------------------------------------------------------------ #
    #  Path 2: DA3 — use direct file paths, no prepare_views             #
    # ------------------------------------------------------------------ #
    if model_name in ('da3', 'da3_nested'):
        try:
            model = _load_da3_model(
                da3_model=model_config.get('da3_model', 'da3-large'),
                device=device,
            )
        except Exception as e:
            print(f"  Failed to load {model_name}: {e}")
            import traceback; traceback.print_exc()
            return
        print(f"  Model loaded in {time.time()-t0:.1f}s")

        t0 = time.time()
        print(f"  Running DA3 inference on {len(gt_data['frames'])} images...")
        try:
            pose_results = _run_da3_inference(model, gt_data, data_dir, sequence)
        except Exception as e:
            print(f"  DA3 inference failed: {e}")
            import traceback; traceback.print_exc()
            return
        inf_time = time.time() - t0
        n = len(gt_data['frames'])
        print(f"  Inference completed in {inf_time:.1f}s ({n/inf_time:.1f} fps)")

        results = {
            'model': run_name,
            'n_views': n,
            'positions': pose_results['positions'],
            'rotations': pose_results['rotations'],
            'depths': None,
            'pts3d': None,
            'inference_time': inf_time,
            'frame_indices': frame_indices,
        }
        save_results(results, out_path)
        del model
        torch.cuda.empty_cache()
        return

    # ------------------------------------------------------------------ #
    #  Path 3: standard MapAnything / Pi3X / VGGT / DUSt3R-family        #
    # ------------------------------------------------------------------ #
    try:
        model = _load_model(model_name, device=device)
    except Exception as e:
        print(f"  Failed to load {model_name}: {e}")
        import traceback; traceback.print_exc()
        return
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    print(f"  Preparing {len(gt_data['frames'])} views...")
    views = prepare_views(
        gt_data, data_dir, sequence, model_config,
        model_name=model_name,
        provide_intrinsics=provide_intrinsics,
        provide_depth=provide_depth,
        provide_poses=provide_poses,
        device=device,
    )
    # Add Pow3R-required key aliases (camera_intrinsics, depthmap, camera_pose).
    # Only for pow3r models — MapAnything rejects these keys via strict validation.
    if model_name in ('pow3r', 'pow3r_ba'):
        views = _normalize_view_keys(views)

    t0 = time.time()
    print(f"  Running inference (memory_efficient={memory_efficient})...")
    predictions = run_inference(model, views, run_name, memory_efficient, device=device)
    if predictions is None:
        print(f"  Inference returned no results for {run_name} — skipping save.")
        del model, views
        torch.cuda.empty_cache()
        return
    inf_time = time.time() - t0
    n = len(gt_data['frames'])
    print(f"  Inference completed in {inf_time:.1f}s ({n/inf_time:.1f} fps)")

    results = extract_results(predictions, run_name)
    results['inference_time'] = inf_time
    results['frame_indices'] = frame_indices
    save_results(results, out_path)

    del model, predictions, views
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description='Run foundation models on extracted MovingDrone frames.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--sequence', default=None,
                        help='Single sequence name (e.g. potsdamerplatz1). '
                             'Use --sequences for multiple sequences.')
    parser.add_argument('--sequences', nargs='+', default=None,
                        help='One or more sequence names.  Processes each '
                             'sequentially with the model kept in memory.  '
                             'Ideal for windowed inference (load model once, '
                             'process all windows).')
    parser.add_argument('--models', nargs='+', default=['mapanything'],
                        metavar='MODEL',
                        help=f'Models to run. Available: {list(MODEL_CONFIGS.keys())}')
    parser.add_argument('--data_dir',   default='outputs/foundation_comparison',
                        help='Directory containing extracted frames')
    parser.add_argument('--output_dir', default='outputs/foundation_comparison',
                        help='Output root directory for predictions')
    parser.add_argument('--device',     default='cuda')
    parser.add_argument('--provide_intrinsics', action='store_true',
                        help='Provide GT camera intrinsics (mapanything → _K suffix)')
    parser.add_argument('--provide_depth',      action='store_true',
                        help='Provide GT depth maps (mapanything → _K+D suffix, requires --provide_intrinsics)')
    parser.add_argument('--provide_poses',      action='store_true',
                        help='Provide GT camera poses — oracle upper bound (→ _K+D+P suffix)')
    parser.add_argument('--memory_efficient',   action='store_true', default=True)
    parser.add_argument('--no_memory_efficient', action='store_false', dest='memory_efficient')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Subsample to at most N frames (uniform). '
                             'Useful for local testing on small GPUs. '
                             'DUSt3R/MASt3R need 48 GB for 30 frames; ~12 GB fits 10 frames.')
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Resolve sequence list: --sequences overrides --sequence
    if args.sequences:
        sequences = args.sequences
    elif args.sequence:
        sequences = [args.sequence]
    else:
        import sys as _sys
        print("ERROR: provide --sequence or --sequences", file=_sys.stderr)
        _sys.exit(1)

    # Warn about unknown models
    valid_models = []
    for m in args.models:
        if m not in MODEL_CONFIGS:
            print(f"[WARNING] Unknown model '{m}', skipping. Available: {list(MODEL_CONFIGS.keys())}")
        else:
            valid_models.append(m)
    args.models = valid_models

    # Model ordering: metric_dust3r must run BEFORE must3r (same process).
    # must3r imports its own dust3r submodule which changes sys.modules and
    # causes "Attention.forward() missing xpos" in metric_dust3r afterwards.
    _ORDER_BEFORE = {'metric_dust3r': 'must3r'}
    for early, late in _ORDER_BEFORE.items():
        if early in args.models and late in args.models:
            ei, li = args.models.index(early), args.models.index(late)
            if ei > li:
                print(f"[INFO] Reordering: '{early}' moved before '{late}' to avoid module conflict.")
                args.models.remove(early)
                args.models.insert(li, early)

    # Process each sequence (windows share the model via _model_cache)
    for seq in sequences:
        gt_data = load_ground_truth(data_dir, seq)
        print(f"Loaded GT for {seq}: {len(gt_data['frames'])} frames")

        frame_indices = None
        if args.max_frames and len(gt_data['frames']) > args.max_frames:
            frame_indices = np.linspace(0, len(gt_data['frames']) - 1, args.max_frames, dtype=int)
            gt_data['frames'] = [gt_data['frames'][i] for i in frame_indices]
            print(f"  Subsampled to {len(gt_data['frames'])} frames")

        for model_name in args.models:
            run_model(
                model_name=model_name,
                gt_data=gt_data,
                data_dir=data_dir,
                sequence=seq,
                output_dir=output_dir,
                device=args.device,
                provide_intrinsics=args.provide_intrinsics,
                provide_depth=args.provide_depth,
                provide_poses=args.provide_poses,
                memory_efficient=args.memory_efficient,
                frame_indices=frame_indices,
            )

    print("\nAll models completed.")


if __name__ == '__main__':
    main()
