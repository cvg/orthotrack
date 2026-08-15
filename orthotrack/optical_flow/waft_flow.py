"""
WAFT Optical Flow Wrapper

Loads and runs WAFT (Warping-Alone Field Transforms) for dense optical flow
estimation. Replaces Lucas-Kanade optical flow in the tracking pipeline for
better accuracy on large inter-frame motions (~100+ pixel displacements)."""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F

import gc

# WAFT root directory
_WAFT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'thirdparty', 'WAFT')


def _import_waft():
    """
    Import WAFT modules. Handles sys.path + sys.modules carefully to avoid
    collisions with our own 'utils' and 'model' packages.
    
    Returns:
        (fetch_model, load_ckpt, InferenceWrapper) callables + waft module dict"""
    # Save and temporarily remove conflicting modules
    saved_modules = {}
    for prefix in ('utils', 'model', 'config'):
        for key in list(sys.modules.keys()):
            if key == prefix or key.startswith(prefix + '.'):
                saved_modules[key] = sys.modules.pop(key)
    
    # Remove project root from sys.path to avoid finding our utils/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_backup = sys.path.copy()
    sys.path = [_WAFT_DIR] + [p for p in sys.path
                               if os.path.abspath(p) != project_root and p != '.']
    
    try:
        from model import fetch_model
        from inference_tools import InferenceWrapper
        from utils.utils import load_ckpt
        
        # Collect all WAFT-imported modules to keep them alive
        waft_modules = {}
        for prefix in ('utils', 'model', 'config', 'inference_tools',
                        'thirdparty', 'criterion'):
            for key in list(sys.modules.keys()):
                if key == prefix or key.startswith(prefix + '.'):
                    waft_modules[key] = sys.modules[key]
    finally:
        # Restore sys.path
        sys.path = path_backup
        
        # Remove WAFT modules from sys.modules
        for key in list(sys.modules.keys()):
            for prefix in ('utils', 'model', 'config'):
                if key == prefix or key.startswith(prefix + '.'):
                    del sys.modules[key]
        
        # Restore our original modules
        sys.modules.update(saved_modules)
    
    return fetch_model, load_ckpt, InferenceWrapper, waft_modules


class _WaftContext:
    """
    Context manager that temporarily swaps sys.modules for WAFT forward pass.
    WAFT model's forward() internally calls functions from its own 'utils' module,
    so we need the WAFT modules available during execution."""
    def __init__(self, waft_modules):
        self.waft_modules = waft_modules
    
    def __enter__(self):
        self.saved = {}
        for prefix in ('utils', 'model', 'config', 'inference_tools',
                        'thirdparty', 'criterion'):
            for key in list(sys.modules.keys()):
                if key == prefix or key.startswith(prefix + '.'):
                    self.saved[key] = sys.modules.pop(key)
        
        self.path_backup = sys.path.copy()
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path = [_WAFT_DIR] + [p for p in sys.path
                    if os.path.abspath(p) != project_root and p != '.']
        sys.modules.update(self.waft_modules)
        return self
    
    def __exit__(self, *args):
        # Save updated WAFT modules back
        for key in list(self.waft_modules.keys()):
            if key in sys.modules:
                self.waft_modules[key] = sys.modules[key]
        
        sys.path = self.path_backup
        for prefix in ('utils', 'model', 'config', 'inference_tools',
                        'thirdparty', 'criterion'):
            for key in list(sys.modules.keys()):
                if key == prefix or key.startswith(prefix + '.'):
                    del sys.modules[key]
        sys.modules.update(self.saved)


class WAFTOpticalFlow:
    """
    WAFT-based dense optical flow estimator.
    
    Replaces Lucas-Kanade optical flow with a learned dense flow model that
    handles large motions much better (~100+ pixel displacements)."""
    
    _default_config = 'config/a1/chairs-things.json'
    _default_ckpt = 'checkpoints/waft/waft_a1.pth'
    
    def __init__(self, checkpoint_path: str = None, config_path: str = None,
                 device: str = 'cuda', scale: float = 0.0):
        """
        Args:
            checkpoint_path: Path to WAFT .pth checkpoint.
            config_path: Path to WAFT config JSON.
            device: torch device
            scale: Resolution scale for inference (0 = native)"""
        self.device = device
        self.scale = scale
        
        # Resolve paths
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if checkpoint_path is None:
            checkpoint_path = os.path.join(project_root, self._default_ckpt)
        if config_path is None:
            config_path = os.path.join(_WAFT_DIR, self._default_config)
        
        # Load config
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        args = argparse.Namespace(**cfg)
        
        # Import WAFT
        fetch_model, load_ckpt, InferenceWrapper, self._waft_modules = _import_waft()
        self._ctx = _WaftContext(self._waft_modules)
        
        # Build model (needs WAFT modules on sys.path)
        with self._ctx:
            # depth-anything-ckpts must be found relative to WAFT dir
            original_cwd = os.getcwd()
            os.chdir(_WAFT_DIR)
            try:
                model = fetch_model(args)
                load_ckpt(model, checkpoint_path)
                model = model.to(device).eval()
                
                self.model = InferenceWrapper(
                    model, scale=scale,
                    train_size=args.image_size,
                    pad_to_train_size=False,
                    tiling=False
                )
            finally:
                os.chdir(original_cwd)
        
        self._on_gpu = True
        print(f"[WAFT] Loaded optical flow model from {os.path.basename(checkpoint_path)}")
        print(f"[WAFT] Device: {device}, Scale: {scale}")
    
    def offload_to_cpu(self):
        """Move WAFT model to CPU to free GPU memory (e.g. for RoMaV2 keyframe matching)."""
        if hasattr(self.model, 'model'):
            self.model.model = self.model.model.cpu()
        self._on_gpu = False
        gc.collect()
        torch.cuda.empty_cache()
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        print(f"[WAFT] Offloaded to CPU. GPU memory allocated: {mem_alloc:.2f} GB")
    
    def reload_to_gpu(self):
        """Move WAFT model back to GPU after offload."""
        if hasattr(self.model, 'model') and not self._on_gpu:
            self.model.model = self.model.model.to(self.device)
            self._on_gpu = True
    
    def _ensure_gpu(self):
        """Auto-reload to GPU if offloaded. Called before every inference."""
        if not self._on_gpu:
            self.reload_to_gpu()

    # Maximum long-edge resolution for flow inference.  Images larger than
    # this are downscaled before running flow and the result is upscaled back.
    MAX_FLOW_LONG_EDGE: int = 1280

    @torch.no_grad()
    def predict(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        Compute dense optical flow from image1 to image2.
        
        Args:
            image1: (H, W, 3) RGB uint8 or float32 [0-255]
            image2: (H, W, 3) RGB uint8 or float32 [0-255]
            
        Returns:
            flow: (H, W, 2) optical flow (dx, dy) in pixels at original resolution"""
        self._ensure_gpu()

        H_orig, W_orig = image1.shape[:2]
        long_edge = max(H_orig, W_orig)

        # Downscale large images to avoid OOM
        if long_edge > self.MAX_FLOW_LONG_EDGE:
            import cv2
            scale = self.MAX_FLOW_LONG_EDGE / long_edge
            new_W = int(W_orig * scale) // 8 * 8
            new_H = int(H_orig * scale) // 8 * 8
            image1 = cv2.resize(image1, (new_W, new_H), interpolation=cv2.INTER_AREA)
            image2 = cv2.resize(image2, (new_W, new_H), interpolation=cv2.INTER_AREA)

        img1 = self._to_tensor(image1)
        img2 = self._to_tensor(image2)
        
        with self._ctx:
            output = self.model.calc_flow(img1, img2)
            flow = output['flow'][-1]  # Last (finest) prediction
        
        flow_np = flow[0].permute(1, 2, 0).cpu().numpy()

        # Upscale flow to original resolution if we downscaled
        if flow_np.shape[0] != H_orig or flow_np.shape[1] != W_orig:
            import torch.nn.functional as Fnn
            flow_t = torch.from_numpy(flow_np).permute(2, 0, 1).unsqueeze(0)
            flow_t = Fnn.interpolate(flow_t, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
            flow_t[:, 0] *= W_orig / flow_np.shape[1]
            flow_t[:, 1] *= H_orig / flow_np.shape[0]
            flow_np = flow_t[0].permute(1, 2, 0).numpy()

        return flow_np
    
    @torch.no_grad()
    def track_points(self, image1: np.ndarray, image2: np.ndarray,
                     pts_2d: np.ndarray) -> np.ndarray:
        """
        Track 2D points from image1 to image2 using dense flow.
        
        Args:
            image1: (H, W, 3) RGB
            image2: (H, W, 3) RGB
            pts_2d: (N, 2) points in image1 as (x, y) pixel coords
            
        Returns:
            new_pts: (N, 2) tracked points in image2"""
        flow = self.predict(image1, image2)
        return self._sample_flow_at_points(flow, pts_2d)
    
    
    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image (H,W,3) to WAFT input tensor (1,3,H,W) in [0,255]."""
        if image.dtype == np.uint8:
            image = image.astype(np.float32)
        return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
    
    @staticmethod
    def _sample_flow_at_points(flow: np.ndarray, pts_2d: np.ndarray) -> np.ndarray:
        """
        Bilinear-interpolate dense flow at sparse point locations.
        
        Args:
            flow: (H, W, 2) dense flow field
            pts_2d: (N, 2) point locations (x, y)
        Returns:
            new_pts: (N, 2) displaced points"""
        H, W = flow.shape[:2]
        x = np.clip(pts_2d[:, 0], 0, W - 1).astype(np.float64)
        y = np.clip(pts_2d[:, 1], 0, H - 1).astype(np.float64)
        
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, W - 1)
        y1 = np.minimum(y0 + 1, H - 1)
        
        fx = (x - x0).reshape(-1, 1)
        fy = (y - y0).reshape(-1, 1)
        
        # Bilinear interp
        flow_at_pts = (flow[y0, x0] * (1 - fx) * (1 - fy) +
                       flow[y0, x1] * fx * (1 - fy) +
                       flow[y1, x0] * (1 - fx) * fy +
                       flow[y1, x1] * fx * fy)
        
        return pts_2d + flow_at_pts


# Singleton
_waft_instance = None

