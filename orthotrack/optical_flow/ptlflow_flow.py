"""
PTLFlow Optical Flow Wrapper

Unified wrapper around the ptlflow library (https://github.com/hmorimitsu/ptlflow)
which provides 80+ pretrained optical flow models under a common API.

This replaces per-model wrappers (like waft_flow.py) with a single class that
can load any model by name: RAFT, FlowFormer, GMA, SEA-RAFT, WAFT, Flow-Anything, etc."""

import gc
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


# Default models for ablation study — curated for quality vs speed tradeoffs
# Maps model_name -> preferred checkpoint name
# Only includes models verified to work with pretrained weights (tested Feb 2025)
RECOMMENDED_MODELS = {
    # --- Classic / Baseline ---
    'raft': 'things',              # Baseline RAFT (ECCV 2020)
    'raft_small': 'things',        # Lightweight RAFT variant
    'gma': 'things',               # GMA — global motion aggregation (ICCV 2021)
    # --- Transformer-based ---
    'craft': 'things',             # CRAFT — cross-attentional flow transformer (ECCV 2022)
    'skflow': 'things',            # SK-Flow — shift kernel flow (NeurIPS 2022)
    # --- Global matching ---
    'gmflow': 'things',            # GMFlow — global matching (CVPR 2022)
    'gmflow_p': 'things',          # GMFlow+ (TPAMI 2024)
    'unimatch': 'things',          # UniMatch — unified matching (TPAMI 2023)
    # --- SEA-RAFT family (ECCV 2024) ---
    'sea_raft_s': 'things',        # SEA-RAFT small — efficient
    'sea_raft_m': 'things',        # SEA-RAFT medium
    'sea_raft_l': 'things',        # SEA-RAFT large
    # --- Speed-optimized ---
    'neuflow2': 'things',          # NeuFlow v2 — lightweight (2024)
    'rapidflow': 'things',         # RAPIDFlow — speed-optimized (ICRA 2024)
    # --- Recent / Advanced ---
    'rpknet': 'things',            # RPKNet (AAAI 2024)
    'dpflow': 'things',            # DPFlow (2025)
    'ccmr': 'sintel',              # CCMR — cross-scale cost aggregation (2023)
    'waft': 'things',              # WAFT — warping-based flow transformer (2024)
    'flow_anything': 'mixed288',   # Flow-Anything — foundation flow model (2025)
    'memflow': 'things',           # MemFlow — memory-augmented (2024)
    'memflow_t': 'things',         # MemFlow with Twins backbone
    'videoflow_bof': 'things',     # VideoFlow bidirectional (ICCV 2023)
    'ms_raft_p': 'mixed',          # MS-RAFT+ multi-scale (2024)
    # --- Other verified methods ---
    'gmflownet': 'things',         # GMFlowNet (ECCV 2022)
    'scopeflow': 'things',         # ScopeFlow (ECCV 2020)
    'maskflownet_s': 'things',     # MaskFlowNet-S (2020)
    'starflow': 'things',          # STaRFlow (2020)
    'llaflow': 'things',           # LLA-Flow (CVPR 2022)
    'llaflow_raft': 'things',      # LLA-Flow (RAFT backbone)
    'dip': 'things',               # DIP (2022)
    'flow1d': 'things',            # Flow1D (ICCV 2021)
    'csflow': 'things',            # CSFlow (2022)
}

# Models that require 3+ frames input (not compatible with 2-frame tracking)
_MULTI_FRAME_MODELS = {'memfof'}

# Models without any pretrained checkpoints
_NO_CHECKPOINT_MODELS = {'lcv_raft_small', 'sea_raft'}

# Models that fail to load due to missing CUDA extensions or incompatible weights.
# These require special compilation steps not available in standard environments.
_BROKEN_MODELS = {
    'matchflow', 'matchflow_raft',  # Requires QuadTreeAttention CUDA extension
    'scv4', 'scv8',                 # Requires torch_scatter
    'separableflow',                # Requires GANet CUDA extension
    'splatflow',                    # Requires cupy
    'vcn', 'vcn_small',             # Incompatible checkpoint format
    'rapidflow_it1', 'rapidflow_it2',  # State dict shape mismatch (ptlflow bug)
}

# Models with resolution-sensitive positional encodings.
# FlowFormer/FlowFormer++ fail at non-standard resolutions (< ~960px width).
# Use full resolution (960px+) or avoid these for low-res inputs.
_RESOLUTION_SENSITIVE_MODELS = {'flowformer', 'flowformer_pp'}

# Models that fail at lower resolutions due to architecture constraints
_LOW_RES_BROKEN_MODELS = {'streamflow', 'dicl'}


class PTLFlowOpticalFlow:
    """
    Dense optical flow estimator using any model from ptlflow.

    Mirrors the WAFTOpticalFlow interface so it can be used as a drop-in
    replacement in the FeatureTracker."""

    # Models known to be unstable with fp16 (produce NaN/Inf or type mismatches)
    _FP16_BLOCKLIST = {
        'gma', 'flowformer', 'flowformer_pp', 'lcv_raft', 'lcv_raft_small',
        'matchflow', 'matchflow_raft', 'separableflow',
        'waft',          # fp16 type mismatch in cross-attention
        'flow_anything', # fp16 causes resolution alignment errors
        'craft',         # fp16 NaN in attention layers
        'memflow', 'memflow_t',  # fp16 instability in memory module
        'videoflow_bof', 'videoflow_mof',  # fp16 NaN in bidirectional flow
        'streamflow',    # fp16 channel mismatch
        'hd3', 'hd3_ctxt',  # fp16 NaN in correlation
        'irr_pwc', 'irr_pwcnet', 'irr_pwcnet_irr',  # fp16 instability
    }

    def __init__(self, model_name: str = 'raft',
                 pretrained_ckpt: str = 'things',
                 device: str = 'cuda',
                 fp16: bool = True):
        """
        Args:
            model_name: ptlflow model name (e.g. 'raft', 'flowformer', 'gma').
            pretrained_ckpt: Pretrained checkpoint name (e.g. 'things', 'sintel',
                             'kitti'). Use 'things' for general-purpose flow.
            device: Torch device.
            fp16: Use half-precision inference for speed/memory. Auto-disabled
                  for models known to be unstable with fp16."""
        import ptlflow

        self.model_name = model_name
        self.device = device
        self._on_gpu = True

        # Check if model is multi-frame (not supported for 2-frame tracking)
        if model_name in _MULTI_FRAME_MODELS:
            raise ValueError(
                f"{model_name} requires 3+ frames and is not supported for "
                f"2-frame optical flow. Use a 2-frame model instead."
            )

        # Check if model has pretrained checkpoints
        if model_name in _NO_CHECKPOINT_MODELS:
            raise ValueError(
                f"{model_name} has no pretrained checkpoints available. "
                f"Use a variant with checkpoints (e.g. sea_raft_s instead of sea_raft)."
            )

        # Check if model requires special CUDA extensions
        if model_name in _BROKEN_MODELS:
            raise ValueError(
                f"{model_name} requires CUDA extensions that are not installed. "
                f"See _BROKEN_MODELS in ptlflow_flow.py for details."
            )

        # Warn about resolution-sensitive models
        if model_name in _RESOLUTION_SENSITIVE_MODELS:
            print(f"[PTLFlow] WARNING: {model_name} has fixed positional encodings "
                  f"and may fail at non-standard resolutions. Use 960px+ input width.")

        # Disable fp16 for known-problematic models
        if fp16 and model_name in self._FP16_BLOCKLIST:
            print(f"[PTLFlow] Disabling fp16 for {model_name} (known incompatible)")
            fp16 = False
        self.fp16 = fp16

        # Use recommended checkpoint if available, otherwise use provided
        if pretrained_ckpt == 'things' and model_name in RECOMMENDED_MODELS:
            pretrained_ckpt = RECOMMENDED_MODELS[model_name]

        # Auto-discover checkpoint if specified one is not available
        ref = ptlflow.get_model_reference(model_name)
        available_ckpts = list(ref.pretrained_checkpoints.keys()) if hasattr(ref, 'pretrained_checkpoints') else []
        if pretrained_ckpt not in available_ckpts and available_ckpts:
            # Prefer 'things' > 'sintel' > 'kitti' > first available
            for fallback in ['things', 'sintel', 'kitti', 'mixed', 'mix', 'chairs']:
                if fallback in available_ckpts:
                    pretrained_ckpt = fallback
                    break
            else:
                pretrained_ckpt = available_ckpts[0]
            print(f"[PTLFlow] Auto-selected checkpoint: {pretrained_ckpt} "
                  f"(available: {available_ckpts})")

        # Load model with pretrained weights (auto-downloads)
        print(f"[PTLFlow] Loading {model_name} (ckpt={pretrained_ckpt})...")
        self.model = ptlflow.get_model(model_name, ckpt_path=pretrained_ckpt)
        self.model = self.model.to(device).eval()

        if fp16:
            self.model = self.model.half()

        print(f"[PTLFlow] {model_name} ready on {device} (fp16={fp16})")

    def offload_to_cpu(self):
        """Move model to CPU to free GPU memory (e.g. for keyframe matching)."""
        self.model = self.model.cpu()
        self._on_gpu = False
        gc.collect()
        torch.cuda.empty_cache()
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        print(f"[PTLFlow] {self.model_name} offloaded to CPU. GPU mem: {mem_alloc:.2f} GB")

    def reload_to_gpu(self):
        """Move model back to GPU after offload."""
        if not self._on_gpu:
            self.model = self.model.to(self.device)
            self._on_gpu = True

    def _ensure_gpu(self):
        """Auto-reload to GPU if offloaded."""
        if not self._on_gpu:
            self.reload_to_gpu()

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert (H,W,3) uint8/float32 RGB to (1,3,H,W) float tensor in [0,1].

        ptlflow models expect input in [0, 1] range (their ToTensor divides by 255)."""
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            # Already float but in [0, 255] range — normalize
            image = image / 255.0
        t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        if self.fp16:
            t = t.half()
        return t.to(self.device)

    # Maximum long-edge resolution for flow inference.  Images larger than
    # this are downscaled before running flow and the resulting flow field is
    # upscaled back to the original resolution with proportionally-adjusted
    # flow vectors.  This avoids OOM on high-res inputs (e.g. 5280×3956 UAVD4L
    # images would need ~200 GB for RAFT's correlation volume at full res).
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

        # Downscale large images to avoid OOM (RAFT correlation volume is O(H*W*H*W))
        if long_edge > self.MAX_FLOW_LONG_EDGE:
            scale = self.MAX_FLOW_LONG_EDGE / long_edge
            new_W = int(W_orig * scale) // 8 * 8  # ensure divisible by 8
            new_H = int(H_orig * scale) // 8 * 8
            image1 = cv2.resize(image1, (new_W, new_H), interpolation=cv2.INTER_AREA)
            image2 = cv2.resize(image2, (new_W, new_H), interpolation=cv2.INTER_AREA)

        img1 = self._to_tensor(image1)  # (1, 3, H, W)
        img2 = self._to_tensor(image2)

        # ptlflow expects dict with 'images' key: (B, 2, C, H, W)
        images = torch.stack([img1[:, :3], img2[:, :3]], dim=1)
        inputs = {'images': images}

        preds = self.model(inputs)

        # Output: 'flows' is (B, N_iters, 2, H, W) or list of (B, 2, H, W)
        flow = preds['flows']
        if isinstance(flow, (list, tuple)):
            flow = flow[-1]  # finest / final iteration
        # Handle 5D tensor (B, N_iters, 2, H, W) — take last iteration
        while flow.dim() > 4:
            flow = flow[:, -1]
        # Now flow is (B, 2, H, W)
        flow_np = flow[0].float().permute(1, 2, 0).cpu().numpy()  # (H, W, 2)

        # Upscale flow back to original resolution if we downscaled, or if the
        # model padded/resized internally.
        if flow_np.shape[0] != H_orig or flow_np.shape[1] != W_orig:
            flow_t = torch.from_numpy(flow_np).permute(2, 0, 1).unsqueeze(0)
            flow_t = F.interpolate(flow_t, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
            # Scale flow values proportionally to match original pixel coordinates
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

        # Bilinear interpolation
        flow_at_pts = (flow[y0, x0] * (1 - fx) * (1 - fy) +
                       flow[y0, x1] * fx * (1 - fy) +
                       flow[y1, x0] * (1 - fx) * fy +
                       flow[y1, x1] * fx * fy)

        return pts_2d + flow_at_pts




