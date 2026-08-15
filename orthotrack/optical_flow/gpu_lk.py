"""
GPU-accelerated sparse pyramidal Lucas-Kanade optical flow.

Provides two backends:
1. **OpenCV CUDA** (cv2.cuda.SparsePyrLKOpticalFlow) — fastest, requires OpenCV
   built with CUDA support.
2. **PyTorch grid_sample** — pure-PyTorch fallback that runs on any CUDA GPU.

The factory function ``create_gpu_lk()`` auto-selects the best available backend."""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detect OpenCV CUDA availability at import time
# ---------------------------------------------------------------------------
_OPENCV_CUDA_LK_AVAILABLE = False
try:
    import cv2
    # This attribute only exists when OpenCV is compiled with CUDA
    _test = cv2.cuda.SparsePyrLKOpticalFlow.create()
    _OPENCV_CUDA_LK_AVAILABLE = True
    del _test
except (AttributeError, cv2.error, Exception):
    pass




# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_gpu_lk(
    win_size: int = 21,
    max_level: int = 3,
    max_iter: int = 30,
    eps: float = 0.01,
    min_eig_threshold: float = 1e-4,
    backend: str = "auto",
) -> "GPUSparsePyrLKBase":
    """Create the best available GPU LK tracker.

    Args:
        backend: ``"auto"`` (default) picks OpenCV CUDA if available, else
            PyTorch.  ``"opencv_cuda"`` forces OpenCV CUDA (raises if
            unavailable).  ``"pytorch"`` forces the PyTorch backend."""
    if backend == "auto":
        if _OPENCV_CUDA_LK_AVAILABLE:
            logger.info("GPU LK backend: OpenCV CUDA")
            return OpenCVCudaSparsePyrLK(
                win_size=win_size, max_level=max_level,
                max_iter=max_iter, eps=eps,
            )
        else:
            logger.info("GPU LK backend: PyTorch (OpenCV CUDA not available)")
            return GPUSparsePyrLK(
                win_size=win_size, max_level=max_level,
                max_iter=max_iter, eps=eps,
                min_eig_threshold=min_eig_threshold,
            )
    elif backend == "opencv_cuda":
        if not _OPENCV_CUDA_LK_AVAILABLE:
            raise RuntimeError(
                "OpenCV CUDA LK requested but cv2.cuda.SparsePyrLKOpticalFlow "
                "is not available. Build OpenCV with -DWITH_CUDA=ON."
            )
        return OpenCVCudaSparsePyrLK(
            win_size=win_size, max_level=max_level,
            max_iter=max_iter, eps=eps,
        )
    elif backend == "pytorch":
        return GPUSparsePyrLK(
            win_size=win_size, max_level=max_level,
            max_iter=max_iter, eps=eps,
            min_eig_threshold=min_eig_threshold,
        )
    else:
        raise ValueError(f"Unknown GPU LK backend: {backend!r}")


# ---------------------------------------------------------------------------
# Abstract base (optional typing hint)
# ---------------------------------------------------------------------------
class GPUSparsePyrLKBase:
    """Common interface for GPU LK backends."""

    def calc(self, prev_gray, next_gray, prev_pts):
        raise NotImplementedError

    def calc_with_fb_check(self, prev_gray, next_gray, prev_pts, fb_threshold=1.0):
        raise NotImplementedError

    def calc_numpy(self, prev_gray_np, next_gray_np, prev_pts_np, fb_threshold=None):
        raise NotImplementedError


# ===================================================================
# Backend 1: OpenCV CUDA  (cv2.cuda.SparsePyrLKOpticalFlow)
# ===================================================================
class OpenCVCudaSparsePyrLK(GPUSparsePyrLKBase):
    """Wrapper around cv2.cuda.SparsePyrLKOpticalFlow.

    Accepts the same tensor/numpy inputs as GPUSparsePyrLK and converts
    internally to GpuMat.  Points come back as torch tensors on CUDA."""

    def __init__(
        self,
        win_size: int = 21,
        max_level: int = 3,
        max_iter: int = 30,
        eps: float = 0.01,
    ):
        import cv2
        self.win_size = win_size
        self.max_level = max_level
        self.max_iter = max_iter
        self.eps = eps

        self._lk = cv2.cuda.SparsePyrLKOpticalFlow.create(
            winSize=(win_size, win_size),
            maxLevel=max_level,
            iters=max_iter,
        )
        self.device = "cuda"

    # ---------- helpers ----------

    @staticmethod
    def _to_gpumat_gray(img) -> "cv2.cuda.GpuMat":
        """Convert torch tensor or numpy array to a CV_8U GpuMat."""
        import cv2
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
        else:
            arr = img
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        gm = cv2.cuda.GpuMat()
        gm.upload(arr)
        return gm

    @staticmethod
    def _pts_to_gpumat(pts) -> "cv2.cuda.GpuMat":
        """Convert (N,2) points to a (1, N, 2) float32 GpuMat (CV_32FC2)."""
        import cv2
        if isinstance(pts, torch.Tensor):
            arr = pts.detach().cpu().numpy()
        else:
            arr = np.asarray(pts)
        arr = arr.reshape(1, -1, 2).astype(np.float32)
        gm = cv2.cuda.GpuMat()
        gm.upload(arr)
        return gm

    @staticmethod
    def _gpumat_pts_to_numpy(gm) -> np.ndarray:
        """Download a (1, N, 2) GpuMat to (N, 2) numpy."""
        arr = gm.download()            # (1, N, 2)
        return arr.reshape(-1, 2)

    # ---------- public API ----------

    @torch.no_grad()
    def calc(
        self,
        prev_gray: "torch.Tensor | np.ndarray",
        next_gray: "torch.Tensor | np.ndarray",
        prev_pts: "torch.Tensor | np.ndarray",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Track points forward.  Returns (new_pts, status) as CUDA tensors."""
        gm_prev = self._to_gpumat_gray(prev_gray)
        gm_next = self._to_gpumat_gray(next_gray)
        gm_pts = self._pts_to_gpumat(prev_pts)

        gm_new, gm_status, _gm_err = self._lk.calc(gm_prev, gm_next, gm_pts, None)

        new_np = self._gpumat_pts_to_numpy(gm_new)
        status_np = gm_status.download().flatten().astype(bool)

        # Bounds check
        if isinstance(prev_gray, torch.Tensor):
            H, W = prev_gray.shape[-2], prev_gray.shape[-1]
        else:
            H, W = prev_gray.shape[:2]
        in_bounds = (
            (new_np[:, 0] >= 0) & (new_np[:, 0] < W) &
            (new_np[:, 1] >= 0) & (new_np[:, 1] < H)
        )
        status_np = status_np & in_bounds

        new_t = torch.from_numpy(new_np).cuda()
        status_t = torch.from_numpy(status_np).cuda()
        return new_t, status_t

    @torch.no_grad()
    def calc_with_fb_check(
        self,
        prev_gray,
        next_gray,
        prev_pts,
        fb_threshold: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Track with forward-backward consistency check.

        Returns:
            new_pts: (N, 2) tracked positions
            status: (N,) bool mask — bidir convergence + in-bounds + fb < threshold
            fb_errors: (N,) forward-backward error per point (pixels).
                       Valid only where bidir convergence succeeded (fwd_ok & bwd_ok & in_bounds).
                       For failed points, fb_error may be meaningless."""
        gm_prev = self._to_gpumat_gray(prev_gray)
        gm_next = self._to_gpumat_gray(next_gray)
        gm_pts = self._pts_to_gpumat(prev_pts)

        # Forward
        gm_fwd, gm_fwd_status, _ = self._lk.calc(gm_prev, gm_next, gm_pts, None)

        # Backward (no initial flow — starts from tracked positions)
        gm_bwd, gm_bwd_status, _ = self._lk.calc(gm_next, gm_prev, gm_fwd, None)

        # Download
        new_np = self._gpumat_pts_to_numpy(gm_fwd)
        back_np = self._gpumat_pts_to_numpy(gm_bwd)
        fwd_ok = gm_fwd_status.download().flatten().astype(bool)
        bwd_ok = gm_bwd_status.download().flatten().astype(bool)

        # Original points
        if isinstance(prev_pts, torch.Tensor):
            orig_np = prev_pts.detach().cpu().numpy().reshape(-1, 2)
        else:
            orig_np = np.asarray(prev_pts).reshape(-1, 2)

        fb_err = np.linalg.norm(orig_np - back_np, axis=1)

        # Bounds
        if isinstance(prev_gray, torch.Tensor):
            H, W = prev_gray.shape[-2], prev_gray.shape[-1]
        else:
            H, W = prev_gray.shape[:2]
        in_bounds = (
            (new_np[:, 0] >= 0) & (new_np[:, 0] < W) &
            (new_np[:, 1] >= 0) & (new_np[:, 1] < H)
        )

        bidir_ok = fwd_ok & bwd_ok & in_bounds
        # Set fb_err to inf for points that failed bidir convergence
        # so callers can safely filter by fb_err < threshold
        fb_err[~bidir_ok] = float('inf')
        status_np = bidir_ok & (fb_err < fb_threshold)

        new_t = torch.from_numpy(new_np).cuda()
        status_t = torch.from_numpy(status_np).cuda()
        fb_err_t = torch.from_numpy(fb_err.astype(np.float32)).cuda()
        return new_t, status_t, fb_err_t

    def calc_numpy(self, prev_gray_np, next_gray_np, prev_pts_np,
                   fb_threshold=None):
        """NumPy convenience wrapper."""
        if fb_threshold is not None:
            new_t, status_t, fb_err_t = self.calc_with_fb_check(
                prev_gray_np, next_gray_np, prev_pts_np, fb_threshold,
            )
            return new_t.cpu().numpy(), status_t.cpu().numpy()
        else:
            new_t, status_t = self.calc(
                prev_gray_np, next_gray_np, prev_pts_np,
            )
            return new_t.cpu().numpy(), status_t.cpu().numpy()


# ===================================================================
# Backend 2: PyTorch grid_sample  (original implementation)
# ===================================================================


class GPUSparsePyrLK(GPUSparsePyrLKBase):
    """GPU-accelerated sparse pyramidal Lucas-Kanade optical flow.

    Tracks sparse 2D points from prev_gray to next_gray using a coarse-to-fine
    pyramid scheme.  Patch extraction is done via grid_sample with all N points
    folded into a single call, and the 2x2 structure-tensor solve is fully
    vectorized.

    Args:
        win_size: Side length of the square tracking window (must be odd).
        max_level: Number of pyramid levels (0 = single level).
        max_iter: Maximum LK iterations per pyramid level.
        eps: Convergence threshold (L2 norm of displacement update).
        min_eig_threshold: Minimum eigenvalue of the structure tensor;
            points below this are marked as lost.
        device: CUDA device string.  ``None`` = auto-detect."""

    def __init__(
        self,
        win_size: int = 21,
        max_level: int = 3,
        max_iter: int = 30,
        eps: float = 0.01,
        min_eig_threshold: float = 1e-4,
        device: Optional[str] = None,
    ):
        assert win_size % 2 == 1, "win_size must be odd"
        self.win_size = win_size
        self.half_win = win_size // 2
        self.max_level = max_level
        self.max_iter = max_iter
        self.eps = eps
        self.min_eig_threshold = min_eig_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Pre-compute the local sampling grid offsets for one patch.
        # Shape: (win_size*win_size, 2) with values in [-half_win, +half_win]
        hw = self.half_win
        gy, gx = torch.meshgrid(
            torch.arange(-hw, hw + 1, dtype=torch.float32),
            torch.arange(-hw, hw + 1, dtype=torch.float32),
            indexing="ij",
        )
        # (P, 2) where P = win_size^2  — flattened (dx, dy) offsets
        self._patch_offsets_flat = torch.stack(
            [gx.reshape(-1), gy.reshape(-1)], dim=-1
        ).to(self.device)  # (P, 2)
        self._P = self.win_size * self.win_size

        # Scharr kernels (pre-allocated)
        self._scharr_x = (
            torch.tensor(
                [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
                dtype=torch.float32, device=self.device,
            )
            .unsqueeze(0).unsqueeze(0) / 32.0
        )
        self._scharr_y = self._scharr_x.transpose(2, 3).contiguous()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def calc(
        self,
        prev_gray: torch.Tensor,
        next_gray: torch.Tensor,
        prev_pts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Track points from *prev_gray* to *next_gray*.

        Args:
            prev_gray: (H, W) float32 tensor, values in [0, 255] or [0, 1].
            next_gray: same shape / range as *prev_gray*.
            prev_pts:  (N, 2) float32 tensor of (x, y) coordinates in pixel space.

        Returns:
            new_pts: (N, 2) float32 tensor of tracked positions.
            status:  (N,) bool tensor -- True if the point was tracked successfully."""
        if prev_pts.numel() == 0:
            return prev_pts.clone(), torch.zeros(0, dtype=torch.bool, device=self.device)

        prev_gray = self._to_gpu(prev_gray)
        next_gray = self._to_gpu(next_gray)
        prev_pts = self._to_gpu(prev_pts).float()

        # Build Gaussian pyramids
        prev_pyr = self._build_pyramid(prev_gray)
        next_pyr = self._build_pyramid(next_gray)

        N = prev_pts.shape[0]
        P = self._P
        flow = torch.zeros(N, 2, device=self.device, dtype=torch.float32)

        for level in range(self.max_level, -1, -1):
            scale = 1.0 / (2 ** level)
            prev_img = prev_pyr[level]  # (H_l, W_l)
            next_img = next_pyr[level]
            H, W = prev_img.shape

            # Image in (1, 1, H, W) for grid_sample — allocated once per level
            prev_4d = prev_img.unsqueeze(0).unsqueeze(0)
            next_4d = next_img.unsqueeze(0).unsqueeze(0)

            # Scale points to this pyramid level
            pts_level = prev_pts * scale  # (N, 2)

            # Compute spatial gradients of prev_img
            Ix, Iy = self._compute_gradients(prev_4d)
            Ix_4d = Ix.unsqueeze(0).unsqueeze(0)
            Iy_4d = Iy.unsqueeze(0).unsqueeze(0)

            # Compute sampling grid for prev patches (fixed at this level)
            # grid_prev: (1, N*P, 1, 2) normalized to [-1, 1]
            grid_prev = self._make_grid(pts_level, H, W)

            # Extract gradient patches and prev patches — single grid_sample each
            patch_Ix = F.grid_sample(Ix_4d, grid_prev, mode='bilinear',
                                     padding_mode='zeros', align_corners=True)
            patch_Ix = patch_Ix.reshape(N, P)  # (N, P)

            patch_Iy = F.grid_sample(Iy_4d, grid_prev, mode='bilinear',
                                     padding_mode='zeros', align_corners=True)
            patch_Iy = patch_Iy.reshape(N, P)

            patch_prev = F.grid_sample(prev_4d, grid_prev, mode='bilinear',
                                       padding_mode='zeros', align_corners=True)
            patch_prev = patch_prev.reshape(N, P)

            # Structure tensor components (summed over patch)  — (N,)
            Ixx = (patch_Ix * patch_Ix).sum(dim=1)
            Ixy = (patch_Ix * patch_Iy).sum(dim=1)
            Iyy = (patch_Iy * patch_Iy).sum(dim=1)

            # Determinant and inverse for 2x2 solve  — (N,)
            det = Ixx * Iyy - Ixy * Ixy
            valid = det.abs() > 1e-7
            inv_det = torch.where(valid, 1.0 / det.clamp(min=1e-12),
                                  torch.zeros_like(det))

            # Iterative refinement at this level
            for _ in range(self.max_iter):
                # Current destination points in this level's coordinate system
                dst_pts = pts_level + flow * scale  # (N, 2)

                # Sampling grid for next_img patches
                grid_next = self._make_grid(dst_pts, H, W)

                patch_next = F.grid_sample(next_4d, grid_next, mode='bilinear',
                                           padding_mode='zeros', align_corners=True)
                patch_next = patch_next.reshape(N, P)

                # Temporal gradient
                It = patch_next - patch_prev  # (N, P)

                # Right-hand side
                bx = -(patch_Ix * It).sum(dim=1)  # (N,)
                by = -(patch_Iy * It).sum(dim=1)

                # Solve => displacement update at this level
                du = inv_det * (Iyy * bx - Ixy * by)
                dv = inv_det * (-Ixy * bx + Ixx * by)

                # Convert delta back to original-scale flow
                delta = torch.stack([du, dv], dim=-1) / scale
                flow = flow + delta

                # Early convergence check
                if (delta.norm(dim=-1).max() < self.eps):
                    break

        new_pts = prev_pts + flow  # (N, 2)

        # Determine status: minimum eigenvalue + bounds check
        H0, W0 = prev_pyr[0].shape
        trace = Ixx + Iyy
        disc = torch.sqrt(((Ixx - Iyy) ** 2 + 4 * Ixy ** 2).clamp(min=0))
        min_eig = 0.5 * (trace - disc)
        status = (min_eig / P > self.min_eig_threshold)
        status = status & (new_pts[:, 0] >= 0) & (new_pts[:, 0] < W0)
        status = status & (new_pts[:, 1] >= 0) & (new_pts[:, 1] < H0)

        return new_pts, status

    @torch.no_grad()
    def calc_with_fb_check(
        self,
        prev_gray: torch.Tensor,
        next_gray: torch.Tensor,
        prev_pts: torch.Tensor,
        fb_threshold: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Track with forward-backward consistency check.

        Tracks prev->next, then next->prev, and keeps only points where the
        round-trip error is below *fb_threshold* pixels.

        Returns:
            new_pts: (N, 2) tracked positions.
            status:  (N,) bool tensor.
            fb_error: (N,) float tensor of forward-backward errors (pixels)."""
        prev_gray = self._to_gpu(prev_gray)
        next_gray = self._to_gpu(next_gray)
        prev_pts = self._to_gpu(prev_pts).float()

        # Build pyramids once, share across forward / backward
        prev_pyr = self._build_pyramid(prev_gray)
        next_pyr = self._build_pyramid(next_gray)

        # Forward pass
        new_pts, fwd_status = self._calc_with_pyramids(
            prev_pyr, next_pyr, prev_pts)

        # Backward pass (no initial flow -- starts from tracked positions)
        back_pts, bwd_status = self._calc_with_pyramids(next_pyr, prev_pyr, new_pts)

        # Forward-backward consistency
        fb_error = (prev_pts - back_pts).norm(dim=-1)
        bidir_ok = fwd_status & bwd_status
        # Set fb_error to inf for points that failed bidir convergence
        fb_error = torch.where(bidir_ok, fb_error, torch.tensor(float('inf'), device=fb_error.device))
        status = bidir_ok & (fb_error < fb_threshold)

        return new_pts, status, fb_error

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calc_with_pyramids(
        self,
        prev_pyr: list,
        next_pyr: list,
        prev_pts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core LK tracker operating on pre-built pyramids."""
        N = prev_pts.shape[0]
        P = self._P

        if N == 0:
            return prev_pts.clone(), torch.zeros(0, dtype=torch.bool, device=self.device)

        flow = torch.zeros(N, 2, device=self.device, dtype=torch.float32)

        for level in range(min(self.max_level, len(prev_pyr) - 1), -1, -1):
            scale = 1.0 / (2 ** level)
            prev_img = prev_pyr[level]
            next_img = next_pyr[level]
            H, W = prev_img.shape

            prev_4d = prev_img.unsqueeze(0).unsqueeze(0)
            next_4d = next_img.unsqueeze(0).unsqueeze(0)
            pts_level = prev_pts * scale

            Ix, Iy = self._compute_gradients(prev_4d)
            Ix_4d = Ix.unsqueeze(0).unsqueeze(0)
            Iy_4d = Iy.unsqueeze(0).unsqueeze(0)

            grid_prev = self._make_grid(pts_level, H, W)

            patch_Ix = F.grid_sample(Ix_4d, grid_prev, mode='bilinear',
                                     padding_mode='zeros', align_corners=True).reshape(N, P)
            patch_Iy = F.grid_sample(Iy_4d, grid_prev, mode='bilinear',
                                     padding_mode='zeros', align_corners=True).reshape(N, P)
            patch_prev = F.grid_sample(prev_4d, grid_prev, mode='bilinear',
                                       padding_mode='zeros', align_corners=True).reshape(N, P)

            Ixx = (patch_Ix * patch_Ix).sum(dim=1)
            Ixy = (patch_Ix * patch_Iy).sum(dim=1)
            Iyy = (patch_Iy * patch_Iy).sum(dim=1)

            det = Ixx * Iyy - Ixy * Ixy
            valid = det.abs() > 1e-7
            inv_det = torch.where(valid, 1.0 / det.clamp(min=1e-12),
                                  torch.zeros_like(det))

            for _ in range(self.max_iter):
                dst_pts = pts_level + flow * scale
                grid_next = self._make_grid(dst_pts, H, W)
                patch_next = F.grid_sample(next_4d, grid_next, mode='bilinear',
                                           padding_mode='zeros',
                                           align_corners=True).reshape(N, P)

                It = patch_next - patch_prev
                bx = -(patch_Ix * It).sum(dim=1)
                by = -(patch_Iy * It).sum(dim=1)

                du = inv_det * (Iyy * bx - Ixy * by)
                dv = inv_det * (-Ixy * bx + Ixx * by)
                delta = torch.stack([du, dv], dim=-1) / scale
                flow = flow + delta

                if delta.norm(dim=-1).max() < self.eps:
                    break

        new_pts = prev_pts + flow
        H0, W0 = prev_pyr[0].shape

        trace = Ixx + Iyy
        disc = torch.sqrt(((Ixx - Iyy) ** 2 + 4 * Ixy ** 2).clamp(min=0))
        min_eig = 0.5 * (trace - disc)
        status = (min_eig / P > self.min_eig_threshold)
        status = status & (new_pts[:, 0] >= 0) & (new_pts[:, 0] < W0)
        status = status & (new_pts[:, 1] >= 0) & (new_pts[:, 1] < H0)

        return new_pts, status

    def _to_gpu(self, t: torch.Tensor) -> torch.Tensor:
        if t.device.type != self.device.split(":")[0]:
            return t.to(self.device)
        return t

    def _build_pyramid(self, img: torch.Tensor) -> list:
        """Build a Gaussian image pyramid.

        Returns list of (H_l, W_l) tensors from finest (level 0) to coarsest."""
        pyr = [img]
        current = img
        for _ in range(self.max_level):
            h, w = current.shape
            if h < 4 or w < 4:
                break
            current_4d = current.unsqueeze(0).unsqueeze(0)
            # Gaussian-like smoothing + 2x downsample
            smoothed = F.avg_pool2d(current_4d, kernel_size=3, stride=1, padding=1)
            downsampled = F.avg_pool2d(smoothed, kernel_size=2, stride=2)
            current = downsampled.squeeze(0).squeeze(0)
            pyr.append(current)
        return pyr

    def _compute_gradients(
        self, img_4d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial gradients using Scharr operator.

        Args:
            img_4d: (1, 1, H, W) float32 tensor.
        Returns:
            Ix, Iy: (H, W) gradient tensors."""
        Ix = F.conv2d(img_4d, self._scharr_x, padding=1).squeeze(0).squeeze(0)
        Iy = F.conv2d(img_4d, self._scharr_y, padding=1).squeeze(0).squeeze(0)
        return Ix, Iy

    def _make_grid(self, pts: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Build a normalized sampling grid for all N points at once.

        Returns (1, N*P, 1, 2) grid suitable for grid_sample on a (1,1,H,W) image.
        This avoids the N-fold batch expansion entirely."""
        N = pts.shape[0]
        P = self._P
        # pts: (N, 2),  offsets: (P, 2)
        # grid_px: (N, P, 2) = pts[:, None, :] + offsets[None, :, :]
        grid_px = pts.unsqueeze(1) + self._patch_offsets_flat.unsqueeze(0)  # (N, P, 2)

        # Normalize to [-1, 1]
        grid_norm = torch.empty_like(grid_px)
        grid_norm[..., 0] = 2.0 * grid_px[..., 0] / max(W - 1, 1) - 1.0
        grid_norm[..., 1] = 2.0 * grid_px[..., 1] / max(H - 1, 1) - 1.0

        # Reshape to (1, N*P, 1, 2) for a single grid_sample call
        return grid_norm.reshape(1, N * P, 1, 2)

    # ------------------------------------------------------------------
    # Convenience: np <-> torch conversion wrappers
    # ------------------------------------------------------------------

    def calc_numpy(
        self,
        prev_gray_np,
        next_gray_np,
        prev_pts_np,
        fb_threshold: Optional[float] = None,
    ):
        """NumPy wrapper matching the interface used by FeatureTracker._track_lk.

        Args:
            prev_gray_np: (H, W) uint8 or float32 numpy array (grayscale).
            next_gray_np: same.
            prev_pts_np: (N, 1, 2) or (N, 2) float32 numpy array of (x, y).
            fb_threshold: If not None, run forward-backward check.

        Returns:
            new_pts_np: (N, 2) float32 numpy array.
            status_np:  (N,) bool numpy array."""
        import numpy as np

        prev_t = torch.from_numpy(prev_gray_np.astype(np.float32)).to(self.device)
        next_t = torch.from_numpy(next_gray_np.astype(np.float32)).to(self.device)

        pts = prev_pts_np.reshape(-1, 2)
        pts_t = torch.from_numpy(pts.astype(np.float32)).to(self.device)

        if fb_threshold is not None:
            new_pts_t, status_t, fb_err_t = self.calc_with_fb_check(
                prev_t, next_t, pts_t, fb_threshold
            )
        else:
            new_pts_t, status_t = self.calc(prev_t, next_t, pts_t)

        return new_pts_t.cpu().numpy(), status_t.cpu().numpy()
