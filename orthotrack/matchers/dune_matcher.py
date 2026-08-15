"""
DUNE + MASt3R matcher for OrthoTrack.

Uses the DUNE universal encoder (CVPR 2025, Naver Labs) pre-trained with
DINOv2 + MASt3R + Multi-HMR distillation, combined with the MASt3R decoder
for 3D-aware dense matching.

The checkpoint ``DUNEMASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth``
is downloaded from Hugging Face on first use."""

import sys
import os
import time
import warnings
import numpy as np
from typing import List, Optional, Tuple

from orthotrack.matchers.base_matcher import BaseMatcher
from utils.matching import MatchResult


class DuneMast3rMatcher(BaseMatcher):
    """Dense 3D-aware matcher using DUNE encoder + MASt3R decoder.

    This matcher produces dense 2D correspondences between a query and DOP
    image.  The DUNE encoder provides cross-domain robustness (important for
    the aerial-to-ortho domain gap), while MASt3R's decoder provides
    3D-aware matching that handles scale changes.

    Parameters
    ----------
    device : str
        ``'cuda'`` or ``'cpu'``.
    model_name : str
        Hugging Face model name for the DUNE+MASt3R checkpoint.
    image_size : int
        Input resolution (images are resized to ``image_size × image_size``)."""

    name = "dune_mast3r"

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "naver/DUNEMASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        image_size: int = 512,
    ):
        self._device = device
        self._image_size = image_size
        self._model = None
        self._model_name = model_name

        print(f"Initializing DuneMast3rMatcher (image_size={image_size}) ...")
        t0 = time.time()
        self._load_model()
        print(f"  DuneMast3rMatcher loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    #  Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the DUNE+MASt3R model from Hugging Face."""
        try:
            # mast3r is typically installed from naver/mast3r
            from mast3r.model import AsymmetricMASt3R
            self._model = AsymmetricMASt3R.from_pretrained(self._model_name)
            self._model = self._model.to(self._device).eval()
        except ImportError as exc:
            raise ImportError(
                "mast3r package not found. Install it via:\n"
                "  pip install git+https://github.com/naver/mast3r.git\n"
                "or clone and install:\n"
                "  git clone --recursive https://github.com/naver/mast3r.git thirdparty/mast3r\n"
                "  pip install -e thirdparty/mast3r"
            ) from exc

    def _prepare_image(self, image: np.ndarray):
        """Convert numpy HWC uint8 image to mast3r's expected input format."""
        import torch


        # mast3r expects dict with 'img' tensor (1, 3, H, W) in [0, 1]
        # and 'true_shape' (1, 2)
        h, w = image.shape[:2]
        img = image.astype(np.float32) / 255.0
        # Resize to target resolution
        if h != self._image_size or w != self._image_size:
            import cv2
            img = cv2.resize(img, (self._image_size, self._image_size),
                             interpolation=cv2.INTER_LANCZOS4)

        # HWC -> CHW, add batch dim
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        # Normalize to [-1, 1] as expected by MASt3R
        img_tensor = img_tensor * 2 - 1

        return {
            "img": img_tensor.to(self._device),
            "true_shape": torch.tensor([[h, w]], device=self._device),
        }

    # ------------------------------------------------------------------
    #  BaseMatcher interface
    # ------------------------------------------------------------------

    def match(
        self,
        query_image: np.ndarray,
        dop_image: np.ndarray,
        num_matches: int = 5000,
    ) -> MatchResult:
        import torch

        query_h, query_w = query_image.shape[:2]
        dop_h, dop_w = dop_image.shape[:2]

        view1 = self._prepare_image(query_image)
        view2 = self._prepare_image(dop_image)

        with torch.inference_mode():
            pred1, pred2 = self._model(view1, view2)

        # pred1/pred2 each contain 'pts3d', 'conf', 'desc', 'desc_conf'
        # 'pts3d': (1, H, W, 3) — 3D point maps
        # 'conf':  (1, H, W)    — feature confidence

        # Use reciprocal nearest-neighbour matching on descriptors
        pts0, pts1, confidences = self._extract_correspondences(
            pred1, pred2, query_h, query_w, dop_h, dop_w, num_matches,
        )

        return MatchResult(
            kpts_query=pts0,
            kpts_dop=pts1,
            confidences=confidences,
            query_size=(query_h, query_w),
            dop_size=(dop_h, dop_w),
        )

    def match_batch(
        self,
        query_image: np.ndarray,
        dop_images: List[np.ndarray],
        num_matches_per_crop: int = 3000,
    ) -> List[MatchResult]:
        """Sequential matching — MASt3R runs pairwise."""
        return [
            self.match(query_image, dop, num_matches=num_matches_per_crop)
            for dop in dop_images
        ]

    # ------------------------------------------------------------------
    #  Correspondence extraction
    # ------------------------------------------------------------------

    def _extract_correspondences(
        self,
        pred1: dict,
        pred2: dict,
        query_h: int,
        query_w: int,
        dop_h: int,
        dop_w: int,
        num_matches: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract 2D correspondences via reciprocal nearest-neighbour on MASt3R descriptors."""
        import torch
        import torch.nn.functional as F

        # Descriptors: (1, H, W, D)
        desc1 = pred1["desc"][0]   # (H_feat, W_feat, D)
        desc2 = pred2["desc"][0]

        # Confidence
        conf1 = pred1["desc_conf"][0]  # (H_feat, W_feat)
        conf2 = pred2["desc_conf"][0]

        H_f, W_f = desc1.shape[:2]

        # Flatten to (N, D)
        d1 = desc1.reshape(-1, desc1.shape[-1])  # (H*W, D)
        d2 = desc2.reshape(-1, desc2.shape[-1])

        # Normalise
        d1 = F.normalize(d1, dim=-1)
        d2 = F.normalize(d2, dim=-1)

        # Cosine similarity → NN in both directions
        sim = d1 @ d2.T  # (N1, N2)

        nn12 = sim.argmax(dim=1)      # (N1,)
        nn21 = sim.argmax(dim=0)      # (N2,)

        # Reciprocal check
        idx1 = torch.arange(len(d1), device=d1.device)
        mutual = nn21[nn12] == idx1

        idx_query = idx1[mutual].cpu().numpy()
        idx_dop = nn12[mutual].cpu().numpy()

        # Feature-grid coordinates → pixel coordinates
        gy1, gx1 = np.divmod(idx_query, W_f)
        gy2, gx2 = np.divmod(idx_dop, W_f)

        pts_query = np.stack([
            (gx1 + 0.5) / W_f * query_w,
            (gy1 + 0.5) / H_f * query_h,
        ], axis=1)

        pts_dop = np.stack([
            (gx2 + 0.5) / W_f * dop_w,
            (gy2 + 0.5) / H_f * dop_h,
        ], axis=1)

        # Confidence = geometric mean of both descriptor confidences
        c1 = conf1.reshape(-1).cpu().numpy()[idx_query]
        c2 = conf2.reshape(-1).cpu().numpy()[idx_dop]
        confidences = np.sqrt(np.maximum(c1, 0) * np.maximum(c2, 0))

        # Keep top-k by confidence
        if len(pts_query) > num_matches:
            top_idx = np.argsort(-confidences)[:num_matches]
            pts_query = pts_query[top_idx]
            pts_dop = pts_dop[top_idx]
            confidences = confidences[top_idx]

        return pts_query, pts_dop, confidences
