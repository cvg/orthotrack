"""
UFM (Unified Flow Matching) wrapper for OrthoTrack.

Paper: "UniFlowMatch: Unified Dense Correspondence Matching"
Repo: https://github.com/UniFlowMatch/UFM

UFM is a dense flow matcher (0.3-0.4B params) that outputs pixel-level
optical flow and covisibility confidence.  Weights auto-download from
HuggingFace Hub."""

import sys
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from orthotrack.matchers.base_matcher import BaseMatcher
from utils.matching import MatchResult

_UFM_ROOT = str(Path(__file__).resolve().parents[2] / "thirdparty" / "UFM")


def _load_ufm_model(
    device: str = "cuda",
    variant: str = "base",
):
    """Build and load a UFM model from HuggingFace."""
    if _UFM_ROOT not in sys.path:
        sys.path.insert(0, _UFM_ROOT)

    # Also need UniCeption which is bundled as a subfolder
    uniception_path = os.path.join(_UFM_ROOT, "UniCeption")
    if os.path.isdir(uniception_path) and uniception_path not in sys.path:
        sys.path.insert(0, uniception_path)

    from uniflowmatch.models.ufm import (  # noqa: E402
        UniFlowMatchConfidence,
        UniFlowMatchClassificationRefinement,
    )

    _MODEL_IDS = {
        "base": "infinity1096/UFM-Base",
        "refine": "infinity1096/UFM-Refine",
        "base_980": "infinity1096/UFM-Base-980",
        "refine_980": "infinity1096/UFM-Refine-980",
    }
    model_id = _MODEL_IDS.get(variant, variant)

    if "refine" in variant.lower():
        model = UniFlowMatchClassificationRefinement.from_pretrained(model_id)
    else:
        model = UniFlowMatchConfidence.from_pretrained(model_id)

    # Ensure float32 — HuggingFace weights may be stored in bfloat16,
    # which causes dtype mismatches with float32 DINOv2 backbone inputs.
    model = model.float().to(device).eval()
    return model


class UFMMatcher(BaseMatcher):
    """UFM dense flow matcher wrapper."""

    name = "ufm"

    def __init__(self, device: str = "cuda", **kwargs):
        self.device = device
        self.variant = kwargs.pop("variant", "base")
        self._model = _load_ufm_model(device=device, variant=self.variant)
        print(f"[UFMMatcher] loaded variant={self.variant} on {device}")

    @torch.inference_mode()
    def match(
        self,
        query_image: np.ndarray,
        dop_image: np.ndarray,
        num_matches: int = 5000,
    ) -> MatchResult:
        query_h, query_w = query_image.shape[:2]
        dop_h, dop_w = dop_image.shape[:2]

        # UFM expects RGB uint8 HWC tensors
        src_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
        tgt_rgb = cv2.cvtColor(dop_image, cv2.COLOR_BGR2RGB)

        result = self._model.predict_correspondences_batched(
            source_image=torch.from_numpy(src_rgb).to(self.device),
            target_image=torch.from_numpy(tgt_rgb).to(self.device),
        )

        flow = result.flow.flow_output[0].cpu().numpy()       # (2, H_out, W_out)
        covis = result.covisibility.mask[0].cpu().numpy()      # (H_out, W_out)

        flow_h, flow_w = flow.shape[1], flow.shape[2]

        # Build source pixel grid
        ys, xs = np.mgrid[:flow_h, :flow_w]

        # Scale source grid to original query coords
        scale_x_src = query_w / flow_w
        scale_y_src = query_h / flow_h

        src_x = xs.astype(np.float32) * scale_x_src
        src_y = ys.astype(np.float32) * scale_y_src

        # Target coords = source + flow, then scale
        # The flow is in the flow-resolution pixel space;
        # scale to original DOP coords
        scale_x_tgt = dop_w / flow_w
        scale_y_tgt = dop_h / flow_h

        tgt_x = (xs.astype(np.float32) + flow[0]) * scale_x_tgt
        tgt_y = (ys.astype(np.float32) + flow[1]) * scale_y_tgt

        # Flatten and filter by covisibility
        covis_flat = covis.ravel()
        src_x_flat = src_x.ravel()
        src_y_flat = src_y.ravel()
        tgt_x_flat = tgt_x.ravel()
        tgt_y_flat = tgt_y.ravel()

        # Sort by confidence, take top num_matches
        order = np.argsort(-covis_flat)[:num_matches]
        # Also threshold: keep only reasonable confidence
        mask = covis_flat[order] > 0.1
        order = order[mask]

        if len(order) == 0:
            return MatchResult(
                kpts_query=np.zeros((0, 2)),
                kpts_dop=np.zeros((0, 2)),
                confidences=np.zeros(0),
                query_size=(query_h, query_w),
                dop_size=(dop_h, dop_w),
            )

        kpts_query = np.stack([src_x_flat[order], src_y_flat[order]], axis=1)
        kpts_dop = np.stack([tgt_x_flat[order], tgt_y_flat[order]], axis=1)
        confidences = covis_flat[order]

        return MatchResult(
            kpts_query=kpts_query,
            kpts_dop=kpts_dop,
            confidences=confidences,
            query_size=(query_h, query_w),
            dop_size=(dop_h, dop_w),
        )

    @torch.inference_mode()
    def match_batch(
        self,
        query_image: np.ndarray,
        dop_images: List[np.ndarray],
        num_matches_per_crop: int = 3000,
    ) -> List[MatchResult]:
        return [self.match(query_image, dop, num_matches_per_crop)
                for dop in dop_images]
