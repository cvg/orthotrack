"""
L2M++ (Learn to Match++) wrapper for OrthoTrack.

Paper: "L2M: Learn to Match"
Repo: https://github.com/Sharpiless/L2M

L2M++ is a dense matcher built on top of RoMa with a DINOv2 encoder.
Outputs dense warp and certainty, from which keypoints are sampled.
Weights auto-download from HuggingFace."""

import sys
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from orthotrack.matchers.base_matcher import BaseMatcher
from utils.matching import MatchResult

_L2M_ROOT = str(Path(__file__).resolve().parents[2] / "thirdparty" / "L2M")


def _load_l2m_model(device: str = "cuda", version: str = "v2"):
    """Build and load an L2M++ model."""
    if _L2M_ROOT not in sys.path:
        sys.path.insert(0, _L2M_ROOT)

    from romatch.models.model_zoo import l2mpp_model  # noqa: E402

    model = l2mpp_model(
        device=torch.device(device),
        version=version,
        amp_dtype=torch.float16,
    )
    model.eval()
    return model


class L2MMatcher(BaseMatcher):
    """L2M++ dense matcher wrapper."""

    name = "l2m"

    def __init__(self, device: str = "cuda", **kwargs):
        self.device = device
        self._model = _load_l2m_model(device=device,
                                       version=kwargs.pop("version", "v2"))
        print(f"[L2MMatcher] loaded on {device}")

    @torch.inference_mode()
    def match(
        self,
        query_image: np.ndarray,
        dop_image: np.ndarray,
        num_matches: int = 5000,
    ) -> MatchResult:
        query_h, query_w = query_image.shape[:2]
        dop_h, dop_w = dop_image.shape[:2]

        # L2M expects PIL RGB images
        im_a = Image.fromarray(query_image[..., ::-1]) if query_image.shape[-1] == 3 else Image.fromarray(query_image)
        im_b = Image.fromarray(dop_image[..., ::-1]) if dop_image.shape[-1] == 3 else Image.fromarray(dop_image)
        im_a = im_a.convert("RGB")
        im_b = im_b.convert("RGB")

        warp, certainty = self._model.match(im_a, im_b, device=self.device)
        matches, cert = self._model.sample(warp, certainty, num=num_matches)

        if matches is None or len(matches) == 0:
            return MatchResult(
                kpts_query=np.zeros((0, 2)),
                kpts_dop=np.zeros((0, 2)),
                confidences=np.zeros(0),
                query_size=(query_h, query_w),
                dop_size=(dop_h, dop_w),
            )

        # Convert normalized [-1, 1] to pixel coords
        kpts_a, kpts_b = self._model.to_pixel_coordinates(
            matches, query_h, query_w, dop_h, dop_w
        )
        kpts_a = kpts_a.cpu().numpy()
        kpts_b = kpts_b.cpu().numpy()
        cert = cert.cpu().numpy()

        return MatchResult(
            kpts_query=kpts_a,
            kpts_dop=kpts_b,
            confidences=cert,
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
