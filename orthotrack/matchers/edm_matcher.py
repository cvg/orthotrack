"""
EDM (Efficient Dense Matching) wrapper for OrthoTrack.

Paper: "EDM: Efficient Dense Matching"
Repo: https://github.com/chicleee/EDM

EDM is a sparse matcher based on LoFTR-style architecture with ResNet18
backbone.  It outputs mutual keypoints and confidence scores.

Weights must be downloaded from the EDM Google Drive and placed in
``thirdparty/EDM/weights/edm_outdoor.ckpt``."""

import sys
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from orthotrack.matchers.base_matcher import BaseMatcher
from utils.matching import MatchResult

_EDM_ROOT = str(Path(__file__).resolve().parents[2] / "thirdparty" / "EDM")


def _load_edm_model(
    config_path: str = "configs/edm/outdoor/edm_base.py",
    weight_path: str = "weights/edm_outdoor.ckpt",
    device: str = "cuda",
):
    """Build and load an EDM model."""
    # Temporarily add EDM root to path (imports use src.* relative packages)
    if _EDM_ROOT not in sys.path:
        sys.path.insert(0, _EDM_ROOT)

    from src.config.default import get_cfg_defaults   # noqa: E402
    from src.utils.misc import lower_config           # noqa: E402
    from src.edm.edm import EDM as EDMModel            # noqa: E402

    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(_EDM_ROOT, config_path))

    # NPE must be set for RoPE positional encoding (not in edm_base.py,
    # only in data configs).  Values: [train_H, train_W, test_H, test_W].
    if cfg.EDM.NECK.NPE is None:
        cfg.EDM.NECK.NPE = [
            cfg.EDM.TRAIN_RES_H, cfg.EDM.TRAIN_RES_W,
            cfg.EDM.TEST_RES_H,  cfg.EDM.TEST_RES_W,
        ]

    _config = lower_config(cfg)

    model = EDMModel(config=_config["edm"])
    model = model.to(device).eval()

    weight_file = os.path.join(_EDM_ROOT, weight_path)
    if not os.path.isfile(weight_file):
        raise FileNotFoundError(
            f"EDM weights not found at {weight_file}. "
            "Download from https://drive.google.com/drive/folders/"
            "1PkYNihwgnNwqQeeewBz4OUDrvY8xFdH0 and place in "
            f"{os.path.join(_EDM_ROOT, 'weights')}/"
        )
    ckpt = torch.load(weight_file, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    # Strip 'matcher.' prefix if present
    state = {k.replace("matcher.", "", 1) if k.startswith("matcher.") else k: v
             for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model


class EDMMatcher(BaseMatcher):
    """EDM sparse matcher wrapper."""

    name = "edm"

    def __init__(self, device: str = "cuda", **kwargs):
        self.device = device
        # Resolution must be divisible by 32 (outdoor default: 1152x1152)
        self.resize = kwargs.pop("resize", 832)
        self._model = _load_edm_model(device=device)
        print(f"[EDMMatcher] loaded, resize={self.resize}")

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """Convert uint8 BGR/RGB (H,W,3) -> grayscale float (1,1,H',W')."""
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        # Resize to self.resize x self.resize (must be divisible by 32)
        gray = cv2.resize(gray, (self.resize, self.resize))
        t = torch.from_numpy(gray).float()[None, None] / 255.0
        return t.to(self.device)

    @torch.inference_mode()
    def match(
        self,
        query_image: np.ndarray,
        dop_image: np.ndarray,
        num_matches: int = 5000,
    ) -> MatchResult:
        query_h, query_w = query_image.shape[:2]
        dop_h, dop_w = dop_image.shape[:2]

        img0 = self._preprocess(query_image)
        img1 = self._preprocess(dop_image)

        batch = {"image0": img0, "image1": img1}
        self._model(batch)

        mkpts0 = batch["mkpts0_f"].cpu().numpy()  # (N, 2) in resize coords
        mkpts1 = batch["mkpts1_f"].cpu().numpy()
        mconf = batch["mconf"].cpu().numpy()

        if len(mkpts0) == 0:
            return MatchResult(
                kpts_query=np.zeros((0, 2)),
                kpts_dop=np.zeros((0, 2)),
                confidences=np.zeros(0),
                query_size=(query_h, query_w),
                dop_size=(dop_h, dop_w),
            )

        # Scale from resize coords to original pixel coords
        mkpts0[:, 0] *= query_w / self.resize
        mkpts0[:, 1] *= query_h / self.resize
        mkpts1[:, 0] *= dop_w / self.resize
        mkpts1[:, 1] *= dop_h / self.resize

        # Keep top num_matches by confidence
        if len(mkpts0) > num_matches:
            idx = np.argsort(-mconf)[:num_matches]
            mkpts0 = mkpts0[idx]
            mkpts1 = mkpts1[idx]
            mconf = mconf[idx]

        return MatchResult(
            kpts_query=mkpts0,
            kpts_dop=mkpts1,
            confidences=mconf,
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
