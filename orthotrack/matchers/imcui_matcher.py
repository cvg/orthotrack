"""
IMCUI-based matcher adapter for OrthoTrack.

Wraps matchers from the Image Matching WebUI (imcui) library — accessed via
OrthoLoC's ``MatcherIMCUI`` — to conform to OrthoTrack's ``BaseMatcher``
interface.  This enables plug-and-play evaluation of 50+ feature matchers
(SIFT+LightGlue, SuperPoint+LightGlue, LoFTR, MASt3R, RDD, RIPE, LiftFeat,
DaD, XFeat, …)."""

import sys
import os
import time
import numpy as np
from typing import List, Optional

from orthotrack.matchers.base_matcher import BaseMatcher
from utils.matching import MatchResult

# Add OrthoLoC to path so we can import its MatcherIMCUI
_ORTHOLOC_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "thirdparty", "OrthoLoC")
if _ORTHOLOC_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_ORTHOLOC_ROOT))


def _patch_datasets_import():
    """Temporarily fix the ``datasets`` import clash.

    ``imcui.ui.utils`` does ``from datasets import load_dataset`` at the
    module level, referring to the HuggingFace ``datasets`` package.  When
    ``PYTHONPATH=.`` is set, our local ``datasets/`` package shadows it.

    This helper:
      1. Removes the project root (``'.'``) from ``sys.path``.
      2. Evicts any cached ``datasets`` and ``datasets.*`` modules.
      3. Returns a context-manager that restores everything on exit."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        _hide = [p for p in sys.path if os.path.abspath(p) == _project_root or p == "."]

        # --- save & remove ---
        for p in _hide:
            while p in sys.path:
                sys.path.remove(p)

        # Evict our local ``datasets`` and ``utils`` modules so that
        # HuggingFace's datasets and third-party utils (e.g. LiftFeat's
        # utils.featurebooster) load correctly instead of our project's.
        _stashed_modules = {}
        _conflict_prefixes = ("datasets", "utils")
        for key in list(sys.modules):
            if any(key == p or key.startswith(p + ".") for p in _conflict_prefixes):
                _stashed_modules[key] = sys.modules.pop(key)

        try:
            yield
        finally:
            # --- restore ---
            for p in reversed(_hide):
                if p not in sys.path:
                    sys.path.insert(0, p)
            # Restore the local datasets module references
            for key, mod in _stashed_modules.items():
                if key not in sys.modules:
                    sys.modules[key] = mod

    return _ctx()


def _ensure_matcher_in_zoo(name: str) -> None:
    """Patch OrthoLoC's MATCHER_ZOO with entries from imcui's app.yaml
    for matchers that imcui 0.0.4+ supports but that are missing from
    OrthoLoC's matchers_imcui.yaml (e.g. RDD, RIPE, LiftFeat, DaD).

    Also re-enables matchers that exist in OrthoLoC's config but are
    disabled (e.g. Mast3R has ``enable: false``)."""
    import importlib

    with _patch_datasets_import():
        _mod_real = importlib.import_module("ortholoc.image_matching.MatcherIMCUI")

    zoo = _mod_real.MATCHER_ZOO
    raw_cfg = _mod_real.CONFIG

    if name in zoo and name in raw_cfg.get("matcher_zoo", {}):
        return  # already present in both processed zoo and raw config

    # Load imcui's own config and merge missing entry
    try:
        with _patch_datasets_import():
            import imcui
            from imcui.ui.utils import get_matcher_zoo, load_config
            pkg = os.path.dirname(imcui.__file__)
            app_cfg = load_config(os.path.join(pkg, "config", "app.yaml"))
            app_zoo = get_matcher_zoo(app_cfg["matcher_zoo"])
    except ImportError:
        raise ImportError(
            f"Matcher '{name}' not in OrthoLoC config and imcui package "
            f"not available for fallback. Upgrade: pip install imcui>=0.0.4"
        )

    if name in app_zoo:
        zoo[name] = app_zoo[name]
        # Also patch the raw CONFIG dict — MatcherIMCUI.__init__ accesses
        # CONFIG['matcher_zoo'][name]['matcher'] for dense/feature matchers.
        raw_cfg = _mod_real.CONFIG
        if name not in raw_cfg.get("matcher_zoo", {}):
            raw_entry = app_cfg["matcher_zoo"].get(name)
            if raw_entry is not None:
                raw_cfg["matcher_zoo"][name] = raw_entry
        print(f"  [IMCUIMatcher] Registered '{name}' from imcui app.yaml")
    else:
        raise KeyError(
            f"Matcher '{name}' not found in OrthoLoC or imcui configs. "
            f"Available: {sorted(list(zoo.keys()) + list(app_zoo.keys()))}"
        )


class IMCUIMatcher(BaseMatcher):
    """Adapter that wraps OrthoLoC's MatcherIMCUI to the OrthoTrack interface.

    The underlying IMCUI matcher produces ``Correspondences2D2D`` (with
    normalized or pixel coordinates).  This class converts them into
    ``MatchResult`` objects with *pixel* coordinates — the format expected
    by the tracking pipeline.

    Supported matchers (non-exhaustive):
        Dense:  ``loftr``, ``roma``, ``Mast3R``, ``dkm``, ``xfeat``
        Sparse: ``superpoint+lightglue``, ``sift+lightglue``,
                ``superpoint+superglue``, ``disk+lightglue``, ``rdd``,
                ``xfeat+lightglue``, ``aliked+lightglue``

    Parameters
    ----------
    name : str
        Key into the IMCUI matcher zoo (see ``matchers_imcui.yaml``).
    device : str
        ``'cuda'`` or ``'cpu'``.
    extract_max_keypoints : int | None
        Cap on feature extraction (sparse matchers only).
    angles : list[float] | None
        Rotation angles to try.  ``None`` → matcher default.
    keypoint_threshold : float
        Detector threshold (sparse matchers only).
    use_rotation_matching : bool
        If ``True``, calls ``run()`` (multi-angle) rather than ``__call__()``
        and concatenates all correspondences.  Useful for Keyframe Localization
        where the DOP may be rotated relative to the query."""

    def __init__(
        self,
        name: str,
        device: str = "cuda",
        extract_max_keypoints: Optional[int] = None,
        angles: Optional[List[float]] = None,
        keypoint_threshold: float = 0.015,
        use_rotation_matching: bool = False,
    ):
        self.name = name
        self.use_rotation_matching = use_rotation_matching

        # Ensure matcher is registered (patches zoo for imcui>=0.0.4 matchers)
        _ensure_matcher_in_zoo(name)

        print(f"Initializing IMCUIMatcher ({name}) ...")
        t0 = time.time()
        with _patch_datasets_import():
            from ortholoc.image_matching.MatcherIMCUI import MatcherIMCUI as _MatcherIMCUI
            self._matcher = _MatcherIMCUI(
                name=name,
                device=device,
                extract_max_keypoints=extract_max_keypoints,
                angles=angles,
                keypoint_threshold=keypoint_threshold,
            )
        print(f"  IMCUIMatcher ({name}) loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    #  BaseMatcher interface
    # ------------------------------------------------------------------

    def match(
        self,
        query_image: np.ndarray,
        dop_image: np.ndarray,
        num_matches: int = 5000,
    ) -> MatchResult:
        query_h, query_w = query_image.shape[:2]
        dop_h, dop_w = dop_image.shape[:2]

        if self.use_rotation_matching:
            # Multi-angle matching → list of Correspondences2D2D
            corrs_list = self._matcher.run(
                query_image, dop_image,
                covisible_only=True, normalized=False, silent=True,
            )
            # Concatenate all rotations
            pts0_all, pts1_all, conf_all = [], [], []
            for corrs in corrs_list:
                if len(corrs) == 0:
                    continue
                # Ensure pixel coords
                if corrs.is_normalized:
                    corrs = corrs.denormalized(
                        h0=query_h, w0=query_w, h1=dop_h, w1=dop_w,
                    )
                pts0_all.append(corrs.pts0)
                pts1_all.append(corrs.pts1)
                conf = corrs.confidences if corrs.confidences is not None else np.ones(len(corrs))
                conf_all.append(conf)

            if pts0_all:
                pts0 = np.concatenate(pts0_all)
                pts1 = np.concatenate(pts1_all)
                confidences = np.concatenate(conf_all)
            else:
                pts0 = np.zeros((0, 2))
                pts1 = np.zeros((0, 2))
                confidences = np.zeros(0)
        else:
            # Single-angle matching
            try:
                corrs = self._matcher(
                    query_image, dop_image,
                    covisible_only=True, normalized=False, silent=True,
                )
                if corrs.is_normalized:
                    corrs = corrs.denormalized(
                        h0=query_h, w0=query_w, h1=dop_h, w1=dop_w,
                    )
                pts0 = corrs.pts0 if len(corrs) > 0 else np.zeros((0, 2))
                pts1 = corrs.pts1 if len(corrs) > 0 else np.zeros((0, 2))
                confidences = corrs.confidences if (len(corrs) > 0 and corrs.confidences is not None) else np.ones(max(len(corrs), 0))
            except (IndexError, Exception):
                # imcui scale_keypoints crashes on empty keypoint tensors
                pts0 = np.zeros((0, 2))
                pts1 = np.zeros((0, 2))
                confidences = np.zeros(0)

        # Down-sample to requested num_matches
        if len(pts0) > num_matches:
            # Prefer high-confidence matches
            idx = np.argsort(-confidences)[:num_matches]
            pts0 = pts0[idx]
            pts1 = pts1[idx]
            confidences = confidences[idx]

        return MatchResult(
            kpts_query=pts0.astype(np.float64),
            kpts_dop=pts1.astype(np.float64),
            confidences=confidences.astype(np.float64),
            query_size=(query_h, query_w),
            dop_size=(dop_h, dop_w),
        )

    def match_batch(
        self,
        query_image: np.ndarray,
        dop_images: List[np.ndarray],
        num_matches_per_crop: int = 3000,
    ) -> List[MatchResult]:
        """Sequentially match because IMCUI matchers lack native batching."""
        return [
            self.match(query_image, dop, num_matches=num_matches_per_crop)
            for dop in dop_images
        ]
