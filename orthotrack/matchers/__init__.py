"""
OrthoTrack matcher package.

Provides a unified ``BaseMatcher`` interface and a ``create_matcher()`` factory
that instantiates any supported matcher by name.

Supported matcher names
-----------------------
RoMaV2 variants:
    ``roma_turbo``, ``roma_fast``, ``roma_base``, ``roma_precise``

IMCUI-based (via OrthoLoC's MatcherIMCUI + imcui >= 0.0.4 extensions):
    Dense:  ``loftr``, ``RoMa``, ``Mast3R``, ``GIM(dkm)``, ``dkm``,
            ``xfeat+lightglue``, ``xfeat(dense)``, ``rdd(dense)``,
            ``dad(RoMa)``, ``DUSt3R``, ``eloftr``, ``xoftr``,
            ``aspanformer``, ``topicfm``, ``omniglue``, ``gluestick``
    Sparse: ``superpoint+lightglue``, ``sift+lightglue``,
            ``superpoint+superglue``, ``disk+lightglue``,
            ``aliked+lightglue``, ``xfeat(sparse)``, ``dedode``,
            ``rdd(sparse)``, ``ripe(+mnn)``, ``liftfeat(sparse)``

Standalone:
    ``dune_mast3r`` -- DUNE encoder + MASt3R decoder (CVPR 2025)
    ``edm`` -- EDM sparse matcher (ResNet18 backbone)
    ``l2m`` -- L2M++ dense matcher (DINOv2 + RoMa)
    ``ufm`` / ``ufm_refine`` / ``ufm_980`` -- UFM dense flow matcher"""

from orthotrack.matchers.base_matcher import BaseMatcher
from orthotrack.matchers.roma_matcher import RoMaV2Matcher

__all__ = [
    "BaseMatcher",
    "RoMaV2Matcher",
    "create_matcher",
]

# Maps short aliases -> (class_key, kwargs)
_ROMA_SETTINGS = {
    "roma_turbo": "turbo",
    "roma_fast": "fast",
    "roma_base": "base",
    "roma_precise": "precise",
    "roma_lr800": "lr800",
    "roma_lr800_bidir": "lr800_bidir",
    "roma_turbo_lr256": "turbo_lr256",
    "roma_turbo_lr224": "turbo_lr224",
    # Short aliases
    "turbo": "turbo",
    "fast": "fast",
    "base": "base",
    "precise": "precise",
    "lr800": "lr800",
    "lr800_bidir": "lr800_bidir",
    "turbo_lr256": "turbo_lr256",
    "turbo_lr224": "turbo_lr224",
    "precise_unidir": "precise_unidir",
    "precise_1024": "precise_1024",
}


def create_matcher(
    name: str,
    device: str = "cuda",
    **kwargs,
) -> BaseMatcher:
    """Instantiate a matcher by name.

    Parameters
    ----------
    name : str
        Matcher identifier.  One of:
        - ``roma_turbo`` / ``roma_fast`` / ``roma_base`` / ``roma_precise``
          (or just ``turbo`` / ``fast`` / ``base`` / ``precise``)
        - ``dune_mast3r``
        - Any key from the IMCUI matcher zoo (e.g. ``superpoint+lightglue``,
          ``rdd(dense)``, ``ripe(+mnn)``, ``liftfeat(sparse)``,
          ``dad(RoMa)``)
    device : str
        ``'cuda'`` or ``'cpu'``.
    **kwargs
        Extra arguments forwarded to the matcher constructor.

    Returns
    -------
    BaseMatcher
        Ready-to-use matcher instance."""
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            device = "cpu"

    # --- RoMaV2 -----------------------------------------------------------
    if name in _ROMA_SETTINGS:
        setting = _ROMA_SETTINGS[name]
        return RoMaV2Matcher(setting=setting, **kwargs)

    # --- DUNE + MASt3R ----------------------------------------------------
    if name == "dune_mast3r":
        from orthotrack.matchers.dune_matcher import DuneMast3rMatcher
        return DuneMast3rMatcher(device=device, **kwargs)

    # --- EDM (sparse) -----------------------------------------------------
    if name == "edm":
        from orthotrack.matchers.edm_matcher import EDMMatcher
        return EDMMatcher(device=device, **kwargs)

    # --- L2M++ (dense) ----------------------------------------------------
    if name == "l2m":
        from orthotrack.matchers.l2m_matcher import L2MMatcher
        return L2MMatcher(device=device, **kwargs)

    # --- UFM (dense flow) -------------------------------------------------
    if name.startswith("ufm"):
        from orthotrack.matchers.ufm_matcher import UFMMatcher
        variant = "base"
        if name == "ufm_refine":
            variant = "refine"
        elif name == "ufm_980":
            variant = "base_980"
        return UFMMatcher(device=device, variant=variant, **kwargs)

    # --- IMCUI catch-all --------------------------------------------------
    from orthotrack.matchers.imcui_matcher import IMCUIMatcher
    return IMCUIMatcher(name=name, device=device, **kwargs)
