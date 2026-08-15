"""
orthotrack/depth_estimators/__init__.py
=======================================
Depth estimator registry for OrthoTrack."""

from orthotrack.depth_estimators.base_depth_estimator import BaseDepthEstimator

# Registry: estimator_key -> (class_path, variant_kwargs)
# Loaded lazily to avoid importing all heavy frameworks at startup.
DEPTH_ESTIMATOR_REGISTRY = {
    # -------------------------------------------------------------------------
    # Metric depth estimators (is_metric=True)
    # -------------------------------------------------------------------------
    'depth_pro': {
        'class': 'orthotrack.depth_estimators.depth_pro_estimator.DepthProEstimator',
        'kwargs': {},
        'metric': True,
        'paper': 'Depth Pro (Apple, ICLR 2025)',
        'citation': 'bochkovskii2024depthpro',
    },
    'moge2': {
        'class': 'orthotrack.depth_estimators.moge2_estimator.MoGe2Estimator',
        'kwargs': {},
        'metric': True,
        'paper': 'MoGe-2 (Microsoft, arXiv 2025)',
        'citation': 'wang2025moge2',
    },
    'unidepth_v2': {
        'class': 'orthotrack.depth_estimators.unidepth_estimator.UniDepthEstimator',
        'kwargs': {'variant': 'vitl14'},
        'metric': True,
        'paper': 'UniDepth v2 (ETH, arXiv 2025)',
        'citation': 'piccinelli2025unidepthv2',
    },
    'metric3d_v2': {
        'class': 'orthotrack.depth_estimators.metric3d_estimator.Metric3DEstimator',
        'kwargs': {'variant': 'large'},
        'metric': True,
        'paper': 'Metric3D v2 (TPAMI 2024)',
        'citation': 'hu2024metric3dv2',
    },
    'da2_metric': {
        'class': 'orthotrack.depth_estimators.da2_estimator.DepthAnythingV2MetricEstimator',
        'kwargs': {'variant': 'large'},
        'metric': True,
        'paper': 'Depth Anything V2 Metric-Outdoor (NeurIPS 2024)',
        'citation': 'yang2024depthanythingv2',
    },
    # -------------------------------------------------------------------------
    # Relative depth estimators (is_metric=False; requires scale alignment)
    # -------------------------------------------------------------------------
    'da2_relative': {
        'class': 'orthotrack.depth_estimators.da2_estimator.DepthAnythingV2Estimator',
        'kwargs': {'variant': 'large'},
        'metric': False,
        'paper': 'Depth Anything V2 Relative (NeurIPS 2024)',
        'citation': 'yang2024depthanythingv2',
    },
    'marigold': {
        'class': 'orthotrack.depth_estimators.marigold_estimator.MarigoldEstimator',
        'kwargs': {'n_steps': 10, 'ensemble_size': 1},
        'metric': False,
        'paper': 'Marigold (ETH, CVPR 2024)',
        'citation': 'ke2024marigold',
    },
    'da3': {
        'class': 'orthotrack.depth_estimators.da3_estimator.DepthAnything3Estimator',
        'kwargs': {'variant': 'large'},
        'metric': False,
        'paper': 'Depth Anything 3 (ByteDance, arXiv 2025)',
        'citation': 'lin2025depthanything3',
    },
}


def get_depth_estimator(name: str, device: str = 'cuda', **override_kwargs) -> BaseDepthEstimator:
    """Instantiate a depth estimator by registry key.

    Args:
        name:    Estimator name from DEPTH_ESTIMATOR_REGISTRY.
        device:  Target device string.
        **override_kwargs: Override any default kwargs.

    Returns:
        Instantiated BaseDepthEstimator subclass."""
    import importlib

    if name not in DEPTH_ESTIMATOR_REGISTRY:
        raise ValueError(
            f"Unknown depth estimator: '{name}'.\n"
            f"Available: {list(DEPTH_ESTIMATOR_REGISTRY.keys())}"
        )

    entry = DEPTH_ESTIMATOR_REGISTRY[name]
    module_path, class_name = entry['class'].rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    kwargs = dict(entry['kwargs'])
    kwargs['device'] = device
    kwargs.update(override_kwargs)

    return cls(**kwargs)


__all__ = ['BaseDepthEstimator', 'DEPTH_ESTIMATOR_REGISTRY', 'get_depth_estimator']
