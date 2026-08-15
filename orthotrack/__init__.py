"""
orthotrack — UAV tracking pipeline using orthographic geodata.

Re-exports the main public API so callers can do:

    from orthotrack import TrackingPipeline, FeatureTracker, FrameResult"""

from orthotrack.pipeline import TrackingPipeline
from orthotrack.feature_tracker import FeatureTracker
from orthotrack.types import FrameResult

__all__ = [
    "TrackingPipeline",
    "FeatureTracker",
    "FrameResult",
]
