"""Explicit pipeline failures (no silent fallbacks)."""


class OrthoTrackError(RuntimeError):
    """Base class for OrthoTrack pipeline errors."""


class IntrinsicsRequiredError(OrthoTrackError):
    """Missing or unusable camera intrinsics (need intrinsics.json / meta or --force-calibration)."""


class VisibleCropError(OrthoTrackError):
    """DSM-based visible DOP footprint could not be computed."""


class InsufficientConfidentMatchesError(OrthoTrackError):
    """Not enough correspondences above the confidence threshold."""


class PnPSolverError(OrthoTrackError):
    """OpenCV PnP RANSAC failed or raised (e.g. degenerate configuration)."""


class InvalidGeometryError(OrthoTrackError):
    """Invalid altitude, FOV, or DSM sample for geometry helpers."""


class LocalizationGeometryError(OrthoTrackError):
    """Fine-stage geometry inconsistent (e.g. PnP vs coarse pose)."""


class FlowOutlierError(OrthoTrackError):
    """Tracked pose inconsistent with motion / previous frame."""


class KeyframeRotationRequiredError(OrthoTrackError):
    """Keyframe acceptance requires an estimated rotation."""


class KeyframeLocalizationError(OrthoTrackError):
    """Keyframe re-localization did not produce an accepted pose."""


class FirstFrameLocalizationError(OrthoTrackError):
    """First keyframe cannot be localized (coarse stage failed)."""
