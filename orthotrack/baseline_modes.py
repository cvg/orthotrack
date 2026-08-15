"""
Baseline tracking modes used for paper ablations.

- ``run_localize_every_frame``: localizes every frame independently (no tracking).
- ``run_dsm_tracking_only``: localizes the first frame, then tracks via
  DSM-projected optical flow (no keyframe re-triggering).

Both accept a ``TrackingPipeline`` instance as the first argument so they can
reuse the shared ``_localize_full_pipeline`` helper and pipeline state."""

import numpy as np
import cv2
import time
import random
import torch

from typing import List
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from orthotrack import localization as loc
from orthotrack import evaluation as evl
from orthotrack.types import FrameResult
from orthotrack.exceptions import (
    FirstFrameLocalizationError,
    InsufficientConfidentMatchesError,
)


class _NullGTPose:
    x = y = z = None
    qw = qx = qy = qz = None


def run_localize_every_frame(pipeline, frame_indices: List[int],
                             verbose: bool = False) -> List[FrameResult]:
    """Localize every frame independently (no tracking).

    Measures the accuracy ceiling of the localization front-end
    at the cost of speed (one matcher invocation per frame)."""
    np.random.seed(42); torch.manual_seed(42); random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    cv2.setRNGSeed(42)

    results: List[FrameResult] = []
    pipeline.last_localized_pos = None
    pipeline.prev_position = None
    pipeline.prev_R_c2w = None

    for i, frame_id in enumerate(tqdm(frame_indices, desc="Localizing every frame")):
        start_time = time.time()
        image = pipeline.load_image(frame_id)
        gt_pose = pipeline.gt_reader.get_pose(frame_id) if pipeline.gt_reader is not None else _NullGTPose()
        h, w = image.shape[:2]

        # First frame (no prior): coarse + fine
        # Subsequent frames: fine-only with tracked prior
        position = None
        num_inliers = 0
        est_rotation = None
        pts_2d = np.zeros((0, 2))

        try:
            position, est_rotation, num_inliers, pts_2d, pts_3d, confs, crop_spec, _ = \
                pipeline._localize_full_pipeline(
                    frame_id, image, h, w, verbose=verbose,
                    tracked_prior=pipeline.prev_position,
                )
        except (FirstFrameLocalizationError, InsufficientConfidentMatchesError) as e:
            if verbose:
                print(f"  Localization error: {e}")

        proc_time = time.time() - start_time
        if position is not None:
            errors = evl.compute_errors(position, gt_pose, est_rotation)
            quat = None
            if est_rotation is not None:
                try:
                    quat = Rotation.from_matrix(est_rotation).as_quat()
                except Exception:
                    pass
            results.append(FrameResult(
                frame_id=frame_id, is_keyframe=True, success=True,
                est_x=position[0], est_y=position[1], est_z=position[2],
                est_qw=quat[3] if quat is not None else None,
                est_qx=quat[0] if quat is not None else None,
                est_qy=quat[1] if quat is not None else None,
                est_qz=quat[2] if quat is not None else None,
                gt_x=gt_pose.x, gt_y=gt_pose.y, gt_z=gt_pose.z,
                gt_qw=gt_pose.qw, gt_qx=gt_pose.qx, gt_qy=gt_pose.qy, gt_qz=gt_pose.qz,
                position_error=errors['position_error'],
                horizontal_error=errors['horizontal_error'],
                vertical_error=errors['vertical_error'],
                rotation_error=errors['rotation_error'],
                num_tracked_points=len(pts_2d), num_inliers=num_inliers,
                tracked_points_threshold=0, method="keyframe",
                processing_time=proc_time,
            ))
            pipeline.prev_position = position.copy()
            pipeline.prev_R_c2w = est_rotation
            pipeline.last_keyframe_R_c2w = est_rotation
        else:
            results.append(FrameResult(
                frame_id=frame_id, is_keyframe=True, success=False,
                gt_x=gt_pose.x, gt_y=gt_pose.y, gt_z=gt_pose.z,
                gt_qw=gt_pose.qw, gt_qx=gt_pose.qx, gt_qy=gt_pose.qy, gt_qz=gt_pose.qz,
                method="failed", processing_time=proc_time,
            ))

    return results


def run_dsm_tracking_only(pipeline, frame_indices: List[int],
                          verbose: bool = False) -> List[FrameResult]:
    """First-frame localization + DSM-projected optical flow tracking.

    After the first frame is localized, subsequent frames generate 2D-3D
    correspondences by projecting DSM points using the previous pose, then
    tracking via Lucas-Kanade optical flow with forward-backward check."""
    np.random.seed(42); torch.manual_seed(42); random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    cv2.setRNGSeed(42)

    results: List[FrameResult] = []
    pipeline.last_localized_pos = None
    pipeline.prev_position = None
    pipeline.prev_R_c2w = None
    prev_gray = None
    first_frame_done = False

    lk_params = dict(
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    fb_threshold = 1.0

    for i, frame_id in enumerate(tqdm(frame_indices, desc="DSM tracking only")):
        start_time = time.time()
        image = pipeline.load_image(frame_id)
        gt_pose = pipeline.gt_reader.get_pose(frame_id) if pipeline.gt_reader is not None else _NullGTPose()
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # ---- First frame: full localization ----
        if not first_frame_done:
            if verbose:
                print(f"\n[Frame {frame_id}] First frame — full localization")

            position = None
            num_inliers = 0
            est_rotation = None
            n_pts = 0

            try:
                position, est_rotation, num_inliers, pts_2d, pts_3d, confs, crop_spec, _ = \
                    pipeline._localize_full_pipeline(
                        frame_id, image, h, w, verbose=verbose,
                    )
                n_pts = len(pts_2d)
            except (FirstFrameLocalizationError, InsufficientConfidentMatchesError) as e:
                if verbose:
                    print(f"  First frame error: {e}")

            proc_time = time.time() - start_time
            if position is not None:
                first_frame_done = True
                pipeline.prev_position = position.copy()
                pipeline.prev_R_c2w = est_rotation
                prev_gray = gray
                errors = evl.compute_errors(position, gt_pose, est_rotation)
                quat = None
                if est_rotation is not None:
                    try:
                        quat = Rotation.from_matrix(est_rotation).as_quat()
                    except Exception:
                        pass
                results.append(FrameResult(
                    frame_id=frame_id, is_keyframe=True, success=True,
                    est_x=position[0], est_y=position[1], est_z=position[2],
                    est_qw=quat[3] if quat is not None else None,
                    est_qx=quat[0] if quat is not None else None,
                    est_qy=quat[1] if quat is not None else None,
                    est_qz=quat[2] if quat is not None else None,
                    gt_x=gt_pose.x, gt_y=gt_pose.y, gt_z=gt_pose.z,
                    gt_qw=gt_pose.qw, gt_qx=gt_pose.qx, gt_qy=gt_pose.qy, gt_qz=gt_pose.qz,
                    position_error=errors['position_error'],
                    horizontal_error=errors['horizontal_error'],
                    vertical_error=errors['vertical_error'],
                    rotation_error=errors['rotation_error'],
                    num_tracked_points=n_pts,
                    num_inliers=num_inliers,
                    tracked_points_threshold=0, method="keyframe",
                    processing_time=proc_time,
                ))
            else:
                print(f"  ERROR: First frame localization failed!")
                results.append(FrameResult(
                    frame_id=frame_id, is_keyframe=True, success=False,
                    gt_x=gt_pose.x, gt_y=gt_pose.y, gt_z=gt_pose.z,
                    method="failed", processing_time=proc_time,
                ))
                if pipeline.stop_on_first_frame_failure:
                    print("  stop_on_first_frame_failure is set — aborting sequence.")
                    break
            continue

        # ---- Subsequent frames: DSM-projected + optical flow tracking ----
        if pipeline.prev_position is None or pipeline.prev_R_c2w is None or prev_gray is None:
            results.append(FrameResult(
                frame_id=frame_id, is_keyframe=False, success=False,
                gt_x=gt_pose.x, gt_y=gt_pose.y, gt_z=gt_pose.z,
                method="failed", processing_time=time.time() - start_time,
            ))
            continue

        dsm_pts_2d, dsm_pts_3d = loc.generate_dsm_correspondences(
            pipeline.prev_position, pipeline.prev_R_c2w,
            (h, w), pipeline.intrinsics.fov_vertical, pipeline.geo_handler,
            verbose=verbose,
            K=pipeline.intrinsics.K,
        )

        if len(dsm_pts_2d) < 20:
            if verbose:
                print(f"  [Frame {frame_id}] DSM projection yielded too few points ({len(dsm_pts_2d)})")
            results.append(FrameResult(
                frame_id=frame_id, is_keyframe=False, success=False,
                gt_x=gt_pose.x, gt_y=gt_pose.y, gt_z=gt_pose.z,
                method="failed", processing_time=time.time() - start_time,
            ))
            prev_gray = gray
            continue

        pts_prev = dsm_pts_2d.reshape(-1, 1, 2).astype(np.float32)
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, pts_prev, None, **lk_params
        )
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, prev_gray, new_pts, None, **lk_params
        )

        fb_error = np.linalg.norm(
            pts_prev.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1
        )
        status_flat = status.flatten()
        back_status_flat = back_status.flatten()
        good_mask = (
            (status_flat == 1) & (back_status_flat == 1) &
            (fb_error < fb_threshold)
        )
        tracked_2d = new_pts[good_mask].reshape(-1, 2)
        tracked_3d = dsm_pts_3d[good_mask]

        if verbose:
            print(f"  [Frame {frame_id}] DSM→LK: {len(dsm_pts_2d)} projected, "
                  f"{good_mask.sum()} tracked (FB<{fb_threshold}px)")

        if len(tracked_2d) < 20:
            if verbose:
                print(f"  [Frame {frame_id}] Too few tracked points ({len(tracked_2d)})")
            results.append(FrameResult(
                frame_id=frame_id, is_keyframe=False, success=False,
                gt_x=gt_pose.x, gt_y=gt_pose.y, gt_z=gt_pose.z,
                method="failed", processing_time=time.time() - start_time,
            ))
            prev_gray = gray
            continue

        position, num_inliers, reproj_error, est_rotation = \
            loc.estimate_pose_from_2d3d_corrspondences(
                tracked_2d, tracked_3d, (h, w), pipeline.intrinsics.fov_vertical,
                verbose=verbose, reproj_threshold=pipeline.pnp_reproj_threshold,
                K=pipeline.intrinsics.K,
            )

        proc_time = time.time() - start_time
        if position is not None and reproj_error < 20.0:
            errors = evl.compute_errors(position, gt_pose, est_rotation)
            quat = None
            if est_rotation is not None:
                try:
                    quat = Rotation.from_matrix(est_rotation).as_quat()
                except Exception:
                    pass
            results.append(FrameResult(
                frame_id=frame_id, is_keyframe=False, success=True,
                est_x=position[0], est_y=position[1], est_z=position[2],
                est_qw=quat[3] if quat is not None else None,
                est_qx=quat[0] if quat is not None else None,
                est_qy=quat[1] if quat is not None else None,
                est_qz=quat[2] if quat is not None else None,
                gt_x=gt_pose.x, gt_y=gt_pose.y, gt_z=gt_pose.z,
                gt_qw=gt_pose.qw, gt_qx=gt_pose.qx, gt_qy=gt_pose.qy, gt_qz=gt_pose.qz,
                position_error=errors['position_error'],
                horizontal_error=errors['horizontal_error'],
                vertical_error=errors['vertical_error'],
                rotation_error=errors['rotation_error'],
                num_tracked_points=len(tracked_2d), num_inliers=num_inliers,
                tracked_points_threshold=0, method="tracked",
                reproj_error=reproj_error,
                processing_time=proc_time,
            ))
            pipeline.prev_position = position.copy()
            pipeline.prev_R_c2w = est_rotation
        else:
            if verbose:
                print(f"  [Frame {frame_id}] DSM tracking PnP failed")
            results.append(FrameResult(
                frame_id=frame_id, is_keyframe=False, success=False,
                gt_x=gt_pose.x, gt_y=gt_pose.y, gt_z=gt_pose.z,
                method="failed", processing_time=proc_time,
            ))

        prev_gray = gray

    return results
