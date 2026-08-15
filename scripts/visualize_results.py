#!/usr/bin/env python3
"""
Overlay LoD2 mesh on video frames using tracking results (results.json).

Uses ``orthotrack.lod.LoD`` for GPU-accelerated edge rendering."""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

try:
    from decord import VideoReader, cpu as decord_cpu
    _DECORD = True
except ImportError:
    _DECORD = False

from utils.lod import LoD


def main():
    parser = argparse.ArgumentParser(description="Overlay LoD2 mesh on video using tracking poses")
    parser.add_argument("--sequence_dir", default=None, help="Sequence directory with video.mp4 and lod2.npz")
    parser.add_argument("--video", default=None, help="Path to custom video file")
    parser.add_argument("--lod", default=None, help="Path to custom LoD mesh (.npz, .obj, .ply)")
    parser.add_argument("--results", required=True, help="Path to results.json from tracking")
    parser.add_argument("--output", "-o", required=True, help="Output video path (.mp4)")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--fov", type=float, default=60.0, help="Vertical FoV in degrees")
    parser.add_argument("--edge_color", type=str, default="0,255,0",
                        help="Edge color as B,G,R (default green: 0,255,0)")
    parser.add_argument("--edge_alpha", type=float, default=0.85, help="Edge overlay alpha [0-1]")
    parser.add_argument("--radius", type=float, default=300.0, help="Spatial filter margin around trajectory (m)")
    parser.add_argument("--max_image_dim", type=int, default=1920, help="Max image dimension (0=no resize)")
    parser.add_argument("--smooth", type=int, default=5, help="Pose smoothing window (0=no smoothing)")
    args = parser.parse_args()

    if args.sequence_dir:
        seq_dir = Path(args.sequence_dir)
        lod_path = seq_dir / "lod2.npz"
        video_path = seq_dir / "video.mp4"
    else:
        if not args.video or not args.lod:
            print("ERROR: Must provide either --sequence_dir OR both --video and --lod")
            return
        lod_path = Path(args.lod)
        video_path = Path(args.video)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    edge_color_bgr = tuple(int(c) for c in args.edge_color.split(','))  # (B, G, R)
    edge_color_rgb = (edge_color_bgr[2], edge_color_bgr[1], edge_color_bgr[0])

    # ── Load LoD mesh ──
    if not lod_path.exists():
        print(f"ERROR: {lod_path} not found")
        return
    lod_full = LoD.from_file(lod_path)
    print(f"LoD2: {lod_full.num_vertices} vertices, {lod_full.num_faces} faces")

    # Load tracking results
    with open(args.results) as f:
        results_data = json.load(f)
    frames = results_data['frames']
    print(f"Loaded {len(frames)} frame results")

    # Build pose lookup: frame_id -> (position, R_c2w)
    pose_lookup = {}
    for fr in frames:
        if not fr['success'] or fr['est_x'] is None:
            continue
        pos = np.array([fr['est_x'], fr['est_y'], fr['est_z']])
        qw, qx, qy, qz = fr['est_qw'], fr['est_qx'], fr['est_qy'], fr['est_qz']
        if qw is None:
            continue
        R_c2w = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        pose_lookup[fr['frame_id']] = (pos, R_c2w)

    # ── Spatial pre-filtering: keep only geometry near camera trajectory ──
    lod = lod_full
    if pose_lookup:
        all_positions = np.array([p[0] for p in pose_lookup.values()])
        traj_center = all_positions.mean(axis=0)
        traj_extent = all_positions.max(axis=0) - all_positions.min(axis=0)
        radius = max(traj_extent[:2].max(), 200.0) + args.radius

        verts_abs = lod_full.vertices_abs
        face_centroids = verts_abs[lod_full.faces].mean(axis=1)
        dist_xy = np.sqrt((face_centroids[:, 0] - traj_center[0])**2 +
                          (face_centroids[:, 1] - traj_center[1])**2)
        face_mask = dist_xy < radius
        kept_faces = lod_full.faces[face_mask]
        kept_labels = (lod_full.labels[face_mask]
                       if lod_full.labels is not None and len(lod_full.labels) > 0
                       else None)

        # Reindex vertices for the filtered LoD
        used_verts = np.unique(kept_faces.ravel())
        vert_remap = np.full(len(verts_abs), -1, dtype=np.int64)
        vert_remap[used_verts] = np.arange(len(used_verts))
        filtered_verts = verts_abs[used_verts] - lod_full.utm_offset  # back to local coords
        kept_faces_reindexed = vert_remap[kept_faces].astype(np.int32)
        lod = LoD.from_arrays(
            filtered_verts.astype(np.float64),
            kept_faces_reindexed,
            labels=kept_labels,
            utm_offset=lod_full.utm_offset,
        )
        print(f"Spatial filter: {lod.num_vertices}/{lod_full.num_vertices} verts, "
              f"{lod.num_faces}/{lod_full.num_faces} faces (r={radius:.0f}m)")

    print(f"Mesh ready: {lod.num_faces} faces, {lod.num_vertices} vertices")

    # ── Video I/O ──
    if not video_path.exists():
        print(f"ERROR: {video_path} not found")
        return

    if _DECORD:
        vr = VideoReader(str(video_path), ctx=decord_cpu(0))
        total_frames = len(vr)
        vid_fps = vr.get_avg_fps()
    else:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_fps = cap.get(cv2.CAP_PROP_FPS)

    fps = vid_fps if vid_fps > 0 else args.fps
    print(f"Video: {total_frames} frames, {vid_fps:.1f} fps")

    # First frame → image dims
    if _DECORD:
        sample = vr[0].asnumpy()
    else:
        ret, sample = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    img_h, img_w = sample.shape[:2]
    scale = 1.0
    if args.max_image_dim > 0 and max(img_h, img_w) > args.max_image_dim:
        scale = args.max_image_dim / max(img_h, img_w)
        img_w = int(img_w * scale)
        img_h = int(img_h * scale)

    # Intrinsics from FoV
    fov_vertical = args.fov
    fy = img_h / (2 * np.tan(np.radians(fov_vertical) / 2))
    fx = fy
    K = np.array([[fx, 0, img_w / 2.0],
                   [0, fy, img_h / 2.0],
                   [0, 0, 1]], dtype=np.float64)
    print(f"FoV={fov_vertical}°, fx={fx:.1f}, image={img_w}x{img_h}")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (img_w, img_h))

    frame_ids = sorted(pose_lookup.keys())
    if not frame_ids:
        print("No successful poses found")
        return

    # ── Temporal pose smoothing ──
    if args.smooth > 1 and len(frame_ids) >= args.smooth:
        positions = np.array([pose_lookup[fid][0] for fid in frame_ids])
        rotations = Rotation.from_matrix(np.array([pose_lookup[fid][1] for fid in frame_ids]))

        for dim in range(3):
            positions[:, dim] = uniform_filter1d(positions[:, dim], size=args.smooth, mode='nearest')

        quats = rotations.as_quat()
        smoothed_quats = np.empty_like(quats)
        half_w = args.smooth // 2
        for i in range(len(quats)):
            lo = max(0, i - half_w)
            hi = min(len(quats), i + half_w + 1)
            window = quats[lo:hi]
            signs = np.sign(np.sum(window * window[len(window) // 2], axis=1, keepdims=True))
            signs[signs == 0] = 1
            avg = (window * signs).mean(axis=0)
            avg /= np.linalg.norm(avg)
            smoothed_quats[i] = avg

        smoothed_rots = Rotation.from_quat(smoothed_quats)
        for i, fid in enumerate(frame_ids):
            pose_lookup[fid] = (positions[i], smoothed_rots[i].as_matrix())
        print(f"Pose smoothing: window={args.smooth} frames")

    max_frame = max(pose_lookup.keys()) if pose_lookup else -1
    render_frames = min(total_frames, max_frame + 1)
    
    for frame_id in tqdm(range(render_frames), desc="Rendering"):
        if _DECORD:
            img = vr[frame_id].asnumpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            ret, img = cap.read()
            if not ret:
                break

        if scale < 1.0:
            img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)

        if frame_id in pose_lookup:
            pos, R_c2w = pose_lookup[frame_id]
            R_w2c = R_c2w.T
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = R_w2c
            w2c[:3, 3] = -R_w2c @ pos

            edge_map = lod.render_edges(w2c, K, (img_h, img_w))
            edge_mask = cv2.dilate(
                (edge_map > 0.5).astype(np.uint8), np.ones((2, 2), np.uint8)
            )
            color_layer = np.zeros_like(img)
            color_layer[edge_mask > 0] = edge_color_bgr
            cv2.addWeighted(color_layer, args.edge_alpha, img, 1.0, 0, img)

        writer.write(img)

    writer.release()
    if not _DECORD and 'cap' in dir() and cap is not None:
        cap.release()

    print(f"\nSaved: {output_path} ({total_frames} frames)")


if __name__ == "__main__":
    main()
