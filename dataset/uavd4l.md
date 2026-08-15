# UAVD4L Dataset

**UAVD4L** is a large-scale dataset designed for 6-DoF UAV localization in GPS-denied environments. It provides synthetic RGB, depth, DSM, and textured 3D reference models.

## Download
You can download the dataset from the official repository: [https://github.com/RingoWRW/UAVD4L](https://github.com/RingoWRW/UAVD4L)

## Preprocessing
* No complex preprocessing is required, but you will need to extract the downloaded image sequences (or combine them into an mp4 video) and identify the corresponding map files (DOP and DSM).
* Ensure that the provided ground-truth poses are parsed correctly if you plan to evaluate trajectory accuracy.

## Minimal Required Directory Structure
You only need the RGB images and the ground-truth poses. You can safely ignore the synthetic depth maps and 3D mesh models unless you need them for custom evaluations.
```text
UAVD4L_Minimal/
├── images/              # RGB images (e.g., from Query/images/ or Render_all/images/)
│   ├── 0001.png
│   └── ...
├── poses.txt            # Ground-truth poses (e.g., gt_pose.txt or db_pose.txt)
└── intrinsics.json      # Camera intrinsic parameters (OrthoTrack format)
```

## Running OrthoTrack

Provide georeferenced DOP/DSM GeoTIFFs with **≤ 0.2 m/px** GSD when possible (see root [README](../README.md)). Place `intrinsics.json` next to the footage or pass `--intrinsics`.

```bash
PYTHONPATH=. python scripts/run_tracking.py \
    --footage path/to/uavd4l/sequence_dir \
    --dop path/to/uavd4l/map.tif \
    --dsm path/to/uavd4l/dsm.tif \
    --output outputs/uavd4l_results \
    --fine_matcher precise
```

Use `--fine_matcher turbo` on GPUs with ~8 GB VRAM. Keep `--skip_frames 1` for best optical-flow quality.

## Benchmarking

1. Run tracking to produce `results.csv`.
2. Align estimates to the dataset ground truth (handle CRS differences if needed).
3. Evaluate with [evo](https://github.com/MichaelGrupp/evo) (ATE, RPE, etc.).
