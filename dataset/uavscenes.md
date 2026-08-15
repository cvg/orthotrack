# UAVScenes Dataset

**UAVScenes** (built upon MARS-LVIG) provides frame-wise semantic annotations, 6-DoF poses, and multi-modal data including 2D camera images and 3D LiDAR point clouds.

## Download
Available via the official repository: [https://github.com/sijieaaa/UAVScenes](https://github.com/sijieaaa/UAVScenes)

## Preprocessing
* Extract the RGB images and the ground-truth trajectory.
* Since OrthoTrack requires an orthophoto (DOP) and surface model (DSM), you must either use the provided map data or download public GIS data (e.g., GeoTIFFs) corresponding to the bounding box of the UAVScenes trajectories.

## Minimal Required Directory Structure
You only need the camera images and poses. You can safely ignore the LiDAR point clouds (`interval1_LIDAR`), semantic labels (`*_label`), and raw Terra mesh outputs.
```text
UAVScenes_Minimal/
├── images/              # RGB camera images (from interval1_CAM_LIDAR/)
│   ├── 0001.jpg
│   └── ...
├── poses.txt            # Extracted 6-DoF poses
└── intrinsics.json      # Calibration parameters (OrthoTrack format)
```

## Running OrthoTrack

Provide DOP/DSM GeoTIFFs covering the flight at **≤ 0.2 m/px** when possible. Place `intrinsics.json` next to the footage or pass `--intrinsics`.

```bash
PYTHONPATH=. python scripts/run_tracking.py \
    --footage path/to/uavscenes/images \
    --dop path/to/uavscenes/dop.tif \
    --dsm path/to/uavscenes/dsm.tif \
    --output outputs/uavscenes_results
```

Use `--fine_matcher turbo` on ~8 GB GPUs. Keep `--skip_frames 1` unless you intentionally want faster, lower-quality runs.

## Benchmarking

1. Run tracking to produce `results.csv`.
2. Align estimates to the dataset ground truth (handle CRS differences if needed).
3. Evaluate with [evo](https://github.com/MichaelGrupp/evo) (ATE, RPE, etc.).
