# Evaluation & Benchmarking (optional)

These scripts compare foundation models, VO/SLAM baselines, and monocular depth estimators on MovingDrone. They are **not required** to install or run OrthoTrack tracking.

**Required first:** follow the root [README](README.md) install (`bash setup.sh`) and download MovingDrone scenes under `data/MovingDrone/scenes/` (see [dataset/movingdrone.md](dataset/movingdrone.md)).

Each section below needs **extra** third-party repos and weights. Commands will fail until those deps are installed.

---

## 1. Foundation Models Benchmarking

Script: `scripts/run_foundation_models.py`

### Supported Models
- **MapAnything**: `mapanything` — use `--provide_intrinsics`, `--provide_depth`, and/or `--provide_poses` for oracle inputs (output names get `_K` / `_K+D` / `_K+D+P` suffixes)
- **Point-Tracking / SfM**: `dust3r`, `mast3r`, `metric_dust3r`, `must3r`, `pow3r`, `pow3r_ba`
- **Monocular Geometry / Depth**: `pi3`, `pi3x`, `moge_1`, `moge_2`, `da3`, `da3_nested`
- **SLAM (via external subprocesses)**: `vggt_slam_v1`, `vggt_slam_v2`, `vggt_long`

See `scripts/run_foundation_models.py` for the full model list.

### Setup (required before running)
```bash
bash scripts/setup_foundation_models.sh
```
This prints clone/install steps for MapAnything, DA3, and Pi3 into `thirdparty/`. MoGe / DUSt3R-style weights typically download from Hugging Face at runtime **after** the corresponding packages are available.

### Usage
`--data_dir` must be the local MovingDrone **scenes** folder:

```bash
PYTHONPATH=. python scripts/run_foundation_models.py \
    --sequences airport10 suburban9 \
    --models mast3r moge_2 \
    --data_dir data/MovingDrone/scenes \
    --output_dir outputs/foundation_bench \
    --device cuda
```

MapAnything with oracle intrinsics + depth:
```bash
PYTHONPATH=. python scripts/run_foundation_models.py \
    --sequence airport10 \
    --models mapanything \
    --data_dir data/MovingDrone/scenes \
    --provide_intrinsics --provide_depth
```

---

## 2. Visual Odometry (VO) & SLAM Baselines

Script: `scripts/run_vo_baselines.py`

### Supported Methods
- Built-in / lighter: `five_point_vo`, `five_point_vo_fast`
- External (must install separately): `droid_slam`, `dpvo`, `dpv_slam`

DROID-SLAM / DPVO expect clones under `thirdparty/` (see error messages in `orthotrack/baselines/vo_wrapper.py` for clone + pip steps). They are **not** installed by `setup.sh`.

### Usage
```bash
PYTHONPATH=. python scripts/run_vo_baselines.py \
    --sequences airport10 \
    --methods five_point_vo \
    --data_root data/MovingDrone \
    --output_dir outputs/vo_bench \
    --table_mode all
```

For `droid_slam` / `dpvo`, install their third-party stacks first, then pass those method names.

---

## 3. Depth Estimation Benchmarking

Script: `scripts/run_depth_benchmark.py`

Requires the chosen estimator’s package/weights (e.g. MoGe) and per-frame GT depth under each scene’s `depth/` folder (downloaded when `load_depth=True` on `MovingDrone`, or present after a full scene download).

```bash
PYTHONPATH=. python scripts/run_depth_benchmark.py \
    --estimator moge2 \
    --sequences airport10 \
    --stride 10 \
    --output_dir outputs/depth_bench \
    --device cuda
```

List estimators: `--list`.

---

## 4. External SLAM (VGGT-*) Dependencies

For VGGT-SLAM / VGGT-Long used via `run_foundation_models.py`:

```bash
bash scripts/setup_slam_baselines.sh
```

This clones repos into `thirdparty/` and prints conda env setup. You must create those environments manually. Set:

```bash
export VGGT_SLAM_V1_DIR=$PWD/thirdparty/VGGT-SLAM-v1
export VGGT_SLAM_V2_DIR=$PWD/thirdparty/VGGT-SLAM
export VGGT_LONG_DIR=$PWD/thirdparty/VGGT-Long
```
