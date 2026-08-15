# MovingDrone Dataset

OrthoTrack uses the **MovingDrone Dataset** — a large-scale benchmark with photorealistic UAV sequences, dense 6-DoF ground truth, and co-registered multi-temporal orthophotos and DSMs.

## Local layout

Each scene is a directory:

```text
data/MovingDrone/scenes/<name>/
    video.mp4          # UAV footage
    poses.csv          # Ground-truth poses (UTM-local after utm_offset)
    intrinsics.json    # Camera calibration
    meta.json          # Sequence metadata (geodata paths, GPS bounding box, ...)
    dsm.npz            # Pre-extracted elevation prior
    dop/               # Multi-year orthophoto crops (e.g. 2013.jpg, 2020.jpg, ...)
    lod1.npz / lod2.npz  # Optional building LoD meshes
```

The dataset root also contains `splits.json` (`train` / `val` / `test_inPlace` / `test_outPlace`).

> **Naming note:** On disk the folder is `scenes/`. Downloads come from the academic server path `…/MovingDrone/sequences/<name>/` (same files, different remote folder name). Always point OrthoTrack scripts at the local `scenes/` path.

## Download

The `MovingDrone` class downloads scenes automatically (on-the-fly, or upfront with `predownload=True`). That includes video, poses, intrinsics, and pre-extracted map priors (`dop/` crops and `dsm.npz` at **0.2 m/px** GSD).

Default cache: `~/.cache/movingdrone`. Override with `MOVINGDRONE_DIR` or `dataset_dir=...`.

**Quick download of one scene:**

```bash
PYTHONPATH=. python -c "
from dataset.MovingDrone import MovingDrone
MovingDrone(
    dataset_dir='data/MovingDrone',
    sequences=['airport10'],
    predownload=True,
    load_dop=True,
    load_dsm=True,
)
"
```

## Using the Dataset Class

```python
from dataset.MovingDrone import MovingDrone
from torch.utils.data import DataLoader

dataset = MovingDrone(
    dataset_dir="data/MovingDrone",  # optional; default ~/.cache/movingdrone
    split="train",
    sequence_length=1,        # 1 = single-frame, >1 = temporal windows
    stride=10,
    predownload=False,        # False = download on first access
    load_dop=True,
    load_dsm=True,
    load_depth=False,         # True fetches per-frame depth (large)
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=MovingDrone.collate_fn,
    shuffle=True,
)
```

## Creating new scenes

Use `dataset/create_movingdrone.py` to turn a **Google Earth Studio (GES) trajectory JSON** into a full MovingDrone scene under `data/MovingDrone/scenes/<name>/`. Run from the repo root with `PYTHONPATH=.` after `bash setup.sh`.

```bash
PYTHONPATH=. python dataset/create_movingdrone.py --help
```

### End-to-end pipeline

```text
Google Earth Studio  →  trajectory JSON (--input)
                              ↓
         ┌────────────────────┴────────────────────┐
         │  velocity bounding (automatic)          │
         │  optional realism/noise (--render)      │
         └────────────────────┬────────────────────┘
                              ↓
              mesh render (--render --mesh-dir / --auto-geodata)
                              ↓
           video.mp4  poses.csv  intrinsics.json  depth/
                              ↓
         crop / download DOP, DSM, LoD, LiDAR, mesh.npz
                              ↓
              MovingDrone scene folder (all modalities)
```

**What the script produces**

| Output | Description |
|--------|-------------|
| `video.mp4` | UAV footage (mesh-rendered or from `--frames-dir` / `--video`) |
| `poses.csv` | Dense 6-DoF GT (position + quaternion), UTM-local after `utm_offset` |
| `intrinsics.json` | `fx, fy, cx, cy` derived from GES resolution + vertical FOV |
| `meta.json` | Region name, bbox, `utm_offset`, dop/dsm metadata |
| `depth/` | Per-frame GT depth maps (when `--render`, default on) |
| `dsm.npz` | Cropped elevation prior at **0.2 m/px** |
| `dop/*.jpg` | Multi-year orthophoto crops (years depend on bbox / sources) |
| `lod1.npz`, `lod2.npz` | Building LoD meshes (when geodata tiles are available) |
| `lidar.npz`, `mesh.npz` | Optional LiDAR + render-mesh archives |

A minimal schema example lives at `dataset/examples/bundestag1_snippet.json` (12 frames).

### Step 1 — Export from Google Earth Studio

In GES, author a camera path and export the **trajectory JSON** (not just video). The file must contain:

- Top-level: `width`, `height`, `frameRate`, `cameraFrames`
- Per frame: `coordinate` (lat/lon/alt), `position` (ECEF), `rotation` (x/y/z degrees), `fovVertical`

Use the **full-length export** (typically 30–100+ seconds). Short ~10 s resampled copies are missing temporal detail and report unrealistic speeds when fed directly to the generator.

The script reads poses from the **`coordinate`** field (WGS84). GES `position` (ECEF) is ignored when it disagrees.

### Step 2 — Generate the scene

#### Path A — Fully automatic (Berlin only, recommended for new users)

Downloads mesh tiles, renders footage, then downloads and crops DOP/DSM/LoD. Tiles are cached under `--map-dir` for reuse.

```bash
PYTHONPATH=. python dataset/create_movingdrone.py \
  --input /path/to/my_flight.json \
  --output data/MovingDrone/scenes/my_flight \
  --auto-geodata \
  --map-dir data/MovingDrone/map \
  --render
```

- **Requires network** on first run (can download several GB of mesh + geodata tiles).
- **Berlin only** — trajectories outside Berlin bounds will be rejected.
- First run may take **30–60+ minutes** depending on bbox size and network speed; later runs reuse the tile cache.
- `--render` enables mesh rendering **and** realism augmentations (trajectory noise, motion blur, wind gusts, random sun). Pass **`--no-realism`** for a clean baseline.
- Velocity bounding runs automatically (default max 100 km/h); use `--no-velocity-bound` to disable.

**Smoke test with the bundled snippet:**

```bash
PYTHONPATH=. python dataset/create_movingdrone.py \
  --input dataset/examples/bundestag1_snippet.json \
  --output test/outputs/create_movingdrone/my_snippet \
  --auto-geodata \
  --map-dir test/outputs/create_movingdrone/map_cache \
  --render \
  --no-realism \
  --keep-rendering
```

#### Path B — Local mesh + geodata tile tree

Use when you already have Berlin tiles on disk (mesh, DOP, DSM, LoD folders):

```bash
PYTHONPATH=. python dataset/create_movingdrone.py \
  --input /path/to/my_flight.json \
  --output data/MovingDrone/scenes/my_flight \
  --mesh-dir /path/to/map/mesh \
  --geodata-dir /path/to/map \
  --render \
  --no-realism
```

The script discovers intersecting tiles, converts LoD GML→OBJ (needs `thirdparty/CityGML2OBJv2`), renders, then crops all modalities to the visible-area bbox.

#### Path C — Reuse priors from an existing scene

Fastest when you only need a new render/poses for the same geographic area:

```bash
PYTHONPATH=. python dataset/create_movingdrone.py \
  --input /path/to/my_flight.json \
  --output data/MovingDrone/scenes/my_flight \
  --mesh-dir /path/to/map/mesh \
  --priors-dir data/MovingDrone/scenes/bundestag1 \
  --render \
  --no-realism \
  --max-frames 30
```

`--priors-dir` copies `dop/`, `dsm.npz`, and optional `lod*.npz` / `lidar.npz` / `mesh.npz` from a reference scene.

#### Path D — Existing GES frames or video (no mesh render)

If GES exported JPEG frames or you already have `video.mp4`:

```bash
PYTHONPATH=. python dataset/create_movingdrone.py \
  --input /path/to/my_flight.json \
  --output data/MovingDrone/scenes/my_flight \
  --frames-dir /path/to/ges_frames \
  --geodata-dir /path/to/map \
  --no-realism
```

Or pass `--video /path/to/video.mp4` instead of `--frames-dir`.

### Processing options (common flags)

| Flag | Default | Purpose |
|------|---------|---------|
| `--render` | off | Mesh-render `video.mp4` + depth (requires `--mesh-dir` or `--auto-geodata`) |
| `--no-realism` | — | Disable trajectory noise, motion blur, wind gusts, random sun |
| `--max-frames N` | all | Cap frames (useful for smoke tests) |
| `--keep-rendering` | off | Keep `rendering/` JPEG folder after encoding `video.mp4` |
| `--no-depth` | depth on | Skip per-frame depth export |
| `--target-fps F` | — | Downsample trajectory temporally (e.g. 5 fps from 30 fps) |
| `--no-velocity-bound` | bound on | Skip speed resampling |

Realism/noise is applied **at scene creation time** by `create_movingdrone.py`, not stored in the input JSON. For offline speed normalization + optional noise on the JSON alone, see `scripts/resample_trajectories.py`.

### Offline vs network

| Step | Offline? |
|------|----------|
| Mesh OBJ render → video/depth/poses | Yes (`--render --mesh-dir`) |
| Frames/video you already have | Yes (`--frames-dir` / `--video`) |
| Map priors you already cropped | Yes (`--priors-dir`) |
| Crop DOP/DSM/LoD from local tiles | Yes (`--geodata-dir`) |
| `--auto-geodata` Berlin download | Needs network (cached in `--map-dir`) |
| Authoring the GES trajectory JSON | Needs GES |

Headless mesh rendering needs EGL/OpenGL (see root [README](../README.md)).

### Run OrthoTrack on the new scene

```bash
PYTHONPATH=. python scripts/run_tracking.py \
  --sequence_dir data/MovingDrone/scenes/my_flight \
  --fine_matcher turbo \
  --output outputs/tracking/my_flight \
  --save_keyframe_vis
```

### Smoke test (developers)

```bash
PYTHONPATH=. python test/test_create_movingdrone.py
```

Outputs land under `test/outputs/create_movingdrone/`. Optional env overrides: `MOVINGDRONE_MESH_DIR`, `MOVINGDRONE_TRAJ_DIR`, `MOVINGDRONE_PRIORS_DIR`, `MOVINGDRONE_GEODATA_DIR`.
