import os
import requests
import zipfile
import shutil
import json as _json
import re
import time
import fcntl
from contextlib import contextmanager
from pathlib import Path
from tqdm import tqdm
from utils.system import download_with_progress
import numpy as np



# Pre-compiled regex patterns for tile name parsing (avoid per-file re.compile overhead)
# Berlin GDI tile names: dop20rgbi_33_392_5820_2_be_2025.jp2 or LoD1_392_5820.xml
# Groups: (prefix) (easting_km) (northing_km) (optional: tile_size_km) (separator)
_RE_BERLIN_TILE = re.compile(r'(?:_|LoD1_|LoD2_|33_)(\d{3})_(\d{4})(?:_(\d)_|\.|_)')
_RE_BB_TILE = re.compile(r'33(\d{3})(-|_)(\d{4})')

# Berlin GDI DOP datasets from 2010 onwards (open data, dl-zero-de/2.0).
# Each year maps to the dataset subfolder name used in the ATOM feed.
# URL pattern: https://gdi.berlin.de/data/{dataset}/atom/{region}.zip
# Verified against https://gdi.berlin.de/data/ (Feb 2026).
DOP_YEARS = {
    2010: "dop_2010",
    2011: "dop_2011",
    2013: "truedop_2013",
    2014: "dop_2014",            # NB: dop_2014, NOT truedop_2014 (404)
    2015: "dop_2015",
    2016: "dop_2016",
    2017: "dop_2017",
    2018: "dop_2018",
    2019: "dop_2019",
    2020: "truedop_2020_sommer",
    2021: "dop_2021",
    2022: "truedop_2022",
    2023: "truedop_2023",
    2024: "truedop_2024",
    2025: "truedop_2025_sommer",
}

# The 9 geographic region archives that each DOP year is split into.
DOP_REGIONS = ["Mitte", "Nord", "Nordost", "Nordwest", "Ost", "Sued", "Suedost", "Suedwest", "West"]

# Years that use umlaut region names (Süd, Südost, Südwest) in their ATOM feed.
# Verified via HTTP HEAD requests against gdi.berlin.de (Feb 2026).
_DOP_UMLAUT_YEARS = {2025}

# Mapping from ASCII canonical names to umlaut variants.
_UMLAUT_MAP = {"Sued": "Süd", "Suedost": "Südost", "Suedwest": "Südwest"}

def get_dop_region_names(year: int) -> list:
    """Return the 9 region archive names for a given DOP year.
    
    Years >= 2025 use German umlauts (Süd, Südost, Südwest) in their
    ATOM feed URLs.  All older years use ASCII (Sued, Suedost, Suedwest)."""
    if year in _DOP_UMLAUT_YEARS:
        return [_UMLAUT_MAP.get(r, r) for r in DOP_REGIONS]
    return list(DOP_REGIONS)

# Approximate bounding boxes for each DOP region in UTM zone 33N (EPSG:25833).
# Format: (min_easting, min_northing, max_easting, max_northing)
# These are rough extents derived from Berlin district geography.
# A generous buffer is included so border areas are never missed.
DOP_REGION_BBOX = {
    "Mitte":    (381000, 5815000, 395000, 5828000),
    "Nord":     (382000, 5828000, 399000, 5840000),
    "Nordost":  (395000, 5825000, 415000, 5842000),
    "Nordwest": (369000, 5828000, 386000, 5840000),
    "Ost":      (396000, 5810000, 415000, 5828000),
    "Sued":     (381000, 5800000, 399000, 5816000),
    "Suedost":  (396000, 5800000, 415000, 5812000),
    "Suedwest": (365000, 5800000, 384000, 5818000),
    "West":     (365000, 5815000, 386000, 5832000),
}

# GSD is always 0.2 m for all Berlin DOP years (same as DSM).
DOP_GSD = 0.2

# WMS base URL and RGB layer names per DOP year.
# Used as fallback when local tiles can't be read (e.g. ECW format).
# WMS endpoint: https://gdi.berlin.de/services/wms/{dataset_name}
# CRS: EPSG:25833 (ETRS89/UTM33N, ≈ EPSG:32633 within <1 m for Berlin).
# Layer names verified against WMS GetCapabilities (Feb 2026).
# CIR layers (Color Infrared): used when the RGB rendering is broken server-side.
# CIR maps NIR→Red, Red→Green, Green→Blue → vegetation appears bright red.
DOP_CIR_LAYERS = {
    2010: "dop_2010_cir",  # RGB layer broken: "4 bands → 3 bands" IllegalArgumentException
}

DOP_WMS_LAYERS = {
    2010: "dop_2010_rgb",  # Will fail → auto-fallback to CIR + conversion
    2011: "dop_2011",
    2013: "truedop_2013",
    2014: "dop_2014",
    2015: "dop_2015_rgb",
    2016: "dop_2016",
    2017: "dop_2017",
    2018: "dop_2018",
    2019: "dop_2019",
    2020: "truedop_2020_sommer_rgb",
    2021: "dop_2021_rgb",
    2022: "truedop_2022",
    2023: "truedop_2023",
    2024: "truedop_2024",
    2025: "truedop_2025_sommer_rgb",
}




# Maximum pixels per WMS GetMap request dimension.  Berlin GDI WMS silently
# fails or times out for very large requests (e.g. 11000×9700).  Splitting
# into chunks ≤ this size avoids the limit.
_MAX_WMS_PX = 4096


def _fetch_wms_single(wms_url, layer, bbox_chunk, w_px, h_px, fmt="image/jpeg", timeout=90):
    """Fetch a single WMS GetMap tile.  Returns PIL Image or None."""
    from PIL import Image
    import io

    min_x, min_y, max_x, max_y = bbox_chunk
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "CRS": "EPSG:25833",
        "BBOX": f"{min_x},{min_y},{max_x},{max_y}",
        "WIDTH": str(w_px),
        "HEIGHT": str(h_px),
        "FORMAT": fmt,
    }
    resp = requests.get(wms_url, params=params, timeout=timeout)
    ct = resp.headers.get("content-type", "")
    if not ct.startswith("image") or len(resp.content) < 1000:
        return None
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def fetch_dop_wms_crop(year, bbox, width_px, height_px, fmt="image/jpeg",
                       cache_dir=None):
    """
    Fetch a DOP crop via Berlin GDI WMS for a given year and UTM bounding box.
    
    Large requests (> _MAX_WMS_PX in either dimension) are automatically split
    into a grid of smaller chunks that are fetched individually and stitched
    back together.  This works around undocumented server-side size limits on
    the Berlin GDI WMS.
    
    For years where the RGB WMS layer is broken (e.g. 2010: server can't render
    4-band RGBI to 3-band), automatically falls back to CIR layer and converts
    the false-colour imagery to approximate true-colour RGB.
    
    When *cache_dir* is provided, WMS responses are cached on a fixed 500 m UTM
    grid so that overlapping requests across different sequences only hit the
    remote server once.  Cache files live at
    ``{cache_dir}/wms_cache/dop_{year}/{e}_{n}.jpg`` and are protected by
    ``fcntl.flock()`` for cross-process safety (parallel SLURM jobs).
    
    Args:
        year: DOP year (int), must be in DOP_YEARS and DOP_WMS_LAYERS
        bbox: (min_x, min_y, max_x, max_y) in EPSG:32633/25833 UTM coordinates
        width_px: Output image width in pixels
        height_px: Output image height in pixels
        fmt: WMS GetMap FORMAT (image/jpeg or image/png)
        cache_dir: Optional path to a persistent cache directory (e.g. map/).
            When set, grid-based caching is used.
    
    Returns:
        numpy array (H, W, 3) uint8 RGB, or None on failure"""
    if cache_dir is not None:
        return _fetch_dop_wms_cached(year, bbox, width_px, height_px,
                                     cache_dir, fmt=fmt)
    import math

    dataset_name = DOP_YEARS.get(year)
    layer_name = DOP_WMS_LAYERS.get(year)
    if not dataset_name or not layer_name:
        return None
    
    min_x, min_y, max_x, max_y = bbox
    wms_url = f"https://gdi.berlin.de/services/wms/{dataset_name}"
    
    # --- Determine tiling grid --------------------------------------------------
    n_cols = max(1, math.ceil(width_px / _MAX_WMS_PX))
    n_rows = max(1, math.ceil(height_px / _MAX_WMS_PX))
    need_tiling = n_cols > 1 or n_rows > 1
    
    if need_tiling:
        print(f"      WMS {year}: request {width_px}x{height_px} exceeds limit, "
              f"tiling into {n_cols}x{n_rows} chunks")
    
    # Pixel sizes per chunk (last chunk may be smaller)
    col_widths = []
    for c in range(n_cols):
        start = c * _MAX_WMS_PX
        end = min(start + _MAX_WMS_PX, width_px)
        col_widths.append(end - start)
    
    row_heights = []
    for r in range(n_rows):
        start = r * _MAX_WMS_PX
        end = min(start + _MAX_WMS_PX, height_px)
        row_heights.append(end - start)
    
    # Corresponding UTM extents per chunk
    utm_width = max_x - min_x
    utm_height = max_y - min_y
    
    def _try_fetch_layer(layer):
        """Attempt to fetch all chunks for a given WMS layer. Returns stitched numpy array or None."""
        canvas = np.zeros((height_px, width_px, 3), dtype=np.uint8)
        ok_chunks = 0
        total_chunks = n_cols * n_rows
        
        y_px_offset = 0
        for r in range(n_rows):
            x_px_offset = 0
            chunk_h = row_heights[r]
            # UTM Y: rows go bottom-to-top in UTM but top-to-bottom in image.
            # Row 0 = top of image = max_y side.
            utm_row_max = max_y - (y_px_offset / height_px) * utm_height
            utm_row_min = max_y - ((y_px_offset + chunk_h) / height_px) * utm_height
            
            for c in range(n_cols):
                chunk_w = col_widths[c]
                utm_col_min = min_x + (x_px_offset / width_px) * utm_width
                utm_col_max = min_x + ((x_px_offset + chunk_w) / width_px) * utm_width
                
                chunk_bbox = (utm_col_min, utm_row_min, utm_col_max, utm_row_max)
                
                try:
                    tile_img = _fetch_wms_single(
                        wms_url, layer, chunk_bbox, chunk_w, chunk_h, fmt=fmt)
                except Exception:
                    tile_img = None
                
                if tile_img is not None:
                    arr = np.array(tile_img)
                    # Handle slight size mismatches from the server
                    ah, aw = arr.shape[:2]
                    paste_h = min(ah, chunk_h)
                    paste_w = min(aw, chunk_w)
                    canvas[y_px_offset:y_px_offset + paste_h,
                           x_px_offset:x_px_offset + paste_w] = arr[:paste_h, :paste_w]
                    ok_chunks += 1
                
                x_px_offset += chunk_w
            y_px_offset += chunk_h
        
        if ok_chunks == 0:
            return None
        if ok_chunks < total_chunks:
            print(f"      WMS {year}: {ok_chunks}/{total_chunks} chunks succeeded (partial)")
        return canvas
    
    try:
        result = _try_fetch_layer(layer_name)
        if result is not None:
            return result
        
        # CIR fallback disabled — DOP must be RGB only.
        # CIR→RGB conversions produce false-color imagery unusable for matching.
        
        print(f"      WMS {year}: all layers failed")
        return None
    except Exception as e:
        print(f"      WMS {year}: request failed - {e}")
        return None


# ---------------------------------------------------------------------------
# Grid-based WMS tile cache
# ---------------------------------------------------------------------------
# Tiles are stored on the standard Berlin GDI 2 km UTM grid so that multiple
# sequences with overlapping geographic regions share the same cached WMS
# responses.  The grid is aligned to the official tile boundaries used by
# Berlin open-data portals (e.g. dop20rgbi_33_390_5820_2_be_2023.jp2) where
# easting/northing step by 2 km and coordinates are even km values.
#
# Tile naming: ``33_{e_km}_{n_km}.jpg`` — matches the GDI pattern and makes
# it easy to cross-reference with downloaded ZIP tiles.
#
# 2 km at 0.2 m GSD = 10 000 px per tile.  This exceeds the Berlin WMS
# single-request limit (_MAX_WMS_PX = 4096), so each tile is fetched as a
# 2×2 grid of 5000 px chunks internally.
_WMS_CACHE_GRID_M = 2000
_WMS_CACHE_GRID_KM = _WMS_CACHE_GRID_M // 1000          # 2
_WMS_CACHE_TILE_PX = int(_WMS_CACHE_GRID_M / DOP_GSD)   # 10000


def _wms_grid_origin(coord_m, grid_m=_WMS_CACHE_GRID_M):
    """Snap a UTM coordinate (metres) down to the nearest grid origin.

    The grid is anchored so that coordinates are always *even* km values
    (384, 386, …) matching the Berlin GDI 2 km tile convention.

    >>> _wms_grid_origin(391_200)   # → 390_000
    390000"""
    return int(np.floor(coord_m / grid_m)) * grid_m


def _fetch_dop_wms_cached(year, bbox, width_px, height_px, cache_dir,
                           fmt="image/jpeg"):
    """Grid-cached WMS DOP fetching.

    Divides the request into fixed **2 km** UTM grid cells aligned to the
    official Berlin GDI tile grid, fetches (or reads from disk cache) each
    cell, composites them, and crops to the exact requested *bbox* /
    resolution.

    Cache layout::

        {cache_dir}/wms_cache/dop_{year}/33_{e_km}_{n_km}.jpg   — cached tile
        {cache_dir}/wms_cache/dop_{year}/33_{e_km}_{n_km}.fail  — known-failed

    For example ``33_390_5820.jpg`` corresponds to the 2 km × 2 km area from
    UTM (390 000, 5 820 000) to (392 000, 5 822 000).

    Cross-process safety: ``fcntl.flock()`` on
    ``{cache_dir}/.locks/wms_dop_{year}_33_{e}_{n}.lock``.

    Returns:
        numpy array (height_px, width_px, 3) uint8 RGB, or None on failure."""
    from PIL import Image

    grid_m  = _WMS_CACHE_GRID_M
    grid_km = _WMS_CACHE_GRID_KM
    tile_px = _WMS_CACHE_TILE_PX
    gsd     = DOP_GSD

    dataset_name = DOP_YEARS.get(year)
    layer_name   = DOP_WMS_LAYERS.get(year)
    if not dataset_name or not layer_name:
        return None

    wms_url = f"https://gdi.berlin.de/services/wms/{dataset_name}"
    min_x, min_y, max_x, max_y = bbox

    # --- Directories ---
    cache_path = Path(cache_dir) / "wms_cache" / f"dop_{year}"
    cache_path.mkdir(parents=True, exist_ok=True)
    lock_dir = Path(cache_dir) / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)

    # --- Grid cells overlapping the bbox (in km, steppped by grid_km) ---
    e_start_km = (int(np.floor(min_x / 1000)) // grid_km) * grid_km
    e_end_km   = (int(np.floor(max_x / 1000)) // grid_km) * grid_km
    n_start_km = (int(np.floor(min_y / 1000)) // grid_km) * grid_km
    n_end_km   = (int(np.floor(max_y / 1000)) // grid_km) * grid_km

    n_cols = (e_end_km - e_start_km) // grid_km + 1
    n_rows = (n_end_km - n_start_km) // grid_km + 1

    # Composite image extent (pixels)
    comp_w_px = n_cols * tile_px
    comp_h_px = n_rows * tile_px
    canvas = np.zeros((comp_h_px, comp_w_px, 3), dtype=np.uint8)

    total_cells  = n_cols * n_rows
    ok_cells     = 0
    cached_cells = 0

    for col, e_km in enumerate(range(e_start_km, e_end_km + grid_km, grid_km)):
        for row_idx, n_km in enumerate(range(n_start_km, n_end_km + grid_km, grid_km)):
            tile_name = f"33_{e_km}_{n_km}"
            tile_file = cache_path / f"{tile_name}.jpg"
            fail_file = cache_path / f"{tile_name}.fail"

            arr = None

            # --- Fast path: already cached on disk ---
            if tile_file.exists():
                try:
                    arr = np.array(Image.open(tile_file).convert("RGB"))
                    cached_cells += 1
                    ok_cells += 1
                except Exception:
                    pass  # corrupt file, will re-fetch below
            elif fail_file.exists():
                # Previously failed; don't retry.
                continue

            # --- Slow path: fetch with cross-process lock ---
            if arr is None and not fail_file.exists():
                lock_name = f"wms_dop_{year}_{tile_name}.lock"
                lock_path = lock_dir / lock_name
                fd = None
                try:
                    fd = open(lock_path, 'w')
                    fcntl.flock(fd, fcntl.LOCK_EX)

                    # Double-check: another process may have fetched while we
                    # waited for the lock.
                    if tile_file.exists():
                        try:
                            arr = np.array(
                                Image.open(tile_file).convert("RGB"))
                            cached_cells += 1
                            ok_cells += 1
                        except Exception:
                            pass

                    if arr is None:
                        arr = _fetch_wms_full_tile(
                            wms_url, layer_name,
                            e_km * 1000, n_km * 1000,
                            grid_m, tile_px, gsd, fmt=fmt)
                        if arr is not None:
                            Image.fromarray(arr).save(
                                str(tile_file), "JPEG", quality=95)
                            ok_cells += 1
                        else:
                            try:
                                fail_file.touch()
                            except Exception:
                                pass
                finally:
                    if fd is not None:
                        try:
                            fcntl.flock(fd, fcntl.LOCK_UN)
                            fd.close()
                        except Exception:
                            pass

            # --- Paste into composite canvas ---
            if arr is not None:
                x_px = col * tile_px
                # Y is inverted: UTM northing increases upward but image
                # row 0 = top = max northing.
                y_px = (n_rows - 1 - row_idx) * tile_px
                ah, aw = arr.shape[:2]
                ph = min(ah, tile_px)
                pw = min(aw, tile_px)
                canvas[y_px:y_px + ph, x_px:x_px + pw] = arr[:ph, :pw]

    if ok_cells == 0:
        return None

    if cached_cells > 0 or total_cells > 1:
        fetched = ok_cells - cached_cells
        print(f"      WMS cache {year}: {cached_cells}/{total_cells} cached, "
              f"{fetched} fetched, {total_cells - ok_cells} failed")

    # --- Crop composite to the exact requested bbox ---
    comp_min_x = e_start_km * 1000
    comp_max_y = (n_end_km + grid_km) * 1000  # top of composite in UTM

    crop_x1 = int(round((min_x - comp_min_x) / gsd))
    crop_y1 = int(round((comp_max_y - max_y) / gsd))  # Y inverted
    crop_x2 = crop_x1 + width_px
    crop_y2 = crop_y1 + height_px

    # Clamp
    crop_x1 = max(0, min(crop_x1, comp_w_px))
    crop_y1 = max(0, min(crop_y1, comp_h_px))
    crop_x2 = max(0, min(crop_x2, comp_w_px))
    crop_y2 = max(0, min(crop_y2, comp_h_px))

    result = canvas[crop_y1:crop_y2, crop_x1:crop_x2]

    # Pad if the crop extends beyond the composite (shouldn't normally happen)
    if result.shape[0] != height_px or result.shape[1] != width_px:
        padded = np.zeros((height_px, width_px, 3), dtype=np.uint8)
        ph = min(result.shape[0], height_px)
        pw = min(result.shape[1], width_px)
        padded[:ph, :pw] = result[:ph, :pw]
        result = padded

    return result


def _fetch_wms_full_tile(wms_url, layer_name, origin_x, origin_y,
                         grid_m, tile_px, gsd, fmt="image/jpeg"):
    """Fetch a full grid tile, splitting into sub-requests if > _MAX_WMS_PX.

    The Berlin GDI WMS silently fails for requests > 4096 px in either
    dimension.  A 2 km tile at 0.2 m GSD = 10 000 px, so we split into
    a grid of sub-chunks automatically.

    Args:
        wms_url: WMS base URL.
        layer_name: WMS layer name.
        origin_x: Tile lower-left easting in metres.
        origin_y: Tile lower-left northing in metres.
        grid_m: Tile extent in metres (2000).
        tile_px: Tile extent in pixels (10000).
        gsd: Ground sample distance (0.2).
        fmt: WMS FORMAT parameter.

    Returns:
        numpy (tile_px, tile_px, 3) uint8 RGB, or None on total failure."""
    import math

    n_chunks = max(1, math.ceil(tile_px / _MAX_WMS_PX))
    chunk_px = math.ceil(tile_px / n_chunks)
    chunk_m  = grid_m / n_chunks

    canvas = np.zeros((tile_px, tile_px, 3), dtype=np.uint8)
    ok = 0

    for row in range(n_chunks):
        for col in range(n_chunks):
            cx = origin_x + col * chunk_m
            cy = origin_y + row * chunk_m
            chunk_bbox = (cx, cy, cx + chunk_m, cy + chunk_m)

            # Pixel extent for this chunk (last chunk may be smaller)
            px_x0 = col * chunk_px
            px_y0 = row * chunk_px
            w = min(chunk_px, tile_px - px_x0)
            h = min(chunk_px, tile_px - px_y0)

            try:
                img = _fetch_wms_single(wms_url, layer_name,
                                        chunk_bbox, w, h, fmt=fmt)
            except Exception:
                img = None

            if img is not None:
                arr = np.array(img)
                ah, aw = arr.shape[:2]
                ph = min(ah, h)
                pw = min(aw, w)
                # Image row 0 = top = max northing; UTM row 0 = bottom.
                # Row index in canvas: bottom row of UTM (row=0) → last
                # pixel rows in the image.
                img_y0 = tile_px - px_y0 - h
                canvas[img_y0:img_y0 + ph, px_x0:px_x0 + pw] = arr[:ph, :pw]
                ok += 1

    return canvas if ok > 0 else None


class GeodataManager:
    """
    Manages automated discovery and downloading of geodata from Geobasis-BB and VirtualCityMap."""
    
    BASE_URLS = {
        "dop": "https://data.geobasis-bb.de/geobasis/daten/dop/rgbi_tif/",  # Fixed: was rgb_jpg
        "dsm": "https://data.geobasis-bb.de/geobasis/daten/bdom/tif/",
        "lod1_bb": "https://data.geobasis-bb.de/geobasis/daten/3d_gebaeude/lod1_gml/",
        "lod2_bb": "https://data.geobasis-bb.de/geobasis/daten/3d_gebaeude/lod2_gml/",
        "lod1_berlin": "https://gdi.berlin.de/data/a_lod1/atom/LoD1.zip",
        "lod2_berlin": "https://gdi.berlin.de/data/a_lod2/atom/",
        "dop_berlin": "https://gdi.berlin.de/data/a_dop20rgbi/atom/",  # Berlin DOP 20cm RGB+IR (fallback)
        "dsm_berlin": "https://gdi.berlin.de/data/a_dgm/atom/",  # Berlin DGM (fallback)
        "mesh": "https://download-berlin3d.virtualcitymap.de/datasource-data/berlin-mesh-2025/",
        "als_berlin_base": "https://gdi.berlin.de/data/a_als/atom/",
        "als_bb": "https://data.geobasis-bb.de/geobasis/daten/als/laz/"
    }
    
    # Berlin ALS Regions/Archives
    ALS_REGIONS = [
        "Mitte.zip", "Nord.zip", "Nordost.zip", "Nordwest.zip", 
        "Ost.zip", "Sued.zip", "Suedost.zip", "Suedwest.zip", "West.zip"
    ]
    
    # Berlin DOP/DSM Regions (2km x 2km tiles organized by region) - Fallback only
    DOP_DSM_REGIONS = [
        "Mitte.zip", "Nord.zip", "Nordost.zip", "Nordwest.zip",
        "Ost.zip", "Sued.zip", "Suedost.zip", "Suedwest.zip", "West.zip"
    ]

    def __init__(self, map_dir="data/MovingDrone/map"):
        self.map_dir = Path(map_dir)
        self.map_dir.mkdir(parents=True, exist_ok=True)
        # Lock directory for cross-process tile download coordination
        self._lock_dir = self.map_dir / ".locks"
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        # Cache of tile download failures (404s etc.) to avoid retrying
        self._download_fail_cache = {}
        self._download_fail_cache_path = self.map_dir / ".download_fail_cache.json"
        self._load_download_fail_cache()

    @contextmanager
    def _tile_lock(self, data_type, tile_id):
        """
        Cross-process file lock for a specific tile.
        Uses fcntl.flock() so multiple SLURM jobs don't download the same tile.
        If another process holds the lock, this blocks until it's released."""
        lock_name = f"{data_type}_{tile_id}.lock".replace('/', '_').replace('\\', '_')
        lock_path = self._lock_dir / lock_name
        fd = None
        try:
            fd = open(lock_path, 'w')
            fcntl.flock(fd, fcntl.LOCK_EX)  # Blocking exclusive lock
            yield
        finally:
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    fd.close()
                except Exception:
                    pass

    def _load_download_fail_cache(self):
        """Load persistent cache of known-failed tile downloads (404s)."""
        if self._download_fail_cache_path.exists():
            try:
                with open(self._download_fail_cache_path) as f:
                    cache_data = _json.load(f)
                # Invalidate cache older than 30 days
                if time.time() - cache_data.get("_timestamp", 0) < 30 * 86400:
                    self._download_fail_cache = cache_data.get("failures", {})
                else:
                    self._download_fail_cache = {}
            except Exception:
                self._download_fail_cache = {}

    def _save_download_fail_cache(self):
        """Persist the download failure cache to disk (with file locking for cross-process safety)."""
        lock_path = self._lock_dir / "fail_cache.lock"
        fd = None
        try:
            fd = open(lock_path, 'w')
            fcntl.flock(fd, fcntl.LOCK_EX)
            # Merge with any updates from other processes
            existing = {}
            if self._download_fail_cache_path.exists():
                try:
                    with open(self._download_fail_cache_path) as f:
                        existing = _json.load(f).get("failures", {})
                except Exception:
                    pass
            existing.update(self._download_fail_cache)
            with open(self._download_fail_cache_path, 'w') as f:
                _json.dump({"_timestamp": time.time(), "failures": existing}, f)
        except Exception:
            pass
        finally:
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    fd.close()
                except Exception:
                    pass

    def get_geobasis_tile_id(self, easting, northing):
        """
        Converts UTM-33N coordinates to Geobasis-BB 1km tile ID."""
        e_km = int(easting // 1000)
        n_km = int(northing // 1000)
        return f"33{e_km}-{n_km}"

    def get_berlin_lod2_tile_id(self, easting, northing):
        """
        Converts UTM-33N coordinates to Berlin LoD2 10km tile ID."""
        e_km = int(easting // 1000)
        n_km = int(northing // 1000)
        return f"LoD2_{e_km}_{n_km}"

    def get_als_tile_id(self, easting, northing):
        """
        Converts UTM-33N coordinates to Berlin ALS 1km tile filename.
        Format: 3dm_33_391_5820_1_be.las"""
        e_km = int(easting // 1000)
        n_km = int(northing // 1000)
        return f"3dm_33_{e_km}_{n_km}_1_be"


    @staticmethod
    def _berlin_als_to_bb_als(berlin_tile_id):
        """
        Converts Berlin ALS tile ID to Brandenburg ALS tile ID.
        Berlin: 3dm_33_399_5802_1_be -> BB: als_33399-5802"""
        import re
        m = re.match(r'3dm_33_(\d+)_(\d+)_1_be', berlin_tile_id)
        if m:
            return f"als_33{m.group(1)}-{m.group(2)}"
        return None







    def discover_tiles(self, utm_positions, data_types=["dop", "dsm", "mesh", "lod2"],
                       bbox=None):
        """
        Identifies required tile IDs for a given trajectory.
        
        Args:
            utm_positions: (N, 2) array of camera positions in UTM
            data_types: list of data types to discover
            bbox: Optional (min_e, min_n, max_e, max_n) override.
                  If provided, uses this bbox instead of computing from positions.
                  Useful when the actual visible ground region (from depth maps)
                  is much larger than the camera position envelope."""
        required = {t: set() for t in data_types}
        
        # Trajectory BBox — use explicit bbox if provided, else compute from positions
        if bbox is not None:
            min_e, min_n, max_e, max_n = bbox
        else:
            min_e, max_e = np.min(utm_positions[:, 0]), np.max(utm_positions[:, 0])
            min_n, max_n = np.min(utm_positions[:, 1]), np.max(utm_positions[:, 1])
        
        # Standard padding for DOP/DSM/LoD
        padding = 500
        
        # 1. Brandenburg-style Geodata (DOP, DSM, LoD1, LoD2)
        # These IDs (e.g. 33391-5820) cover Berlin as well via Brandenburg sources.
        # Grid is 1km x 1km.
        bb_types = [t for t in ["dsm", "lod1", "lod2"] if t in data_types]
        if bb_types:
            # Step in 500m to ensure we catch all 1km tiles in the range
            e_range = np.arange(min_e - padding, max_e + padding, 500) 
            n_range = np.arange(min_n - padding, max_n + padding, 500)
            for e in e_range:
                for n in n_range:
                    tid = self.get_geobasis_tile_id(e, n)
                    for t in bb_types:
                        required[t].add(tid)
                        
        if "dop" in data_types:
            # For DOP, we explicitly want ALL available years from the Berlin GDI,
            # plus the Brandenburg fallback. We use special keys for the years.
            for year in DOP_YEARS.keys():
                req_key = f"dop_{year}"
                if req_key not in required:
                    required[req_key] = set()
                # We add a dummy tile ID for the year (the bounding box handles filtering)
                required[req_key].add("berlin_gdi_region")
            
            # Plus the BB Fallback
            required["dop_bb_fallback"] = set()
            e_range = np.arange(min_e - padding, max_e + padding, 500) 
            n_range = np.arange(min_n - padding, max_n + padding, 500)
            for e in e_range:
                for n in n_range:
                    tid = self.get_geobasis_tile_id(e, n)
                    required["dop_bb_fallback"].add(tid)


        if "lod1" in data_types:
            # Berlin city center LoD1 is most reliably found in the full archive.
            required["lod1"].add("berlin_full")
            
        if "als" in data_types:
            # Check Berlin ALS (1km grid within Zips)
            e_range = np.arange(min_e - padding, max_e + padding, 200) 
            n_range = np.arange(min_n - padding, max_n + padding, 200)
            for e in e_range:
                for n in n_range:
                    tid = self.get_als_tile_id(e, n)
                    required["als"].add(tid)
        
        
        # ALKIS support removed - use LoD1/LoD2 for building footprints instead
        
        if "lod2" in data_types:
            # Check Berlin specific LoD2 (10km grid)
            e_range = np.arange(min_e - padding, max_e + padding, 1000) 
            n_range = np.arange(min_n - padding, max_n + padding, 1000)
            for e in e_range:
                for n in n_range:
                    tid = self.get_berlin_lod2_tile_id(e, n)
                    required["lod2"].add(tid)
        
        # 1. BB-based types (Brandenburg/Berlin outskirts - Fallback or fixed types)
        # Only add BB-style LoD if not already satisfied by Berlin sources
        # (Though some might prefer both, usually Berlin sources are better for Berlin)
        bb_types = [t for t in ["dsm"] if t in data_types]
        # For LoD, if we are in Berlin zone (3xx, 5xxx), maybe don't even add BB-style?
        # But for robustness, we can keep them as fallback.
        
        bb_lod_types = [t for t in ["lod1", "lod2"] if t in data_types]
        
        e_range = np.arange(min_e - padding, max_e + padding, 200) 
        n_range = np.arange(min_n - padding, max_n + padding, 200)
        for e in e_range:
            for n in n_range:
                tid = self.get_geobasis_tile_id(e, n)
                for t in bb_types:
                    required[t].add(tid)
                # Fallback BB tiles (1km grid)
                for t in bb_lod_types:
                    # We add them to a separate key to handle fallback in download_all
                    fallback_key = f"{t}_bb_fallback"
                    if fallback_key not in required: required[fallback_key] = set()
                    required[fallback_key].add(tid)



        if "mesh" in data_types:
            # Trajectory can look far, so use more padding for meshes
            mesh_padding = 1000
            print(f"   Searching mesh candidates (padding={mesh_padding}m)...")
            
            e_range_mesh = np.arange(min_e - mesh_padding, max_e + mesh_padding, 100)
            n_range_mesh = np.arange(min_n - mesh_padding, max_n + mesh_padding, 100)
            
            candidates = set()
            suffixes = ["_-002", "_0002", "_-001", "_0001", "", "_002", "_-003", "_0003"]
            
            for e in e_range_mesh:
                for n in n_range_mesh:
                    base = f"{int(e // 100)}_{int(n // 100)}"
                    for s in suffixes:
                        candidates.add(base + s)

            # Load persistent mesh tile cache (avoids re-probing known tiles)
            mesh_cache_path = self.map_dir / ".mesh_tile_cache.json"
            mesh_cache = {}
            if mesh_cache_path.exists():
                try:
                    with open(mesh_cache_path) as f:
                        cache_data = _json.load(f)
                    # Cache entries: {tile_id: True/False}, with timestamp
                    # Invalidate cache older than 30 days
                    cache_ts = cache_data.get("_timestamp", 0)
                    if time.time() - cache_ts < 30 * 86400:
                        mesh_cache = cache_data.get("tiles", {})
                except Exception:
                    pass
            
            # Split candidates into cached hits, cached misses, and unknown
            already_valid = set()
            to_probe = set()
            for c in candidates:
                if c in mesh_cache:
                    if mesh_cache[c]:
                        already_valid.add(c)
                    # else: known miss, skip
                else:
                    to_probe.add(c)
            
            if already_valid:
                print(f"   {len(already_valid)} tiles from cache, {len(to_probe)} to probe...")
            
            # Probe unknown candidates with higher parallelism
            if to_probe:
                session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    max_retries=2,
                    pool_connections=50,
                    pool_maxsize=50
                )
                session.mount("https://", adapter)
                
                def check_mesh(tid):
                    url = self.BASE_URLS["mesh"] + tid + ".zip"
                    try:
                        r = session.head(url, timeout=5)
                        if r.status_code == 200:
                            return tid
                    except:
                        pass
                    return None

                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=50) as executor:
                    results = list(tqdm(executor.map(check_mesh, list(to_probe)), 
                                        total=len(to_probe), desc="Mesh discovery", leave=False))
                
                # Update cache with new results
                for tid in to_probe:
                    mesh_cache[tid] = False
                for r in results:
                    if r is not None:
                        mesh_cache[r] = True
                        already_valid.add(r)
                
                # Save cache
                try:
                    with open(mesh_cache_path, 'w') as f:
                        _json.dump({"_timestamp": time.time(), "tiles": mesh_cache}, f)
                except Exception:
                    pass
            
            required["mesh"] = already_valid
            print(f"   Found {len(already_valid)} mesh tiles.")

        return {k: sorted(list(v)) for k, v in required.items()}



    def is_in_berlin(self, utm_positions_epsg4326):
        """
        Checks if the trajectory is within Berlin bounds.
        Rough bounds for Berlin: Lat [52.3, 52.7], Lon [13.0, 13.8]"""
        lons = utm_positions_epsg4326[:, 0]
        lats = utm_positions_epsg4326[:, 1]
        
        min_lon, max_lon = lons.min(), lons.max()
        min_lat, max_lat = lats.min(), lats.max()
        
        # Berlin bounds (approximate but safe)
        BERLIN_LAT = (52.3, 52.7)
        BERLIN_LON = (13.0, 13.8)
        
        if not (BERLIN_LON[0] <= min_lon <= BERLIN_LON[1] and 
                BERLIN_LON[0] <= max_lon <= BERLIN_LON[1] and
                BERLIN_LAT[0] <= min_lat <= BERLIN_LAT[1] and 
                BERLIN_LAT[0] <= max_lat <= BERLIN_LAT[1]):
            return False
        return True

    def get_url_for_tile(self, data_type, tile_id):
        """
        Returns the data download URL and expected filename for a given tile."""
        real_dtype = data_type.replace("_bb_fallback", "")
        
        if data_type == "lod1" and tile_id == "berlin_full":
            return self.BASE_URLS["lod1_berlin"], "LoD1_Berlin.zip"
        elif data_type == "lod2" and tile_id.startswith("LoD2_"):
            return self.BASE_URLS["lod2_berlin"] + f"{tile_id}.zip", f"{tile_id}.zip"
        elif data_type == "lod1" or data_type == "lod2":
            prefix = "lod1" if data_type == "lod1" else "lod2"
            filename = f"{prefix}_{tile_id}.zip"
            return self.BASE_URLS.get(f"{data_type}_bb", "") + filename, filename
        elif real_dtype.startswith("dop_") and real_dtype != "dop":
            # Berlin GDI DOP year (e.g. dop_2025, dop_2021)
            # tile_id is the region name (e.g. "Mitte") or dummy "berlin_gdi_region"
            try:
                year = int(real_dtype.split("_")[1])
                dataset = DOP_YEARS.get(year, f"dop_{year}")
                filename = f"{tile_id}.zip"
                url = f"https://gdi.berlin.de/data/{dataset}/atom/{filename}"
                return url, filename
            except (ValueError, IndexError):
                pass
            # Fallback to generic
            filename = f"{tile_id}.zip"
            return filename, filename
        else:
            prefixes = {
                "dop": "dop",
                "dsm": "bdom",
                "lod1": "lod1",
                "lod2": "lod2",
                "mesh": ""
            }
            prefix = prefixes.get(real_dtype, "")
            filename = f"{prefix}_{tile_id}.zip" if prefix else f"{tile_id}.zip"
            source_key = f"{real_dtype}_bb" if f"{real_dtype}_bb" in self.BASE_URLS else real_dtype
            return self.BASE_URLS.get(source_key, "") + filename, filename

    def download_tile(self, data_type, tile_id, bounds=None):
        """
        Downloads and extracts a tile if not already present.
        Skips tiles known to have failed (cached 404s).
        Uses file-based locking for cross-process safety."""
        # Check download failure cache
        cache_key = f"{data_type}:{tile_id}"
        if cache_key in self._download_fail_cache:
            return None
        
        with self._tile_lock(data_type, tile_id):
            # Re-check failure cache inside lock (another process may have updated it)
            self._load_download_fail_cache()
            if cache_key in self._download_fail_cache:
                return None

            # Handle multi-year Berlin DOP
            if data_type.startswith("dop_") and data_type != "dop_bb_fallback":
                try:
                    year = int(data_type.split("_")[1])
                    # download_dop_year handles its own region locking
                    result = self.download_dop_year(year, bbox=bounds)
                    if result is None:
                        self._download_fail_cache[cache_key] = "failed"
                    return result
                except ValueError:
                    pass

            if data_type == "als":
                result = self._download_als_tile_unlocked(tile_id)
                if result is None:
                    self._download_fail_cache[cache_key] = "failed"
                return result
            
            if data_type == "dop" or data_type == "dop_bb_fallback":
                result = self._download_dop_tile_unlocked(tile_id)
                if result is None:
                    self._download_fail_cache[cache_key] = "failed"
                return result
                
            if data_type == "dsm":
                result = self._download_dsm_tile_unlocked(tile_id)
                if result is None:
                    self._download_fail_cache[cache_key] = "failed"
                return result
            
            # ALKIS removed - no longer supported

            # data_type might be "lod1_bb_fallback" etc.
            # we normalize it for storage
            real_dtype = data_type.replace("_bb_fallback", "")
            tile_dir = self.map_dir / real_dtype / tile_id
            
            # Check for actual data files (not just leftover .zip from failed extraction)
            data_exts = {
                'lod1': ['.gml', '.xml'],
                'lod2': ['.gml', '.xml'],
                'mesh': ['.obj'],
            }
            exts = data_exts.get(real_dtype, ['.gml', '.xml', '.obj'])
            if tile_dir.exists() and self._has_data_files(tile_dir, exts):
                return tile_dir
            # Clean up leftover corrupt zips
            if tile_dir.exists():
                for zf in tile_dir.glob('*.zip'):
                    zf.unlink()

            url, filename = self.get_url_for_tile(data_type, tile_id)

            if not url:
                return None
                
            tile_dir.mkdir(parents=True, exist_ok=True)
            zip_path = tile_dir / filename
            
            try:
                # Increase timeout for large city-wide archive (300MB+)
                timeout = 600 if "lod1_berlin" in url or "berlin_full" in tile_id else 120
                r = requests.get(url, stream=True, timeout=timeout)
                r.raise_for_status()

                download_with_progress(r, zip_path, desc=f"{data_type}/{tile_id}")
                
                # Extract
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(tile_dir)
                except (zipfile.BadZipFile, NotImplementedError, RuntimeError):
                    # Fallback to system unzip
                    import subprocess
                    subprocess.run(["unzip", "-o", "-q", str(zip_path), "-d", str(tile_dir)], check=True)
                
                # Clean up ZIP
                zip_path.unlink()
                return tile_dir
                
            except Exception as e:
                # Cache failures (especially 404s) to skip on future runs
                self._download_fail_cache[cache_key] = str(e)[:200]
                self._save_download_fail_cache()
                if '404' not in str(e):
                    print(f"   Error downloading {tile_id} ({url}): {e}")
                if tile_dir.exists():
                    shutil.rmtree(tile_dir)
                return None



    def _resolve_als_regions(self, tile_ids):
        """
        Populates a cache mapping tile_ids (filenames) to their containing Region Zip."""
        if not hasattr(self, "_als_cache"):
            self._als_cache = {}
            
        # Identify missing
        missing = [t for t in tile_ids if t not in self._als_cache]
        if not missing:
            return

        try:
            from remotezip import RemoteZip
        except ImportError:
            print("   Error: 'remotezip' not installed. ALS download unavailable.")
            return

        # Simple approach: Check all regions until found
        # (Optimization: We could stop checking a region once we ruled it out, 
        # but files are scattered. We iterate regions.)
        
        found_count = 0
        for region_zip in self.ALS_REGIONS:
             if found_count == len(missing):
                 break
                 
             url = self.BASE_URLS["als_berlin_base"] + region_zip
             try:
                 with RemoteZip(url) as z:
                     namelist = set(z.namelist())
                     
                 for t in missing:
                     # Check .las filename
                     filename = t + ".las"
                     if filename in namelist:
                         self._als_cache[t] = (url, filename)
                         # Simple optimization: if we found it, we don't need to look in other zips for THIS tile
                         # But we continue with current zip for OTHER tiles
                 
                 found_count = sum(1 for t in missing if t in self._als_cache)
             except Exception as e:
                 print(f"   Error accessing ALS region {region_zip}: {e}")


    def _download_als_tile_unlocked(self, tile_id):
        """
        Downloads a specific ALS tile. Tries Berlin first, then Brandenburg fallback."""
        tile_dir = self.map_dir / "als" / tile_id
        if tile_dir.exists() and self._has_data_files(tile_dir, ['.las', '.laz']):
            return tile_dir
        # Clean up leftover corrupt files from failed previous downloads
        if tile_dir.exists():
            for zf in tile_dir.glob('*.zip'):
                zf.unlink()

        # Try Berlin first (regional ZIPs via remotezip)
        if not hasattr(self, "_als_cache") or tile_id not in self._als_cache:
            self._resolve_als_regions([tile_id])
            
        if tile_id in self._als_cache:
            url, filename = self._als_cache[tile_id]
            tile_dir.mkdir(parents=True, exist_ok=True)
            try:
                from remotezip import RemoteZip
                with RemoteZip(url) as z:
                    z.extract(filename, tile_dir)
                return tile_dir
            except Exception as e:
                print(f"   Error extracting Berlin ALS {tile_id}: {e}")
                if tile_dir.exists():
                    shutil.rmtree(tile_dir)

        # Fallback: Try Brandenburg ALS (direct per-tile ZIP)
        bb_tile_id = self._berlin_als_to_bb_als(tile_id)
        if bb_tile_id:
            return self._download_bb_als_tile(tile_id, bb_tile_id)
        return None

    def _download_bb_als_tile(self, tile_id, bb_tile_id):
        """
        Downloads ALS tile from Brandenburg (Geobasis-BB) source.
        BB ALS tiles are direct per-tile ZIPs containing .laz files."""
        tile_dir = self.map_dir / "als" / tile_id
        if tile_dir.exists() and self._has_data_files(tile_dir, ['.las', '.laz']):
            return tile_dir

        url = self.BASE_URLS["als_bb"] + f"{bb_tile_id}.zip"
        tile_dir.mkdir(parents=True, exist_ok=True)
        zip_path = tile_dir / f"{bb_tile_id}.zip"

        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()

            download_with_progress(r, zip_path, desc=f"ALS/{bb_tile_id}")

            # Extract
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tile_dir)
            finally:
                zip_path.unlink(missing_ok=True)

            # Check we got something
            laz_files = list(tile_dir.glob("*.laz")) + list(tile_dir.glob("*.las"))
            if laz_files:
                return tile_dir
            else:
                print(f"   BB ALS {bb_tile_id}: ZIP contained no .laz/.las files")
                if tile_dir.exists():
                    shutil.rmtree(tile_dir)
                return None

        except Exception as e:
            # Clean up on failure
            zip_path.unlink(missing_ok=True)
            if tile_dir.exists() and not any(tile_dir.iterdir()):
                shutil.rmtree(tile_dir)
            return None


    def download_dop_tile(self, tile_id):
        """Public API: download DOP tile with cross-process locking."""
        with self._tile_lock("dop", tile_id):
            return self._download_dop_tile_unlocked(tile_id)

    def get_dop_year_tile_dir(self, year: int) -> Path:
        """Returns the directory where tiles for a given DOP year are stored."""
        return self.map_dir / f"dop_{year}"

    def download_dop_year(self, year: int, bbox=None) -> Path:
        """
        Downloads region ZIPs for a given DOP year from the Berlin GDI ATOM feed
        into map/dop_{year}/ and extracts them.
        Skips regions that are already present on disk.

        Args:
            year: DOP acquisition year (must be in DOP_YEARS catalog).
            bbox: Optional (min_x, min_y, max_x, max_y) in UTM zone 33N.
                  If given, only regions whose bounding box intersects this bbox
                  are downloaded (typically 1-2 out of 9 for a Berlin sequence).

        Returns the tile directory path, or None if the year is unknown."""
        if year not in DOP_YEARS:
            print(f"   Warning: DOP year {year} not in catalog. Available: {sorted(DOP_YEARS.keys())}")
            return None

        dataset = DOP_YEARS[year]
        tile_dir = self.get_dop_year_tile_dir(year)
        tile_dir.mkdir(parents=True, exist_ok=True)

        base_url = f"https://gdi.berlin.de/data/{dataset}/atom/"

        # Filter regions by bbox if provided
        if bbox is not None:
            qmin_x, qmin_y, qmax_x, qmax_y = bbox
            regions_to_fetch = [
                r for r in DOP_REGIONS
                if r not in DOP_REGION_BBOX or (
                    DOP_REGION_BBOX[r][0] <= qmax_x and
                    DOP_REGION_BBOX[r][2] >= qmin_x and
                    DOP_REGION_BBOX[r][1] <= qmax_y and
                    DOP_REGION_BBOX[r][3] >= qmin_y
                )
            ]
            if len(regions_to_fetch) < len(DOP_REGIONS):
                skipped = [r for r in DOP_REGIONS if r not in regions_to_fetch]
                print(f"   Skipping {len(skipped)} non-overlapping region(s): {skipped}")
        else:
            regions_to_fetch = list(DOP_REGIONS)

        # Resolve URL-safe region names (umlaut for 2025+, ASCII for older)
        url_region_names = get_dop_region_names(year)
        # Build mapping: canonical ASCII name -> URL name
        _ascii_to_url = dict(zip(DOP_REGIONS, url_region_names))

        all_ok = True
        for region in regions_to_fetch:
            url_region = _ascii_to_url.get(region, region)
            region_zip_name = f"{url_region}.zip"
            region_out = tile_dir / region  # always ASCII on disk

            # Skip if already extracted (any image file present)
            if region_out.exists() and self._has_data_files(
                region_out, ['.ecw', '.tif', '.jp2', '.jpg', '.png']
            ):
                continue

            with self._tile_lock(f"dop_{year}", region):
                # Re-check inside lock
                if region_out.exists() and self._has_data_files(
                    region_out, ['.ecw', '.tif', '.jp2', '.jpg', '.png']
                ):
                    continue

                url = base_url + region_zip_name  # requests handles UTF-8 encoding
                zip_path = tile_dir / f"{region}.zip"  # ASCII on disk
                print(f"   Downloading DOP {year} – {url_region} ({url})...")

                try:
                    r = requests.get(url, stream=True, timeout=600)
                    r.raise_for_status()

                    download_with_progress(r, zip_path, desc=f"DOP {year}/{url_region}")

                    region_out.mkdir(parents=True, exist_ok=True)
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zf:
                            zf.extractall(region_out)
                    except (zipfile.BadZipFile, NotImplementedError, RuntimeError):
                        import subprocess
                        subprocess.run(
                            ["unzip", "-o", "-q", str(zip_path), "-d", str(region_out)],
                            check=True
                        )

                    zip_path.unlink(missing_ok=True)
                    print(f"   ✓ DOP {year} – {url_region} extracted")

                except Exception as e:
                    zip_path.unlink(missing_ok=True)
                    print(f"   ✗ DOP {year} – {url_region} failed: {e}")
                    all_ok = False

        return tile_dir if all_ok else tile_dir  # always return dir (partial ok)

    @staticmethod
    def _has_data_files(tile_dir, extensions):
        """
        Check if tile_dir contains actual data files (not just leftover .zip/.xml/.html).
        Returns True if at least one file with one of the given extensions exists
        (searches recursively to handle nested extraction)."""
        for ext in extensions:
            if list(tile_dir.rglob(f"*{ext}")):
                return True
        return False

    def _download_dop_tile_unlocked(self, tile_id):
        """
        Downloads a specific DOP tile from Brandenburg (Geobasis-BB).
        Format: dop_33391-5819.zip"""
        tile_dir = self.map_dir / "dop" / tile_id
        if tile_dir.exists() and self._has_data_files(tile_dir, ['.tif', '.jp2']):
            return tile_dir
        # Clean up leftover corrupt zips from failed previous downloads
        if tile_dir.exists():
            for zf in tile_dir.glob('*.zip'):
                zf.unlink()

        # Brandenburg DOP tiles
        filename = f"dop_{tile_id}.zip"
        url = self.BASE_URLS["dop"] + filename
        
        tile_dir.mkdir(parents=True, exist_ok=True)
        zip_path = tile_dir / filename
        
        try:
            r = requests.get(url, stream=True, timeout=300)
            r.raise_for_status()
            
            download_with_progress(r, zip_path, desc=f"DOP/{tile_id}")
            
            # Extract
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tile_dir)
            except (zipfile.BadZipFile, NotImplementedError, RuntimeError):
                import subprocess
                subprocess.run(["unzip", "-o", "-q", str(zip_path), "-d", str(tile_dir)], check=True)
            
            zip_path.unlink()
            return tile_dir
            
        except Exception as e:
            print(f"   Error downloading DOP {tile_id} ({url}): {e}")
            if tile_dir.exists():
                shutil.rmtree(tile_dir)
            return None

    def download_dsm_tile(self, tile_id):
        """Public API: download DSM tile with cross-process locking."""
        with self._tile_lock("dsm", tile_id):
            return self._download_dsm_tile_unlocked(tile_id)

    def _download_dsm_tile_unlocked(self, tile_id):
        """
        Downloads a specific DSM tile from Brandenburg (Geobasis-BB).
        Format: bdom_33391-5819.zip"""
        tile_dir = self.map_dir / "dsm" / tile_id
        if tile_dir.exists() and self._has_data_files(tile_dir, ['.tif', '.jp2']):
            return tile_dir
        # Clean up leftover corrupt zips from failed previous downloads
        if tile_dir.exists():
            for zf in tile_dir.glob('*.zip'):
                zf.unlink()

        # Brandenburg DSM tiles (BDOM = Basis-DGM)
        filename = f"bdom_{tile_id}.zip"
        url = self.BASE_URLS["dsm"] + filename
        
        tile_dir.mkdir(parents=True, exist_ok=True)
        zip_path = tile_dir / filename
        
        try:
            r = requests.get(url, stream=True, timeout=300)
            r.raise_for_status()
            
            download_with_progress(r, zip_path, desc=f"DSM/{tile_id}")
            
            # Extract
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tile_dir)
            except (zipfile.BadZipFile, NotImplementedError, RuntimeError):
                import subprocess
                subprocess.run(["unzip", "-o", "-q", str(zip_path), "-d", str(tile_dir)], check=True)
            
            zip_path.unlink()
            return tile_dir
            
        except Exception as e:
            print(f"   Error downloading DSM {tile_id} ({url}): {e}")
            if tile_dir.exists():
                shutil.rmtree(tile_dir)
            return None

    def download_all(self, needed_tiles, max_workers=2, bounds=None, per_tile_timeout=600):
        """
        Downloads all needed tiles using multiprocessing.
        needed_tiles: dict { 'dop': set(...), ... }
        bounds: Optional (min_e, min_n, max_e, max_n) for filtering (e.g. LoD1)
        per_tile_timeout: Max seconds to wait per tile result (default 120s)"""
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout
        
        results = {t: [] for t in needed_tiles.keys()}
        
        total_tiles = sum(len(v) for v in needed_tiles.values())
        if total_tiles == 0:
            print("   No tiles to download.")
            return results

        print(f"   Starting download of {total_tiles} tiles using {max_workers} workers...")
        
        tasks = {}
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            for dtype, tids in needed_tiles.items():
                real_dtype = dtype.replace("_bb_fallback", "")
                for tid in tids:
                    future = executor.submit(self.download_tile, dtype, tid, bounds=bounds)
                    tasks[future] = (real_dtype, tid)

            
            with tqdm(total=total_tiles, desc="Downloading Geodata") as pbar:
                done_futures = set()
                try:
                    # Wait at most per_tile_timeout between successive completions
                    for future in as_completed(tasks, timeout=per_tile_timeout):
                        pbar.update(1)
                        done_futures.add(future)
                        dtype, tid = tasks[future]
                        try:
                            tile_dir = future.result(timeout=10)
                            if tile_dir:
                                if dtype not in results:
                                    results[dtype] = []
                                
                                files = self.get_filtered_tile_files(dtype, tile_dir, bounds)
                                
                                for f in files:
                                    if f not in results[dtype]:
                                        results[dtype].append(f)
                        except FutureTimeout:
                            print(f"   TIMEOUT getting result for {dtype}/{tid} — skipping")
                        except Exception as e:
                            print(f"   Error processing {dtype}/{tid}: {e}")
                except FutureTimeout:
                    pass  # No new result within per_tile_timeout — remaining tiles are hung
                
                # Report and skip any remaining hung futures
                remaining = [f for f in tasks if f not in done_futures]
                if remaining:
                    print(f"   {len(remaining)} tile(s) timed out (>{per_tile_timeout}s) — skipping:")
                    for future in remaining:
                        dtype, tid = tasks[future]
                        print(f"     - {dtype}/{tid}")
                        pbar.update(1)
                        future.cancel()
        finally:
            # Don't wait for hung threads — cancel_futures requires Python 3.9+
            executor.shutdown(wait=False, cancel_futures=True)

        
        # Persist failure cache after all downloads complete
        self._save_download_fail_cache()
        
        return results


    def get_filtered_tile_files(self, data_type, tile_dir, bounds=None):
        """
        Returns files from tile_dir, optionally filtered by spatial bounds.
        bounds = (min_e, min_n, max_e, max_n) in UTM.
        Parses filenames like 'LoD1_370_5808.xml' (1km grid)."""
        all_files = []
        if data_type.startswith("dop"):
            # Include multi-year DOP directories and Brandenburg fallback
            # ECW files are included for discovery even though rasterio may not read them;
            # the caller (preprocess_dop_multiyear) falls back to WMS for unreadable tiles.
            all_files = (list(tile_dir.rglob("*.tif")) + list(tile_dir.rglob("*.jp2"))
                         + list(tile_dir.rglob("*.jpg")) + list(tile_dir.rglob("*.ecw")))
        elif data_type == "dsm":

            all_files = list(tile_dir.glob("*.tif")) + list(tile_dir.glob("*.jp2"))
        elif data_type == "mesh":
            all_files = list(tile_dir.glob("*.obj"))
        elif data_type == "lod1" or data_type == "lod2":
             # Recursively find GML/XML
             all_files = list(tile_dir.rglob("*.gml")) + list(tile_dir.rglob("*.xml"))
        elif data_type == "als":
             all_files = list(tile_dir.glob("*.las")) + list(tile_dir.glob("*.laz"))
        elif data_type == "alkis":
             all_files = list(tile_dir.rglob("*.shp")) + list(tile_dir.rglob("*.zip"))
        
        if not bounds or not all_files:
            return [str(f.absolute()) for f in all_files]

        # Filter Logic for LoD1/LoD2 Berlin naming convention
        # Format: LoD1_370_5808 (1km) or LoD2_391_5819 (1km? actually file says 33_391_5819_1_BE)
        min_e, min_n, max_e, max_n = bounds
        filtered_files = []
        
        for f in all_files:
            name = f.name
            try:
                # Pre-compiled patterns (module-level would be even better but keep locality)
                # 1. Berlin patterns: _370_5808_ or _33_370_5808_ or LoD1_370_5808.xml
                # 2. Brandenburg patterns: 33391-5820.tif or dop_33391-5820.tif
                
                # Try Berlin pattern first
                match = _RE_BERLIN_TILE.search(name)
                if match:
                    e_km = int(match.group(1))
                    n_km = int(match.group(2))
                    # group(3) captures tile size in km (e.g. '2' for 2km tiles)
                    tile_size_km = int(match.group(3)) if match.group(3) else 1
                    tile_size = tile_size_km * 1000
                else:
                    # Try Brandenburg pattern: 33391-5820 or 33391_5820
                    match = _RE_BB_TILE.search(name)
                    if match:
                        e_km = int(match.group(1))
                        n_km = int(match.group(3))
                        tile_size = 1000
                
                if match:
                    tile_min_e = e_km * 1000
                    tile_min_n = n_km * 1000
                    tile_max_e = tile_min_e + tile_size
                    tile_max_n = tile_min_n + tile_size
                    
                    # Check overlap
                    if not (tile_max_e < min_e or tile_min_e > max_e or tile_max_n < min_n or tile_min_n > max_n):
                        filtered_files.append(f)
                    continue # Handled
                
                # Fallback: if we can't parse it, include it unless it's the Berlin full archive
                if data_type == "lod1" and "berlin_full" in str(tile_dir):
                    pass # Only strict filtering for the massive Berlin archive
                else:
                    filtered_files.append(f)
                      
            except Exception:
                filtered_files.append(f)
                
        return [str(f.absolute()) for f in filtered_files]
    def get_tile_metadata(self, data_type, tile_id):
        """
        Retrieves the metadata (URL, Capture Date, License, GSD) for a given tile."""
        # Get source URL
        url, _ = self.get_url_for_tile(data_type, tile_id)
        
        metadata = {
            "source_url": url,
            "license": "dl-de/by-2-0", # Open Data license for Berlin/Brandenburg
            "capture_date": "Unknown",
            "publication_date": "Unknown",
        }
        
        # Only include GSD for raster data types (DOP, DSM), not for vector/mesh/LoD
        if data_type.startswith("dop"):
            metadata["gsd"] = DOP_GSD
        elif data_type == "dsm":
            metadata["gsd"] = DOP_GSD  # DSM also 0.2m for Berlin
        
        real_dtype = data_type.replace("_bb_fallback", "")
        tile_dir = self.map_dir / real_dtype / tile_id
        
        if not tile_dir.exists():
            return metadata
            
        if data_type == "mesh":
            if "2025" in url:
                metadata["capture_date"] = "2024 / 2025"
            return metadata

        # For Berlin GDI DOPs (dop_<year>), we know the year from the key
        if data_type.startswith("dop_") and data_type != "dop_bb_fallback":
            try:
                year = int(data_type.split("_")[1])
                metadata["capture_date"] = f"{year}"
                metadata["gsd"] = DOP_GSD
            except ValueError:
                pass

        # Try to parse capture dates from .html or .xml files in the tile directory
        meta_files = list(tile_dir.glob("*.html")) + list(tile_dir.glob("*.xml"))
        
        for meta_file in meta_files:
            try:
                content = meta_file.read_text(encoding='utf-8', errors='ignore')
                
                # Priority 1: Bildflugdatum (Flight / Capture Date)
                match = re.search(r'(?:Bildflugdatum|Flugdatum).*?(?:<td.*?>|:)\s*(\d{4}-\d{2}-\d{2})', content, re.IGNORECASE | re.DOTALL)
                if match:
                    metadata["capture_date"] = match.group(1)
                    
                # Also extract Veröffentlichung (Publication Date)
                match_pub = re.search(r'Ver.*?ffentlichung.*?(?:<td.*?>|:)\s*(\d{4}-\d{2}-\d{2})', content, re.IGNORECASE | re.DOTALL)
                if match_pub:
                    metadata["publication_date"] = match_pub.group(1)
                
                if match:
                    break
                    
                # Priority 2: Aktualisierung (Update Date for LoD mostly)
                match = re.search(r'Aktualisierung.*?(?:<td.*?>|:)\s*(\d{4}-\d{2}-\d{2})', content, re.IGNORECASE | re.DOTALL)
                if match:
                    metadata["capture_date"] = match.group(1)
                    break
                    
                # Priority 3: Erstellungsdatum (Creation Date)
                match = re.search(r'Erstellungsdatum.*?(?:<td.*?>|:)\s*(\d{4}-\d{2}-\d{2})', content, re.IGNORECASE | re.DOTALL)
                if match:
                    metadata["capture_date"] = match.group(1)
                    break
            except Exception:
                pass
                
        return metadata

    def detect_bb_dop_year(self, tile_ids):
        """
        Detect the capture year of Brandenburg (BB) DOP tiles from their HTML metadata.
        Checks the first tile that has an HTML file with a Bildflugdatum.
        
        Args:
            tile_ids: List of BB tile IDs (e.g., ['33391-5820', '33392-5820'])
            
        Returns:
            int: Detected year (e.g. 2024), or None if detection failed"""
        for tid in tile_ids:
            tile_dir = self.map_dir / "dop" / tid
            if not tile_dir.exists():
                continue
            for html_file in tile_dir.glob("*.html"):
                try:
                    content = html_file.read_text(encoding='utf-8', errors='ignore')
                    match = re.search(
                        r'(?:Bildflugdatum|Flugdatum).*?(?:<td.*?>|:)\s*(\d{4})-\d{2}-\d{2}',
                        content, re.IGNORECASE | re.DOTALL
                    )
                    if match:
                        return int(match.group(1))
                except Exception:
                    pass
        return None


if __name__ == "__main__":
    # Test discovery
    manager = GeodataManager()
    # Fernsehturm approx UTM: 391650, 5820100
    pos = np.array([[391650, 5820100]])
    tiles = manager.discover_tiles(pos, data_types=["mesh", "dop"])
    print("Required Tiles:", tiles)
