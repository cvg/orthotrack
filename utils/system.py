import os
import platform
import shutil
import subprocess
from pathlib import Path
import torch

def generate_demo_video(output_dir: Path, fig_ext: str = "png", fps: float = 2.0):
    """Generate a demo video from keyframe and tracking visualizations.

    Collects all per-frame vis images from output_dir/keyframes/, converts to
    PNG if needed (for non-PNG fig_ext), and stitches with ffmpeg."""
    kf_dir = output_dir / "keyframes"
    if not kf_dir.exists():
        print("No keyframes directory found, skipping demo video.")
        return

    # Collect all per-frame vis files, sorted by frame number
    import re
    vis_files = []
    for f in sorted(kf_dir.iterdir()):
        if not f.suffix == f".{fig_ext}":
            continue
        # Match patterns like keyframe_0000.ext, keyframe_0039_tracked.ext
        match = re.match(r'keyframe_(\d+)(?:_tracked)?\.', f.name)
        if match:
            frame_num = int(match.group(1))
            vis_files.append((frame_num, f))

    if not vis_files:
        print("No visualization files found for demo video.")
        return

    vis_files.sort(key=lambda x: x[0])
    print(f"\nGenerating demo video from {len(vis_files)} frames...")

    # If fig_ext is not PNG, convert to temp PNGs for ffmpeg
    if fig_ext != "png":
        tmp_dir = output_dir / "_tmp_demo_video"
        tmp_dir.mkdir(exist_ok=True)

        for i, (frame_num, fpath) in enumerate(vis_files):
            dst = tmp_dir / f"frame_{i:04d}.png"
            # Use matplotlib to read PDF/SVG and re-save as PNG
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(str(fpath), dpi=150, first_page=1, last_page=1)
                images[0].save(str(dst), "PNG")
            except ImportError:
                # Fallback: use ImageMagick convert
                result = subprocess.run(
                    ["convert", "-density", "150", str(fpath), str(dst)],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    print(f"  Failed to convert {fpath.name}: {result.stderr}")
                    continue
        frame_pattern = str(tmp_dir / "frame_%04d.png")
        cleanup_dir = tmp_dir
    else:
        # Rename/symlink PNGs with sequential numbering
        tmp_dir = output_dir / "_tmp_demo_video"
        tmp_dir.mkdir(exist_ok=True)
        for i, (frame_num, fpath) in enumerate(vis_files):
            dst = tmp_dir / f"frame_{i:04d}.png"
            if dst.exists():
                dst.unlink()
            os.symlink(fpath.resolve(), dst)
        frame_pattern = str(tmp_dir / "frame_%04d.png")
        cleanup_dir = tmp_dir

    # Get dimensions from first frame for padding
    first_frame = tmp_dir / "frame_0000.png"
    if first_frame.exists():
        import cv2
        ref = cv2.imread(str(first_frame))
        if ref is not None:
            h, w = ref.shape[:2]
            w = w if w % 2 == 0 else w + 1
            h = h if h % 2 == 0 else h + 1
            scale_filter = f"scale={w}:{h}:flags=lanczos,setsar=1"
        else:
            scale_filter = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    else:
        scale_filter = "pad=ceil(iw/2)*2:ceil(ih/2)*2"

    mp4_path = output_dir / "demo_video.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-vf", scale_filter,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-bf", "0", "-crf", "18",
        str(mp4_path),
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Demo video saved: {mp4_path}")
    else:
        print(f"ffmpeg failed: {result.stderr}")

    # Cleanup temp dir
    if cleanup_dir.exists():
        shutil.rmtree(cleanup_dir, ignore_errors=True)


def get_device_info() -> dict:
    """Collect GPU and CPU information for reproducibility."""
    info = {
        "hostname": platform.node(),
        "cpu_model": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "gpu_name": None,
        "gpu_memory_mb": None,
        "cuda_version": None,
    }
    # Try to get CPU model from /proc/cpuinfo on Linux
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except (OSError, IOError):
        pass
    # GPU info via torch
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**2)
        info["cuda_version"] = torch.version.cuda
    return info

def download_with_progress(url_or_response, dest_path: str, desc: str = None, timeout: int = 120, chunk_size: int = 1024*1024):
    """Download a file with progress bar. Accepts either a URL string or a requests.Response."""
    import requests
    from pathlib import Path
    from tqdm import tqdm
    
    if isinstance(url_or_response, str):
        r = requests.get(url_or_response, stream=True, timeout=timeout)
        r.raise_for_status()
    else:
        r = url_or_response
        
    total = int(r.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f:
        if total > 5 * 1024 * 1024:
            with tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024,
                      desc=desc or Path(dest_path).name, leave=False) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
        else:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
