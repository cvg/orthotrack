"""
Image utility functions."""

import numpy as np
import cv2
import torch
from typing import Tuple
from pathlib import Path

SKY_COLOR_BGR = np.array([235, 206, 135], dtype=np.uint8)


def downsample_image(image: np.ndarray, max_size: int = 1024) -> Tuple[np.ndarray, float]:
    """
    Resize image preserving aspect ratio so that max dimension is at most max_size.

    Args:
        image: Input image (H, W, C) or (H, W).
        max_size: Maximum dimension.

    Returns:
        resized: Resized image.
        scale: Scale factor to map from resized coords back to original
               (i.e. original_coord = resized_coord * scale)."""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image, 1.0

    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), 1.0 / scale


def unnormalize_image(tensor, normalization='01'):
    """Unnormalize image tensor (C, H, W) to numpy (H, W, C) in [0, 255]."""
    if tensor is None:
        return None

    if isinstance(tensor, np.ndarray):
        img = tensor.copy()
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] == 3 and img.ndim == 3:
            img = img.transpose(1, 2, 0)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        return np.ascontiguousarray(img)

    if tensor.ndim == 4:
        tensor = tensor[0]
    img = tensor.clone()
    if normalization == 'imagenet':
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
        img = img * std + mean
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).contiguous().cpu().numpy()
    return np.ascontiguousarray((img * 255).astype(np.uint8))
def is_frame_black(frame_img: np.ndarray, threshold: float = 1.0) -> bool:
    """Check if a frame is considered black based on average pixel intensity."""
    if frame_img is None:
        return True
    return np.mean(frame_img) < threshold

def is_frame_sky_colored(frame_img: np.ndarray, sky_color_bgr: np.ndarray = None, 
                         color_threshold: float = 20.0, sky_pixel_threshold: float = 0.90) -> bool:
    """
    Check if frame is mostly the sky/background color (rendering failure).
    
    A frame is considered sky-colored if >90% of pixels are within color_threshold
    of the sky color."""
    if frame_img is None:
        return True
    if sky_color_bgr is None:
        sky_color_bgr = SKY_COLOR_BGR
    
    # Count pixels that are close to sky color
    diff = np.abs(frame_img.astype(np.float32) - sky_color_bgr.astype(np.float32))
    sky_mask = np.all(diff < color_threshold, axis=2)
    sky_ratio = np.mean(sky_mask)
    
    return sky_ratio > sky_pixel_threshold

def is_frame_invalid(frame_img: np.ndarray, black_threshold: float = 1.0, 
                     sky_color_bgr: np.ndarray = None) -> bool:
    """Check if a frame is invalid (black OR mostly sky-colored background)."""
    if frame_img is None:
        return True
    return is_frame_black(frame_img, black_threshold) or is_frame_sky_colored(frame_img, sky_color_bgr)

def list_video_frames(seq_dir: Path, stride: int = 10) -> list:
    """Return sorted list of frame indices to evaluate (strided)."""
    video_path = seq_dir / 'video.mp4'
    if not video_path.exists():
        return []
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return list(range(0, n_frames, stride))

def read_video_frame(seq_dir: Path, frame_idx: int) -> np.ndarray:
    """Read a single frame from video.mp4 as RGB uint8."""
    cap = cv2.VideoCapture(str(seq_dir / 'video.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, bgr = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {seq_dir}/video.mp4")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def filter_by_shi_tomasi(
    pts_2d: np.ndarray,
    image_gray: np.ndarray,
    min_score: float = 0.001,
    block_size: int = 7,
) -> np.ndarray:
    """Filter 2D points by Shi-Tomasi trackability score."""
    if len(pts_2d) == 0:
        return np.zeros(0, dtype=bool)
    if image_gray.dtype != np.uint8:
        image_gray = np.clip(image_gray, 0, 255).astype(np.uint8)
    eigen_map = cv2.cornerMinEigenVal(image_gray, blockSize=block_size)
    h, w = image_gray.shape[:2]
    coords = np.round(pts_2d).astype(int)
    xs = np.clip(coords[:, 0], 0, w - 1)
    ys = np.clip(coords[:, 1], 0, h - 1)
    scores = eigen_map[ys, xs]
    mask = scores >= min_score
    return mask
