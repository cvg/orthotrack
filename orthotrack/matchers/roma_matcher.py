"""
RoMaV2-based feature matcher for OrthoTrack.
Matches query images to DOP orthophotos using RoMaV2 dense matching.

Supports:
- Single image matching
- Batch matching with multiple DOP crops for better coverage
- Confidence-based filtering"""

import sys
import os
from pathlib import Path

# Add RoMaV2 to path
_roma_src = Path(__file__).resolve().parents[2] / "thirdparty" / "RoMaV2" / "src"
if _roma_src.is_dir():
    sys.path.insert(0, str(_roma_src))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Optional, Dict
from romav2 import RoMaV2
from romav2.device import device
from romav2.geometry import prec_mat_from_prec_params as _prec_mat_from_prec_params

from utils.matching import MatchResult




class RoMaV2Matcher:
    """Wrapper around RoMaV2 for matching query images to DOP orthophotos."""
    
    def __init__(self, setting: str = "precise"):
        """
        Initialize the RoMaV2 matcher.
        
        Args:
            setting: RoMaV2 setting - "turbo", "fast", "base", "precise",
                     or a custom string like "lr800" / "lr800_bidir" to use
                     LR=800 without HR refinement (faster than precise)."""
        torch.set_float32_matmul_precision("highest")
        
        print(f"Initializing RoMaV2 matcher with setting: {setting}")
        
        # Initialize RoMaV2 with compilation disabled for faster startup
        self.model = RoMaV2(RoMaV2.Cfg(compile=False))
        
        # Handle custom settings
        if setting == "lr800":
            # LR=800 (same resolution as precise) but no HR, unidirectional
            # ~3x faster than precise with moderate accuracy loss
            self.model.apply_setting("base")
            self.model.H_lr = 800
            self.model.W_lr = 800
        elif setting == "turbo_lr256":
            # Even smaller than turbo: LR=256, no HR, unidirectional
            # Fastest available variant for CPU edge deployment
            self.model.apply_setting("turbo")
            self.model.H_lr = 256
            self.model.W_lr = 256
        elif setting == "turbo_lr224":
            self.model.apply_setting("turbo")
            self.model.H_lr = 224
            self.model.W_lr = 224
        elif setting == "lr800_bidir":
            # LR=800 with bidirectional but no HR
            # ~2.3x faster than precise
            self.model.apply_setting("precise")
            self.model.H_hr = None
            self.model.W_hr = None
        elif setting == "precise_unidir":
            # Full precise resolution (800+1280) but unidirectional
            # Saves ~40% over precise by skipping backward matching
            self.model.apply_setting("precise")
            self.model.bidirectional = False
        elif setting == "precise_1024":
            # Like precise but HR=1024 instead of 1280 (less HR computation)
            self.model.apply_setting("precise")
            self.model.H_hr = 1024
            self.model.W_hr = 1024
        else:
            self.model.apply_setting(setting)
        
        self.model.eval()
        
        # Get resolution settings
        self.H_lr = self.model.H_lr
        self.W_lr = self.model.W_lr
        self.H_hr = self.model.H_hr
        self.W_hr = self.model.W_hr
        self.name = f"RoMa-{setting}"
        
        print(f"  Low-res: {self.H_lr}x{self.W_lr}")
        if self.H_hr and self.W_hr:
            print(f"  High-res: {self.H_hr}x{self.W_hr}")
        print(f"  Bidirectional: {self.model.bidirectional}")
    
    @torch.inference_mode()
    def match(self, query_image: np.ndarray, dop_image: np.ndarray,
              num_matches: int = 5000) -> MatchResult:
        """
        Match a query image to a DOP orthophoto.
        
        Args:
            query_image: Query image as numpy array (H, W, 3) uint8
            dop_image: DOP image as numpy array (H, W, 3) uint8
            num_matches: Number of matches to sample
            
        Returns:
            MatchResult with matched keypoints"""
        # Store original sizes
        query_h, query_w = query_image.shape[:2]
        dop_h, dop_w = dop_image.shape[:2]
        
        # Match using RoMaV2
        # RoMaV2 expects PIL images or paths
        query_pil = Image.fromarray(query_image)
        dop_pil = Image.fromarray(dop_image)
        
        # Get dense predictions
        preds = self.model.match(query_pil, dop_pil)
        
        # Sample correspondences
        # preds contains warp_AB which maps from A (query) to B (dop)
        matches, confidences, _, _ = self.model.sample(preds, num_matches)
        
        # matches is (N, 4) with normalized coordinates: [x_A, y_A, x_B, y_B]
        matches = matches.cpu().numpy()
        confidences = confidences.cpu().numpy()
        
        # Convert from normalized [-1, 1] to pixel coordinates
        kpts_query = self._normalized_to_pixel(matches[:, :2], query_h, query_w)
        kpts_dop = self._normalized_to_pixel(matches[:, 2:], dop_h, dop_w)
        
        return MatchResult(
            kpts_query=kpts_query,
            kpts_dop=kpts_dop,
            confidences=confidences,
            query_size=(query_h, query_w),
            dop_size=(dop_h, dop_w)
        )
    
    @torch.inference_mode()
    def match_batch(self, query_image: np.ndarray, dop_images: List[np.ndarray],
                    num_matches_per_crop: int = 3000) -> List[MatchResult]:
        """
        Match a query image to multiple DOP orthophotos in a single GPU batch.
        
        Args:
            query_image: Query image (H, W, 3)
            dop_images: List of DOP images
            num_matches_per_crop: Number of matches for each pair
            
        Returns:
            List of MatchResult objects"""
        if not dop_images:
            return []
            
        batch_size = len(dop_images)
        query_h, query_w = query_image.shape[:2]
        dop_sizes = [img.shape[:2] for img in dop_images]
        
        # Load and resize images as tensors
        # We must resize BEFORE stacking if inputs have different sizes
        img_q = self.model._load_image(query_image) # (1, 3, H, W)
        
        # Prepare batch tensors
        img_q_lr = F.interpolate(img_q, size=(self.H_lr, self.W_lr), mode="bicubic", align_corners=False, antialias=True).repeat(batch_size, 1, 1, 1)
        
        imgs_d_lr_list = []
        imgs_d_hr_list = []
        
        for img in dop_images:
            tensor = self.model._load_image(img)
            lr = F.interpolate(tensor, size=(self.H_lr, self.W_lr), mode="bicubic", align_corners=False, antialias=True)
            imgs_d_lr_list.append(lr)
            
            if self.H_hr is not None and self.W_hr is not None:
                hr = F.interpolate(tensor, size=(self.H_hr, self.W_hr), mode="bicubic", align_corners=False, antialias=True)
                imgs_d_hr_list.append(hr)
        
        imgs_d_lr = torch.cat(imgs_d_lr_list, dim=0) # (B, 3, H_lr, W_lr)
        
        # Resize query to high-res if needed
        img_q_hr = None
        imgs_d_hr = None
        if self.H_hr is not None and self.W_hr is not None:
            img_q_hr = F.interpolate(img_q, size=(self.H_hr, self.W_hr), mode="bicubic", align_corners=False, antialias=True).repeat(batch_size, 1, 1, 1)
            imgs_d_hr = torch.cat(imgs_d_hr_list, dim=0)
            
        # Run model in batch
        # self.model(lr_A, lr_B, hr_A, hr_B) returns OrderedDict of predictions
        all_preds = self.model(img_q_lr, imgs_d_lr, img_A_hr=img_q_hr, img_B_hr=imgs_d_hr)
        
        # Extract features and map confidence (replicating RoMaV2.match logic)
        results = []
        for i in range(batch_size):
            # Extract single pair prediction from batch
            pred = {
                "warp_AB": all_preds["warp_AB"][i:i+1],
                "confidence_AB": all_preds["confidence_AB"][i:i+1]
            }
            if "warp_BA" in all_preds and all_preds["warp_BA"] is not None:
                pred["warp_BA"] = all_preds["warp_BA"][i:i+1]
                pred["confidence_BA"] = all_preds["confidence_BA"][i:i+1]
            
            # Replicate _map_confidence from romav2.py
            # Note: We need to handle this manually since it's not easily exposed
            # but RoMaV2.sample expects overlap_AB and precision_AB
            
            overlap_AB, precision_AB = self._map_confidence_tensor(pred["confidence_AB"])
            pred["overlap_AB"] = overlap_AB
            pred["precision_AB"] = precision_AB
            
            if "confidence_BA" in pred:
                overlap_BA, precision_BA = self._map_confidence_tensor(pred["confidence_BA"])
                pred["overlap_BA"] = overlap_BA
                pred["precision_BA"] = precision_BA
            
            # Sample correspondences for this pair
            matches, confidences, _, _ = self.model.sample(pred, num_matches_per_crop)
            
            matches = matches.cpu().numpy()
            confidences = confidences.cpu().numpy()
            
            # Convert to pixel coordinates
            kpts_query = self._normalized_to_pixel(matches[:, :2], query_h, query_w)
            kpts_dop = self._normalized_to_pixel(matches[:, 2:], dop_sizes[i][0], dop_sizes[i][1])
            
            results.append(MatchResult(
                kpts_query=kpts_query,
                kpts_dop=kpts_dop,
                confidences=confidences,
                query_size=(query_h, query_w),
                dop_size=dop_sizes[i]
            ))
            
        return results

    def _map_confidence_tensor(self, confidence: torch.Tensor):
        # Local copy of _map_confidence from romav2.py
        overlap = confidence[..., :1].sigmoid()
        # threshold = self.model.threshold (often None)
        if self.model.threshold is not None:
            overlap[overlap > self.model.threshold] = 1.0
        
        precision = _prec_mat_from_prec_params(confidence[..., 1:4])
        return overlap, precision

    def _normalized_to_pixel(self, coords: np.ndarray, H: int, W: int) -> np.ndarray:
        """Convert normalized [-1, 1] coordinates to pixel coordinates."""
        pixel_coords = np.zeros_like(coords)
        pixel_coords[:, 0] = (coords[:, 0] + 1) / 2 * W  # x
        pixel_coords[:, 1] = (coords[:, 1] + 1) / 2 * H  # y
        return pixel_coords
    
    




