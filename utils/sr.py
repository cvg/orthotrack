import torch
from torch import nn as nn
from torch.nn import functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
from typing import List
import os

# --- RRDBNet Architecture ---

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = lambda: RRDB(nf, gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.body = make_layer(RRDB_block_f, nb)
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # upsampling
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.conv_body(self.body(fea))
        fea = fea + trunk

        fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))

        return out

# --- TextureEnhancer using Real-ESRGAN ---

class TextureEnhancer:
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading Real-ESRGAN for texture enhancement on {self.device}...")
        
        # RealESRGAN_x4plus architecture settings
        self.model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32).to(self.device).eval()
        
        # Load pre-trained weights
        # Using a more reliable URL (official Real-ESRGAN HuggingFace or similar)
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        try:
            # First try torch.hub
            checkpoint = torch.hub.load_state_dict_from_url(url, map_location=self.device, check_hash=False)
        except Exception as e:
            print(f"Warning: Could not load Real-ESRGAN weights from URL via torch.hub: {e}")
            # Fallback: try to download manually or check a different URL
            print("Attempting to continue without weights (will produce poor results)...")
            checkpoint = None

        if checkpoint:
            # Handle potential key mismatch
            if 'params_ema' in checkpoint:
                self.model.load_state_dict(checkpoint['params_ema'], strict=True)
            elif 'params' in checkpoint:
                self.model.load_state_dict(checkpoint['params'], strict=True)
            else:
                self.model.load_state_dict(checkpoint, strict=True)

    @torch.no_grad()
    def process_batch(self, pil_images: List[Image.Image], tile_size: int = 1024, tile_overlap: int = 32, auto_downsample: bool = True) -> List[Image.Image]:
        """
        Process images using tiling and FP16.
        Args:
            pil_images: List of PIL images.
            tile_size: Size of tiles for processing.
            tile_overlap: Overlap between tiles.
            auto_downsample: If True, downsamples large images (8k) to 2k before SR. 
                            If False, upscales input directly (for frame SR).
        """
        if not pil_images:
            return []
            
        results = []
        # Use FP16 if available
        use_fp16 = self.device == "cuda"
        if use_fp16:
            self.model.half()

        for pil_img in pil_images:
            # 1. Convert to tensor
            img_tensor = T.ToTensor()(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
            if use_fp16:
                img_tensor = img_tensor.half()
            
            b, c, h, w = img_tensor.size()
            scale = 4
            
            if auto_downsample and (h >= 4096 or w >= 4096):
                # Optimization for 8k textures: Downsample 8k -> 2k, then SR -> 8k.
                # This is significantly faster and enough for most renders.
                target_h, target_w = h // scale, w // scale
                input_tensor = F.interpolate(img_tensor, size=(target_h, target_w), mode='area')
                out_h, out_w = h, w
            else:
                # Direct SR: Input is current size, output is 4x larger.
                input_tensor = img_tensor
                target_h, target_w = h, w
                out_h, out_w = h * scale, w * scale
            
            # 2. Tiling setup
            tile = tile_size
            stride = tile - tile_overlap
            
            h_idx_list = list(range(0, target_h - tile, stride)) + [target_h - tile]
            w_idx_list = list(range(0, target_w - tile, stride)) + [target_w - tile]
            
            # Ensure indices unique and positive
            h_idx_list = sorted(list(set([max(0, i) for i in h_idx_list])))
            w_idx_list = sorted(list(set([max(0, i) for i in w_idx_list])))
            
            # Output buffer
            E = torch.zeros(b, c, out_h, out_w).to(self.device)
            W = torch.zeros(b, c, out_h, out_w).to(self.device)
            if use_fp16:
                E, W = E.half(), W.half()
            
            # 3. Process tiles
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    th = min(tile, target_h - h_idx)
                    tw = min(tile, target_w - w_idx)
                    in_patch = input_tensor[..., h_idx:h_idx+th, w_idx:w_idx+tw]
                    
                    # SR Inference
                    out_patch = self.model(in_patch)
                    
                    # Map back to output resolution
                    oh, ow = h_idx * scale, w_idx * scale
                    oth, otw = th * scale, tw * scale
                    
                    # Ensure out_patch matches planned size exactly
                    out_patch = out_patch[..., :oth, :otw]
                    
                    E[..., oh:oh+oth, ow:ow+otw].add_(out_patch)
                    W[..., oh:oh+oth, ow:ow+otw].add_(torch.ones_like(out_patch))
            
            # 4. Final averaging
            output = E.div_(W).clamp(0, 1)
            
            # 5. Robust PIL creation
            output_np = (output.squeeze(0).float().cpu().numpy() * 255.0).round().astype(np.uint8)
            output_np = np.transpose(output_np, (1, 2, 0)) # CHW to HWC
            out_pil = Image.fromarray(output_np)
            results.append(out_pil)
            
        return results

    def process_images_batched(self, image_paths: List[str], batch_size: int = 1, tile_size: int = 1024, auto_downsample: bool = True) -> List[Image.Image]:
        """Load, process, and return enhanced PIL images."""
        results = []
        for p in tqdm(image_paths, desc="Real-ESRGAN Processing"):
            try:
                img = Image.open(p)
                enhanced = self.process_batch([img], tile_size=tile_size, auto_downsample=auto_downsample)
                results.extend(enhanced)
            except Exception as e:
                print(f"Error processing {p}: {e}")
                results.append(None)
        return results
