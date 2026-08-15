import torch
import torch.nn.functional as F
import numpy as np


def normalize(tensor, scale, offset, norm_type):
    """
    Normalize DSM/XYZ tensor based on per-scene statistics."""
    def _reshape(v, t):
        if v is None: return None
        if isinstance(v, (torch.Tensor, np.ndarray)) and v.ndim == 1:
            if t.ndim >= 3 and t.shape[-3] == v.shape[0]:
                return v.view(-1, 1, 1) if torch.is_tensor(v) else v.reshape(-1, 1, 1)
            if t.ndim >= 2 and t.shape[-1] == v.shape[0]:
                return v
        return v

    off = _reshape(offset, tensor)
    scl = _reshape(scale, tensor)

    if norm_type == 'minmax_11':
        return 2 * (tensor - off) / scl - 1
    elif norm_type == 'minmax_01':
        return (tensor - off) / scl
    elif norm_type == 'mean_std':
        return (tensor - off) / scl
    return tensor


def denorm(v, dsm_scale, dsm_offset, norm_type):
    """
    Denormalize XYZ/DSM coordinates.
    v: tensor or numpy array (C, H, W) or (B, C, H, W)
    dsm_scale: scalar or (B,)
    dsm_offset: (C,) or (B, C)
    norm_type: str"""
    if v is None:
        return None
    
    # Handle tensor/numpy conversion
    is_tensor = isinstance(v, torch.Tensor)
    if is_tensor:
        v_np = v.detach().cpu().numpy()
        if isinstance(dsm_scale, torch.Tensor):
            s = dsm_scale.item() if dsm_scale.numel() == 1 else dsm_scale.detach().cpu().numpy()
        else:
            s = dsm_scale
        
        if isinstance(dsm_offset, torch.Tensor):
            o = dsm_offset.detach().cpu().numpy()
        else:
            o = dsm_offset
    else:
        v_np = v
        s = dsm_scale
        o = dsm_offset
        
    # Handle shapes for single sample (C, H, W)
    if v_np.ndim == 3:
        if isinstance(o, np.ndarray) and o.ndim == 1 and o.size == v_np.shape[0]:
            o = o.reshape(-1, 1, 1)
        if isinstance(s, np.ndarray) and s.ndim == 1 and s.size == v_np.shape[0]:
            s = s.reshape(-1, 1, 1)

    if norm_type == 'minmax_11':
        return (v_np + 1) / 2 * s + o
    if norm_type == 'minmax_01':
        return v_np * s + o
    if norm_type == 'mean_std':
        return v_np * s + o
    return v_np







def _torch_dict_to_numpy(d):
    """Convert all torch tensors in a dict to numpy for safe pickling."""
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = v.cpu().numpy()
        elif isinstance(v, dict):
            out[k] = _torch_dict_to_numpy(v)
        else:
            out[k] = v
    return out

def _numpy_dict_to_torch(d):
    """Convert numpy arrays back to torch tensors in a dict."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.dtype.kind == 'f':
            out[k] = torch.from_numpy(v)
        elif isinstance(v, dict):
            out[k] = _numpy_dict_to_torch(v)
        else:
            out[k] = v
    return out

def _move_to_device(views: list, device) -> list:



    for view in views:
        for k, v in view.items():
            if isinstance(v, torch.Tensor):
                view[k] = v.to(device)
    return views

