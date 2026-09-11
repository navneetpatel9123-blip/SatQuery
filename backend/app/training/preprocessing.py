"""
SatQuery AI — Remote Sensing Data Preprocessing
Standardized tensor normalization and band extraction for multi-spectral and optical imagery.
"""
from typing import List, Optional, Tuple, Union
import numpy as np

# BigEarthNet 12-band channel names in standard Sentinel-2 order
BIGEARTHNET_S2_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"
]

# Standard Sentinel-2 band statistics (mean, std) for reflectance scaling
S2_MEAN = [1353.72, 1117.20, 1041.82, 946.51, 1199.18, 2003.02, 2374.01, 2301.22, 2460.13, 718.89, 2095.93, 1414.94]
S2_STD = [65.47, 154.01, 187.99, 278.50, 228.12, 356.59, 456.03, 531.55, 496.06, 99.41, 450.88, 395.31]

def normalize_raster_bands(
    data: np.ndarray,
    target_size: Tuple[int, int] = (120, 120),
    max_val: float = 10000.0
) -> np.ndarray:
    """
    Normalize multi-spectral array of shape (C, H, W) to float32 [0.0, 1.0].
    Safely resizes or clips pixels.
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]
        
    c, h, w = data.shape
    normalized = np.clip(data.astype(np.float32) / max_val, 0.0, 1.0)
    
    # Simple nearest-neighbor spatial resize if needed
    if (h, w) != target_size:
        h_target, w_target = target_size
        row_indices = np.linspace(0, h - 1, h_target).astype(int)
        col_indices = np.linspace(0, w - 1, w_target).astype(int)
        normalized = normalized[:, row_indices[:, None], col_indices]
        
    return normalized

def extract_rgb_nir(data: np.ndarray, band_descriptions: Optional[List[str]] = None) -> np.ndarray:
    """
    Extract 4-channel RGB+NIR array from arbitrary band configuration.
    Defaults to first 4 channels or duplicates if fewer.
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]
        
    c, h, w = data.shape
    if c >= 4:
        return data[:4]
    elif c == 3:
        # Append mean of channels as pseudo-NIR
        nir = np.mean(data, axis=0, keepdims=True)
        return np.concatenate([data, nir], axis=0)
    elif c == 1:
        return np.repeat(data, 4, axis=0)
    else:
        # Fallback
        return np.zeros((4, h, w), dtype=data.dtype)
