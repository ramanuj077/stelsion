import numpy as np
import sys
import os

# Add parent directory to sys.path to import from global preprocessing package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.filters import (
    handle_missing_values,
    remove_outliers_sigma_clipping,
    remove_stellar_variability,
    normalize_flux,
    estimate_noise,
    savitzky_golay_filter,
    wavelet_denoising,
    hybrid_denoising
)

def preprocess_light_curve(flux):
    """
    Standard preprocessing pipeline sequence:
    1. Handle missing values (NaNs) via interpolation.
    2. Remove cosmic ray spikes and outliers via sigma clipping.
    3. Detrend the slow stellar variability using median filter detrending.
    4. Normalize the flux around 0 (fractional deviation).
    5. Apply adaptive high-frequency denoising based on estimated noise.
    """
    # 1. Gaps / NaNs
    flux = handle_missing_values(flux, method='interpolate')
    
    # 2. Outliers
    flux = remove_outliers_sigma_clipping(flux, sigma=3.0, iters=2)
    
    # 3. Normalization (Z-score scaling to mean=0, std=1.0 for gradient stability)
    flux = normalize_flux(flux, method='zscore')
    
    # 4. Stellar Detrending (Subtract stellar variability trend)
    flux = remove_stellar_variability(flux, window_size=101)
    
    # 5. Denoising
    noise = estimate_noise(flux)
    if noise < 0.005:
        # Low noise: Savitzky-Golay
        flux = savitzky_golay_filter(flux, window_size=15, polyorder=2)
    elif noise < 0.015:
        # Medium noise: Wavelet
        flux = wavelet_denoising(flux)
    else:
        # High noise: Hybrid detrending
        flux = hybrid_denoising(flux)
        
    return flux

def align_sequence_length(flux, target_len=2000):
    """
    Pads or truncates a 1D flux curve to a fixed target length.
    """
    curr_len = len(flux)
    if curr_len > target_len:
        # Crop symmetrically from center
        start = (curr_len - target_len) // 2
        return flux[start:start + target_len]
    elif curr_len < target_len:
        # Pad with edge values
        pad_width = target_len - curr_len
        return np.pad(flux, (0, pad_width), mode='edge')
    return flux
