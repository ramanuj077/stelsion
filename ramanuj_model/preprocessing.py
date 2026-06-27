import numpy as np
import scipy.signal as signal
from preprocessing.filters import (
    handle_missing_values, normalize_flux, remove_outliers_sigma_clipping,
    wavelet_denoising, savitzky_golay_filter, median_filter,
    remove_stellar_variability, estimate_noise
)
import ramanuj_model.config as config

def apply_preprocessing_pipeline(flux: np.ndarray) -> np.ndarray:
    """
    Applies the Research Model V2 preprocessing pipeline stage-by-stage.
    Each stage is independently switchable via config flags.
    """
    flux = np.array(flux, dtype=float)
    
    # 1. NaN interpolation
    if config.STAGE_NAN_INTERP:
        flux = handle_missing_values(flux, method='interpolate')
        
    # 2. 3-sigma clipping
    if config.STAGE_SIGMA_CLIP:
        flux = remove_outliers_sigma_clipping(flux, sigma=3.0, iters=2)
        
    # 3. Noise estimation & Adaptive Filtering
    if config.STAGE_ADAPTIVE_FILTER:
        noise = estimate_noise(flux) if config.STAGE_NOISE_EST else 0.01
        
        if noise < 0.005:
            # Low Noise -> Savitzky-Golay
            flux = savitzky_golay_filter(flux, window_size=15, polyorder=2)
        elif noise < 0.015:
            # Medium Noise -> Wavelet Denoising
            flux = wavelet_denoising(flux, wavelet='db4', level=2)
        else:
            # High Noise -> Wavelet + Median Filter
            flux = wavelet_denoising(flux, wavelet='db4', level=2)
            flux = median_filter(flux, kernel_size=5)
            
    # 4. Stellar Trend Removal (Detrending)
    if config.STAGE_DETRENDING:
        flux = remove_stellar_variability(flux, window_size=101)
        
    # 5. Median Normalization
    if config.STAGE_NORMALIZATION:
        median_val = np.median(flux)
        if abs(median_val) > 0.001:
            flux = flux / median_val - 1.0
            
        std_val = np.std(flux)
        if std_val > 1e-6:
            flux = flux / std_val
            
    return flux
