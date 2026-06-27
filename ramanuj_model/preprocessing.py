import numpy as np
from preprocessing.filters import (
    handle_missing_values, normalize_flux, remove_outliers_sigma_clipping,
    wavelet_denoising, savitzky_golay_filter, median_filter,
    remove_stellar_variability, estimate_noise, hybrid_denoising
)

def apply_preprocessing(flux, pipeline_name="combined"):
    """
    Applies the chosen preprocessing pipeline strategy to a single raw flux curve.
    Reuses existing filter implementations from the production directory.
    """
    flux = np.array(flux, dtype=float)
    flux = handle_missing_values(flux, method='interpolate')
    
    if pipeline_name == "baseline":
        # Raw median normalization (maintaining original scale trend)
        flux = remove_outliers_sigma_clipping(flux, sigma=3.0, iters=2)
        flux = normalize_flux(flux, method='median')
        
    elif pipeline_name == "wavelet":
        # Simple wavelet denoising only
        flux = remove_outliers_sigma_clipping(flux, sigma=3.0, iters=2)
        flux = wavelet_denoising(flux)
        flux = normalize_flux(flux, method='median')
        
    elif pipeline_name == "sg_filter":
        # Savitzky-Golay filtering only
        flux = remove_outliers_sigma_clipping(flux, sigma=3.0, iters=2)
        flux = savitzky_golay_filter(flux, window_size=15, polyorder=2)
        flux = normalize_flux(flux, method='median')
        
    elif pipeline_name == "adaptive":
        # Adaptive switching based on calculated noise thresholds
        flux = remove_outliers_sigma_clipping(flux, sigma=3.0, iters=2)
        noise = estimate_noise(flux)
        if noise < 0.005:
            flux = savitzky_golay_filter(flux, window_size=15, polyorder=2)
        else:
            flux = wavelet_denoising(flux)
        flux = normalize_flux(flux, method='median')
        
    elif pipeline_name == "combined":
        # Standard combined production processing
        flux = remove_outliers_sigma_clipping(flux, sigma=3.0, iters=2)
        flux = remove_stellar_variability(flux, window_size=101)
        noise = estimate_noise(flux)
        if noise < 0.005:
            flux = savitzky_golay_filter(flux, window_size=15, polyorder=2)
        elif noise < 0.015:
            flux = wavelet_denoising(flux)
        else:
            flux = hybrid_denoising(flux)
            
        # Safe normalization to prevent division by near-zero values on flattened curves
        median_val = np.median(flux)
        if abs(median_val) > 0.001:
            flux = flux / median_val - 1.0
            
    return flux
