import numpy as np
import scipy.signal as signal
try:
    import pywt
except ImportError:
    pywt = None

def handle_missing_values(flux, method='interpolate'):
    """
    Handles missing values (NaNs) in the light curve.
    """
    flux = np.array(flux, dtype=float)
    nans = np.isnan(flux)
    if not np.any(nans):
        return flux
    
    if method == 'interpolate':
        x = np.arange(len(flux))
        flux[nans] = np.interp(x[nans], x[~nans], flux[~nans])
    elif method == 'median':
        median_val = np.nanmedian(flux)
        flux[nans] = median_val
    elif method == 'zero':
        flux[nans] = 0.0
    return flux

def normalize_flux(flux, method='median'):
    """
    Normalizes the flux values of the light curve.
    - 'median': divide by median (relative flux around 1.0)
    - 'minmax': scale between 0 and 1
    - 'zscore': subtract mean and divide by std
    """
    if method == 'median':
        median_val = np.median(flux)
        if median_val == 0:
            return flux
        return flux / median_val - 1.0  # Center around 0
    elif method == 'minmax':
        min_val = np.min(flux)
        max_val = np.max(flux)
        if max_val - min_val == 0:
            return flux
        return (flux - min_val) / (max_val - min_val)
    elif method == 'zscore':
        std_val = np.std(flux)
        if std_val == 0:
            return flux
        return (flux - np.mean(flux)) / std_val
    return flux

def remove_outliers_sigma_clipping(flux, sigma=3.0, iters=2):
    """
    Applies sigma clipping to remove extreme outliers.
    """
    clipped_flux = flux.copy()
    for _ in range(iters):
        mean = np.mean(clipped_flux)
        std = np.std(clipped_flux)
        if std == 0:
            break
        bad_idx = np.abs(clipped_flux - mean) > sigma * std
        clipped_flux[bad_idx] = mean  # Replace with mean
    return clipped_flux

def wavelet_denoising(flux, wavelet='db4', level=2):
    """
    Performs discrete wavelet transform denoising.
    """
    if pywt is None:
        # Fallback to moving average if pywt is not installed
        return moving_average(flux, window_size=5)
    
    coeffs = pywt.wavedec(flux, wavelet, mode='per')
    # Soft thresholding
    sigma = (1/0.6745) * np.median(np.abs(coeffs[-1] - np.median(coeffs[-1])))
    threshold = sigma * np.sqrt(2 * np.log(len(flux)))
    
    new_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
        
    return pywt.waverec(new_coeffs, wavelet, mode='per')[:len(flux)]

def savitzky_golay_filter(flux, window_size=15, polyorder=2):
    """
    Applies Savitzky-Golay filter to smooth the light curve.
    """
    if window_size % 2 == 0:
        window_size += 1
    if len(flux) <= window_size:
        return flux
    return signal.savgol_filter(flux, window_size, polyorder)

def median_filter(flux, kernel_size=5):
    """
    Applies a median filter.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    if len(flux) <= kernel_size:
        return flux
    return signal.medfilt(flux, kernel_size)

def moving_average(flux, window_size=5):
    """
    Applies moving average filter.
    """
    if window_size <= 1:
        return flux
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(flux, window, 'same')

def remove_stellar_variability(flux, window_size=101):
    """
    Removes long-term stellar variability by dividing/subtracting a heavily median-filtered curve.
    """
    if window_size % 2 == 0:
        window_size += 1
    trend = signal.medfilt(flux, window_size)
    # Subtract trend to flatten
    return flux - trend

def estimate_noise(flux):
    """
    Estimates the noise level using standard deviation of differences.
    """
    diff = np.diff(flux)
    return float(np.std(diff) / np.sqrt(2))

def hybrid_denoising(flux, wavelet='db4', level=2, sg_window=15, sg_polyorder=2, median_kernel=5, sigma=3.0, iters=2):
    """
    Applies hybrid wavelet, median filtering, and sigma clipping.
    """
    # 1. Wavelet Denoising
    flux_denoised = wavelet_denoising(flux, wavelet=wavelet, level=level)
    # 2. Median Filter to suppress impulsive spikes
    flux_med = median_filter(flux_denoised, kernel_size=median_kernel)
    # 3. Sigma clipping outliers
    flux_clipped = remove_outliers_sigma_clipping(flux_med, sigma=sigma, iters=iters)
    return flux_clipped

def moving_average_median_filter(flux, window_size=15):
    """
    Smooths the light curve by applying a median filter followed by a moving average filter.
    """
    # 1. Median filter
    flux_med = median_filter(flux, kernel_size=window_size)
    # 2. Moving average filter
    flux_smoothed = moving_average(flux_med, window_size=window_size)
    return flux_smoothed


