import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.timeseries import BoxLeastSquares

# Add parent directory to path to import preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from debo_model.debo_preprocessing import preprocess_light_curve

def run_bls_detection(time, flux, min_period=0.5, max_period=15.0, points_per_grid=5000):
    """
    Applies the Box Least Squares (BLS) algorithm to detect periodic transits.
    Returns the periodogram results and the best fit parameters.
    """
    # Create astropy units for time-series search
    t = time * u.day
    y = flux
    
    # Initialize BLS model
    bls = BoxLeastSquares(t, y)
    
    # Create a grid of trial periods
    periods = np.linspace(min_period, max_period, points_per_grid) * u.day
    
    # Run periodogram search (assuming transit durations between 0.05 and 0.3 days)
    durations = np.linspace(0.05, 0.3, 5) * u.day
    results = bls.power(periods, durations)
    
    # Extract best fit parameters
    best_idx = np.argmax(results.power)
    best_period = results.period[best_idx].value
    best_t0 = results.transit_time[best_idx].value
    best_duration = results.duration[best_idx].value
    best_depth = results.depth[best_idx]
    
    # Calculate Signal-to-Noise Ratio (SNR) of the peak
    # Peak power is compared to standard deviation of power grid
    std_power = np.std(results.power)
    mean_power = np.mean(results.power)
    snr = (results.power[best_idx] - mean_power) / std_power if std_power > 0 else 0
    
    return results, {
        "period": best_period,
        "t0": best_t0,
        "duration": best_duration,
        "depth": best_depth,
        "snr": snr
    }

def plot_bls_results(time, flux, results, params, kepid, save_dir="debo_model/plots"):
    """
    Plots the raw/clean periodogram and the phase-folded light curve.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: BLS Periodogram (Power vs. Period)
    axes[0].plot(results.period.value, results.power, color="purple", alpha=0.8, lw=1.5)
    axes[0].axvline(params["period"], color="crimson", linestyle="--", alpha=0.8, 
                    label=f"Peak Period: {params['period']:.4f} days")
    axes[0].set_title(f"KIC {kepid} - BLS Periodogram", fontsize=14)
    axes[0].set_xlabel("Trial Period (Days)", fontsize=12)
    axes[0].set_ylabel("BLS Power", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Plot 2: Phase-Folded Light Curve
    # Calculate phase-folded time coordinates centered at 0
    period = params["period"]
    t0 = params["t0"]
    folded_phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    
    # Sort points by phase for a clean line plot
    sort_idx = np.argsort(folded_phase)
    phase_sorted = folded_phase[sort_idx]
    flux_sorted = flux[sort_idx]
    
    axes[1].scatter(phase_sorted, flux_sorted, color="dodgerblue", s=5, alpha=0.5, label="Data points")
    
    # Draw transit model box
    box_phase = params["duration"] / (2.0 * period)
    axes[1].axvline(-box_phase, color="darkorange", linestyle=":", alpha=0.8)
    axes[1].axvline(box_phase, color="darkorange", linestyle=":", alpha=0.8, label="Transit Window")
    
    axes[1].set_title(f"Phase-Folded Light Curve (Period = {params['period']:.4f} days)", fontsize=14)
    axes[1].set_xlabel("Phase", fontsize=12)
    axes[1].set_ylabel("Relative Flux", fontsize=12)
    axes[1].set_xlim(-0.5, 0.5)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{kepid}_bls_detection.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Diagnostic plots saved to: {plot_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python debo_model/candidate_detection.py <kepid>")
        print("Example: python debo_model/candidate_detection.py 10797460")
        return
        
    kepid = sys.argv[1]
    filepath = f"debo_model/datasets/{kepid}.npz"
    
    if not os.path.exists(filepath):
        print(f"Error: Dataset file {filepath} not found.")
        return
        
    print(f"Loading KIC {kepid} light curve...")
    data = np.load(filepath)
    time = data["time"]
    raw_flux = data["flux"]
    
    print("Preprocessing light curve...")
    clean_flux = preprocess_light_curve(raw_flux)
    
    print("Running BLS Search (this may take a few seconds)...")
    results, params = run_bls_detection(time, clean_flux)
    
    print("\n" + "=" * 50)
    print(f"          BLS CANDIDATE DETECTION RESULTS (KIC {kepid})          ")
    print("=" * 50)
    print(f"Detected Period:    {params['period']:.5f} days")
    print(f"Transit Epoch (t0): {params['t0']:.5f} (BKJD)")
    print(f"Transit Duration:   {params['duration']*24:.3f} hours ({params['duration']:.4f} days)")
    print(f"Transit Depth:      {params['depth']*100:.4f}%")
    print(f"Signal-to-Noise:    {params['snr']:.2f}")
    print("=" * 50)
    
    # Generate diagnostic plots
    plot_bls_results(time, clean_flux, results, params, kepid)

if __name__ == "__main__":
    main()
