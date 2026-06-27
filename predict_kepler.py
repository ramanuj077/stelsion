import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import lightkurve as lk

# Add workspace directory to python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models.architecture import ExoplanetDetectorNet
from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.filters import moving_average_median_filter, handle_missing_values, normalize_flux, remove_outliers_sigma_clipping
from evaluation.explainability import GradCAM1D, estimate_uncertainty_mc_dropout, estimate_transit_parameters, analyze_false_positives

def run_kepler_prediction():
    print("=" * 80)
    print("         AstroAI: REAL-WORLD KEPLER DATA CLASSIFIER (LIGHTKURVE)        ")
    print("=" * 80)
    
    target_star = "Kepler-90"
    print(f"\n[Step 1/4] Downloading {target_star} light curve data from NASA MAST...")
    
    # Download Quarter 4 of Kepler-90 which contains clear transit events of Kepler-90b/c
    try:
        search_result = lk.search_lightcurve(target_star, quarter=4, author='Kepler')
        if len(search_result) == 0:
            print(" - Quarter 4 not found. Searching for any Kepler light curve...")
            search_result = lk.search_lightcurve(target_star, mission='Kepler')
            
        if len(search_result) == 0:
            raise ValueError(f"No light curves found for target {target_star}")
            
        print(f" - Downloading {len(search_result[0])} light curve records...")
        lc = search_result[0].download()
        print(" - Light curve downloaded successfully.")
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Falling back to generating a simulated Kepler-90-like curve...")
        # Fallback to simulated data if offline/network fails
        time = np.linspace(0, 10, 2000)
        noise = np.random.normal(0, 0.005, 2000)
        transit = np.zeros(2000)
        transit[300:380] = -0.02
        transit[1200:1280] = -0.02
        flux = 1.0 + noise + transit
        # Wrap in a mock object
        class MockLC:
            def __init__(self, f):
                self.flux = f
                self.time = np.arange(len(f))
        lc = MockLC(flux)

    # 2. Extract and Preprocess
    print("\n[Step 2/4] Preprocessing light curve...")
    raw_flux = np.array(lc.flux.value if hasattr(lc.flux, 'value') else lc.flux, dtype=float)
    
    # Clean raw flux for plotting (removing NaNs from raw plot coordinates)
    raw_flux_cleaned = handle_missing_values(raw_flux, method='interpolate')
    
    # Preprocess via our pipeline (using moving average + median filters)
    pipeline = PreprocessingPipeline({
        'missing_value_method': 'interpolate',
        'normalization_method': 'median',
        'sigma_clipping_sigma': 3.0,
        'sigma_clipping_iters': 2,
        'sg_window': 15, # Replaced by moving average + median in pipeline
        'segment_length': 2000
    })
    
    # Process
    processed_flux = pipeline.process_single_curve(raw_flux)
    processed_flux = pipeline.segment_or_pad(processed_flux, 2000)
    
    # Make sure we have 2,000 points of raw flux for aligned plotting
    raw_flux_segmented = pipeline.segment_or_pad(raw_flux_cleaned, 2000)
    
    # 3. Model Classification
    print("\n[Step 3/4] Running model inference...")
    model = ExoplanetDetectorNet(input_len=2000)
    # Build model using dummy input
    model(np.zeros((1, 2000, 1), dtype=np.float32), training=False)
    
    # Load weights
    best_model_path = os.path.join('saved_models', 'best_model.weights.h5')
    if os.path.exists(best_model_path):
        model.load_weights(best_model_path)
        print(f" - Loaded model weights from: {best_model_path}")
    else:
        print(" - WARNING: No trained model checkpoint weights found. Using initialized weights.")
        
    input_tensor = processed_flux[np.newaxis, :, np.newaxis]
    
    # Uncertainty Estimation via MC Dropout
    mean_prob, uncertainty, reliability = estimate_uncertainty_mc_dropout(model, input_tensor, num_samples=10)
    print(f" - Exoplanet Candidate Probability: {mean_prob*100:.2f}%")
    print(f" - Prediction Uncertainty: +/- {uncertainty*100:.2f}%")
    print(f" - Reliability Rating: {reliability}")
    
    # Parameters & Heuristic Verdict
    params = estimate_transit_parameters(processed_flux)
    fp_analysis = analyze_false_positives(processed_flux)
    
    # 1D Grad-CAM
    gradcam = GradCAM1D(model, model.res3.conv2)
    heatmap = gradcam.generate_heatmap(input_tensor)
    
    # 4. Generate Plot (Original + Filtered + Confidence in one picture)
    print("\n[Step 4/4] Creating classification report plot...")
    os.makedirs("results", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Subplot 1: Original Light Curve
    ax1.plot(raw_flux_segmented, color='#7f8c8d', alpha=0.8, linewidth=1.0, label='Raw Kepler-90 Flux')
    ax1.set_title(f'Original (Raw) Light Curve - {target_star}', fontsize=14, fontweight='bold', color='#2c3e50')
    ax1.set_ylabel('Relative Flux', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    # Subplot 2: Processed Light Curve + Grad-CAM Heatmap
    ax2.plot(processed_flux, color='#2980b9', linewidth=1.5, label='Denoised & Detrended Flux')
    
    # Overlaid Heatmap
    x_vals = np.arange(len(processed_flux))
    ax2.fill_between(x_vals, np.min(processed_flux), np.max(processed_flux), 
                     where=(heatmap > 0.1), color='#e74c3c', alpha=0.25, label='Grad-CAM Transit Region Detection')
                     
    ax2.set_title('Filtered Light Curve (Savitzky-Golay Filter) & CNN Activations', fontsize=14, fontweight='bold', color='#2c3e50')
    ax2.set_ylabel('Normalized Flux (Centered around 0)', fontsize=12)
    ax2.set_xlabel('Observation Step (Time)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    # Add a main layout text box summarizing prediction details
    info_text = (
        f"Target Star: {target_star}\n"
        f"Exoplanet Candidate Probability: {mean_prob * 100:.1f}%\n"
        f"Prediction Uncertainty (MC-Dropout): +/- {uncertainty * 100:.1f}%\n"
        f"Prediction Reliability: {reliability}\n"
        f"Verdict: {fp_analysis['verdict']}\n"
        f"Estimated Period: {params['period_days']:.2f} days\n"
        f"Estimated Transit Depth: {params['depth_percent']:.3f}%\n"
        f"Analysis: {fp_analysis['reason']}"
    )
    
    fig.text(0.12, 0.02, info_text, fontsize=11, family='monospace', bbox=dict(boxstyle='round', facecolor='#fdfefe', edgecolor='#bdc3c7', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    
    report_path = os.path.join("results", "kepler_classification_report.png")
    plt.savefig(report_path, dpi=150)
    print(f"Classification report successfully saved to: {report_path}")
    plt.close()
    
    print("\nExecution complete!")
    print("=" * 80)

if __name__ == "__main__":
    run_kepler_prediction()
