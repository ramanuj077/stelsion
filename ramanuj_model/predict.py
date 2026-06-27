import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from ramanuj_model.preprocessing import apply_preprocessing
import ramanuj_model.config as config

def predict_single(model_path, flux_values, name="Target"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
        
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess
    p_flux = apply_preprocessing(flux_values, pipeline_name=config.PREPROCESSING)
    
    # Get shape details
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    non_batch_dims = [d for d in input_shape if d is not None]
    total_elements = 1
    for d in non_batch_dims:
        total_elements *= d
        
    # Segment or pad
    curr_len = len(p_flux)
    if curr_len > total_elements:
        start = (curr_len - total_elements) // 2
        p_flux = p_flux[start:start+total_elements]
    elif curr_len < total_elements:
        p_flux = np.pad(p_flux, (0, total_elements - curr_len), mode='edge')
        
    # Reshape and Predict
    target_shape = [1] + [d if d is not None else 1 for d in input_shape[1:]]
    tensor = np.array(p_flux, dtype=np.float32).reshape(target_shape)
    
    prob = float(model.predict(tensor)[0][0])
    verdict = "Exoplanet Candidate" if prob >= 0.5 else "Rejected"
    
    print("\n" + "=" * 50)
    print(f"PREDICTION FOR {name}")
    print("=" * 50)
    print(f"Probability: {prob*100:.4f}%")
    print(f"Verdict:     {verdict}")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction on light curve data")
    parser.add_argument("model_path", type=str, help="Path to Keras model file (.h5)")
    parser.add_argument("csv_path", type=str, help="Path to CSV containing flux values")
    args = parser.parse_args()
    
    # Read CSV
    try:
        flux = np.genfromtxt(args.csv_path, delimiter=",", skip_header=0)
        # Flatten if 2D
        if len(flux.shape) > 1:
            flux = flux.flatten()
        # Remove NaNs
        flux = flux[~np.isnan(flux)]
        predict_single(args.model_path, flux, name=os.path.basename(args.csv_path))
    except Exception as e:
        print(f"Error loading CSV data: {e}", file=sys.stderr)
