import sys
import os
# Add parent directory to sys.path to enable absolute package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from debo_model.debo_preprocessing import preprocess_light_curve, align_sequence_length

def load_index_and_data(index_path="debo_model/dataset_index.csv", target_len=2000):
    """
    Loads dataset index, maps labels, reads .npz files, and preprocesses all curves.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Index file {index_path} not found. Please make sure dataset_fetcher.py is finished."
        )
        
    df = pd.read_csv(index_path)
    
    x_raw = []
    y_raw = []
    
    for idx, row in df.iterrows():
        filepath = row["file"]
        # Make path absolute/correct relative to root
        if not os.path.exists(filepath):
            # Try prepending current parent path if running from subfolder
            filepath = os.path.join("..", filepath)
            if not os.path.exists(filepath):
                continue
                
        # Load .npz file
        data = np.load(filepath)
        flux = data["flux"]
        
        # Preprocess and align
        p_flux = preprocess_light_curve(flux)
        aligned_flux = align_sequence_length(p_flux, target_len=target_len)
        
        # Map label: 'transit' is 1.0 (positive transit class), others are 0.0
        label = 1.0 if row["label"] == "transit" else 0.0
        
        x_raw.append(aligned_flux)
        y_raw.append(label)
        
    return np.array(x_raw, dtype=np.float32), np.array(y_raw, dtype=np.float32)

def prepare_splits(index_path="debo_model/dataset_index.csv", target_len=2000, test_size=0.15, val_size=0.15, seed=42):
    """
    Loads preprocessed datasets and returns stratified Train, Val, and Test splits.
    """
    X, y = load_index_and_data(index_path, target_len)
    
    if len(X) == 0:
        raise ValueError("Loaded 0 valid light curve samples. Check if downloads are complete.")
        
    # Stratified split: Train/Val and Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # Calculate relative validation size
    rel_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=rel_val_size, random_state=seed, stratify=y_train_val
    )
    
    # Standard reshape to include channel dimension for 1D CNN -> [B, N, 1]
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
