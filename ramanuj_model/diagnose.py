import os
import sys

# Prioritize workspace root and remove local dir to prevent module shadowing
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)
if workspace_root in sys.path:
    sys.path.remove(workspace_root)
sys.path.insert(0, workspace_root)

# Ensure workspace root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

import ramanuj_model.config as config
from ramanuj_model.dataset import prepare_datasets

def diagnose():
    print("=" * 60)
    print("RUNNING MACHINE LEARNING DIAGNOSIS")
    print("=" * 60)
    
    # 1. Load data splits
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_datasets()
    
    # 2. Check distributions
    print(f"Train class count:   0.0: {np.sum(y_train == 0.0)}, 1.0: {np.sum(y_train == 1.0)}")
    print(f"Val class count:     0.0: {np.sum(y_val == 0.0)}, 1.0: {np.sum(y_val == 1.0)}")
    print(f"Test class count:    0.0: {np.sum(y_test == 0.0)}, 1.0: {np.sum(y_test == 1.0)}")
    
    # 3. Load the trained model
    model_path = os.path.join(config.SAVED_MODELS_DIR, f"{config.ARCHITECTURE}_model.h5")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = tf.keras.models.load_model(model_path)
    
    # 4. Check model output layer
    print(f"Model Output Shape: {model.output_shape}")
    
    # 5. Check predictions on training set
    y_train_prob = model.predict(x_train).flatten()
    y_train_pred = (y_train_prob >= 0.5).astype(int)
    
    print("\nTrain Prediction Probabilities (First 15):")
    for i in range(min(15, len(y_train))):
        print(f"  Sample {i+1:02d}: True Label={y_train[i]}, Prob={y_train_prob[i]:.6f}, Pred={y_train_pred[i]}")
        
    # 6. Check predictions on test set
    y_prob = model.predict(x_test).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    
    print("\nTest Prediction Probabilities (Raw):")
    for i, p in enumerate(y_prob):
        print(f"  Sample {i+1:02d}: True Label={y_test[i]}, Prob={p:.6f}, Pred={y_pred[i]}")
        
    # 6. Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("\nConfusion Matrix:")
    print(cm)
    print("=" * 60)

if __name__ == "__main__":
    diagnose()
