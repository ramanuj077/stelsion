import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from ramanuj_model.dataset import load_data_from_dir
from ramanuj_model.preprocessing import apply_preprocessing
from evaluation.metrics import calculate_metrics
import ramanuj_model.config as config

def evaluate_model(model_path, dataset_dir):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
        
    print(f"Loading model: {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Get model parameters
    num_params = model.count_params()
    print(f"Model loaded. Parameters count: {num_params:,}")
    
    # Inspect shape
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
        
    non_batch_dims = [d for d in input_shape if d is not None]
    total_elements = 1
    for d in non_batch_dims:
        total_elements *= d
        
    # Load dataset
    print(f"Loading validation/test dataset from {dataset_dir}...")
    x_val, y_val = load_data_from_dir(dataset_dir, config.PREPROCESSING, total_elements)
    
    if x_val is None or len(x_val) == 0:
        print(f"Error: No valid dataset files found in {dataset_dir}")
        return
        
    target_shape = [-1] + [d if d is not None else 1 for d in input_shape[1:]]
    x_val_reshaped = x_val.reshape(target_shape)
    
    print("Running inference...")
    y_prob = model.predict(x_val_reshaped).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    
    metrics = calculate_metrics(y_val, y_pred, y_prob)
    
    print("\n" + "=" * 50)
    print(f"EVALUATION ON {os.path.basename(dataset_dir).upper()} SET")
    print("=" * 50)
    print(f"Accuracy:                {metrics['accuracy']*100:.2f}%")
    print(f"Precision:               {metrics['precision']*100:.2f}%")
    print(f"Recall:                  {metrics['recall']*100:.2f}%")
    print(f"F1 Score:                {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:                 {metrics['roc_auc']:.4f}")
    print(f"False Positive Rate:     {metrics['false_positive_rate']*100:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a workspace model")
    parser.add_argument("model_path", type=str, help="Path to the model file (.h5)")
    parser.add_argument("dataset_dir", type=str, help="Directory containing dataset files")
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.dataset_dir)
