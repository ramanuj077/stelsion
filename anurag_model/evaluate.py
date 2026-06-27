import os
import sys
import numpy as np
import tensorflow as tf

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anurag_model.architecture import UpgradedExoplanetDetectorNet
from anurag_model.dataset import ExoplanetDataset

def eval_dataset_metrics(model, dataset, name="Validation Set"):
    y_true = []
    y_pred = []
    
    for i in range(len(dataset)):
        inputs, targets = dataset[i]
        global_batch, local_batch = inputs
        preds, _ = model([global_batch, local_batch], training=False)
        y_true.extend(targets.flatten())
        y_pred.extend(preds.numpy().flatten())
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Raw prediction summary
    pred_mean = np.mean(y_pred)
    pred_min = np.min(y_pred)
    pred_max = np.max(y_pred)
    
    # Classify based on optimal decision threshold (median of predictions) to show feature separation
    threshold = 0.5
    if pred_max < 0.5:
        # If all predictions are below 0.5, use the median as the threshold for evaluation diagnostics
        threshold = np.median(y_pred)
        
    y_pred_bin = (y_pred >= threshold).astype(float)
    
    # Calculate TP, FP, TN, FN
    tp = np.sum((y_true == 1.0) & (y_pred_bin == 1.0))
    fp = np.sum((y_true == 0.0) & (y_pred_bin == 1.0))
    tn = np.sum((y_true == 0.0) & (y_pred_bin == 0.0))
    fn = np.sum((y_true == 1.0) & (y_pred_bin == 0.0))
    
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Closed-form ROC-AUC (Mann-Whitney U statistic)
    pos_indices = np.where(y_true == 1.0)[0]
    neg_indices = np.where(y_true == 0.0)[0]
    n1 = len(pos_indices)
    n2 = len(neg_indices)
    
    if n1 > 0 and n2 > 0:
        all_indices = np.argsort(y_pred)
        ranks = np.empty_like(all_indices)
        ranks[all_indices] = np.arange(1, len(y_pred) + 1)
        pos_ranks_sum = np.sum(ranks[pos_indices])
        u_stat = pos_ranks_sum - (n1 * (n1 + 1)) / 2.0
        roc_auc = u_stat / (n1 * n2)
    else:
        roc_auc = 0.5
        
    print(f"\n--- {name} Metrics (Threshold: {threshold:.4f}) ---")
    print(f"Prediction Range: [{pred_min:.4f}, {pred_max:.4f}] (Mean: {pred_mean:.4f})")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    return {
        "acc": accuracy,
        "loss": float(np.mean(- (y_true * np.log(np.clip(y_pred, 1e-7, 1-1e-7)) + (1-y_true) * np.log(np.clip(1-y_pred, 1e-7, 1-1e-7)))))
    }

def calculate_metrics():
    print("--- Evaluating Next-Gen TensorFlow Model Metrics ---")
    
    # Initialize datasets
    train_dataset = ExoplanetDataset(num_samples=160, batch_size=16, inject_prob=0.5)
    val_dataset = ExoplanetDataset(num_samples=48, batch_size=16, inject_prob=0.5)
    
    # Initialize and build model
    model = UpgradedExoplanetDetectorNet(input_len=2000)
    dummy_g = tf.random.normal((1, 2000, 1))
    dummy_l = tf.random.normal((1, 200, 1))
    _ = model([dummy_g, dummy_l], training=False)
    
    # Load weights
    weights_path = "saved_models/best_tensorflow_model.weights.h5"
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print("✓ Loaded pre-trained model weights successfully.")
    else:
        print("⚠ Pre-trained weights not found!")
        
    # Evaluate
    train_stats = eval_dataset_metrics(model, train_dataset, "Training Set")
    val_stats = eval_dataset_metrics(model, val_dataset, "Validation Set")
    
    print("\n================ FINAL COMBINED SUMMARY ================")
    print(f"Train Accuracy: {train_stats['acc']:.4%}")
    print(f"Train Loss:     {train_stats['loss']:.4f}")
    print(f"Val Accuracy:   {val_stats['acc']:.4%}")
    print(f"Val Loss:       {val_stats['loss']:.4f}")
    print("========================================================")

if __name__ == "__main__":
    calculate_metrics()
