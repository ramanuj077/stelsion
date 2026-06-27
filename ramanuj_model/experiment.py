import os
import json
import csv
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import ramanuj_model.config as config

def create_experiment_dir():
    """
    Finds the next incremental run number (e.g. 001_baseline_combined)
    and initializes its folder structure under ramanuj_model/experiments/
    """
    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    existing_dirs = [
        d for d in os.listdir(config.EXPERIMENTS_DIR) 
        if os.path.isdir(os.path.join(config.EXPERIMENTS_DIR, d))
    ]
    
    next_idx = 1
    for d in existing_dirs:
        try:
            parts = d.split("_")
            idx = int(parts[0])
            if idx >= next_idx:
                next_idx = idx + 1
        except ValueError:
            pass
            
    dir_name = f"{next_idx:03d}_{config.ARCHITECTURE}_{config.PREPROCESSING}"
    exp_dir = os.path.join(config.EXPERIMENTS_DIR, dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    return exp_dir

def save_experiment_results(exp_dir, model, history, val_metrics, config_overrides=None, y_val_true=None, y_val_prob=None):
    """
    Persists configuration parameters, metrics, logs, plots, and models to the run folder.
    """
    # 1. Save model
    model.save(os.path.join(exp_dir, "model.h5"))
    
    # 2. Save configuration parameters
    cfg_dict = {
        "ARCHITECTURE": config.ARCHITECTURE,
        "PREPROCESSING": config.PREPROCESSING,
        "INPUT_SHAPE": config.INPUT_SHAPE,
        "BATCH_SIZE": config.BATCH_SIZE,
        "EPOCHS": config.EPOCHS,
        "LEARNING_RATE": config.LEARNING_RATE,
        "LOSS": config.LOSS,
        "OPTIMIZER": config.OPTIMIZER,
        "WEIGHT_DECAY": config.WEIGHT_DECAY,
        "SEED": config.SEED
    }
    if config_overrides:
        cfg_dict.update(config_overrides)
        
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=4)
        
    # 3. Save validation metrics
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=4)
        
    # 4. Save history as CSV
    if hasattr(history, 'history'):
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(os.path.join(exp_dir, "history.csv"), index=False)
        
    # 5. Generate and save plots if validation data is provided
    if y_val_true is not None and y_val_prob is not None:
        # Confusion Matrix
        y_val_pred = (y_val_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_val_true, y_val_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Planet", "Candidate"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Validation Confusion Matrix")
        plt.savefig(os.path.join(exp_dir, "plots", "confusion_matrix.png"), dpi=150)
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_val_true, y_val_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC (AUC = {val_metrics.get('roc_auc', 0.5):.4f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Validation ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(exp_dir, "plots", "roc_curve.png"), dpi=150)
        plt.close()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_val_true, y_val_prob)
        plt.figure()
        plt.plot(recall, precision, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Validation Precision-Recall Curve')
        plt.savefig(os.path.join(exp_dir, "plots", "precision_recall_curve.png"), dpi=150)
        plt.close()
        
    # 6. Save reproducibility notes
    with open(os.path.join(exp_dir, "notes.md"), "w") as f:
        f.write(f"# Experiment Run {os.path.basename(exp_dir)}\n\n")
        f.write(f"- **Architecture**: {cfg_dict['ARCHITECTURE']}\n")
        f.write(f"- **Preprocessing**: {cfg_dict['PREPROCESSING']}\n")
        f.write(f"- **Total Parameters**: {model.count_params():,}\n")
        f.write(f"- **Learning Rate**: {cfg_dict['LEARNING_RATE']}\n")
        f.write(f"- **Batch Size**: {cfg_dict['BATCH_SIZE']}\n")
        f.write(f"- **Epochs Run**: {cfg_dict['EPOCHS']}\n\n")
        f.write("## Validation Metrics Summary\n")
        for k, v in val_metrics.items():
            if isinstance(v, float):
                f.write(f"- **{k}**: {v:.4f}\n")
            else:
                f.write(f"- **{k}**: {v}\n")
        f.write("\n## Observations\n")
        f.write("Training concluded successfully. Write observations here.\n")
        
    print(f"All experiment files saved successfully to: {exp_dir}")
