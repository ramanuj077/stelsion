# STELSION Ramanuj Personal Research Workspace

This is an isolated, highly modular workspace designed for rapid experimentation and model evaluation on exoplanet transit photometries.

## Workspace Structure

```text
ramanuj_model/
├── config.py           # Configuration files & hyperparameter registers
├── dataset.py          # Data ingestion and balancing splits
├── preprocessing.py    # Denoising and smoothing strategies (wavelet, sg, adaptive)
├── augmentation.py     # Data augmentation helpers (rolling, scaling, noise)
├── losses.py           # Custom BCE and Focal Loss modules
├── callbacks.py        # TensorBoard, checkpoints, early stopping array builders
├── utils.py            # Random seed registers and hardware configuration tools
├── trainer.py          # Standard training loops & Optuna HPO study trials
├── experiment.py       # Tracks version directories (experiments/001_baseline_combined, etc.)
├── train.py            # Training/HPO execution interface
├── evaluate.py         # Evaluation pipeline CLI
├── predict.py          # Direct inference CLI
│
├── architectures/      # Architecture registers
│   ├── baseline.py                 # Baseline production match CNN
│   ├── inceptiontime.py            # Parallel multi-scale convolutional module
│   ├── minor_axis_attention.py     # Channel and Spatial Attention CNN
│   ├── hybrid.py                   # Combined Baseline + InceptionTime + Attention
│   └── attention_blocks.py         # Squeeze-and-Excitation + Spatial Attention layers
│
├── saved_models/       # Keeps final trained model checkouts
└── experiments/        # Auto-versioned folders containing logs, histories, configs, and plots
```

---

## Roadmap

* **Phase 1**: Setup Workspace Environment.
* **Phase 2**: Reproduce the baseline CNN (`saved_models/secondary_model.h5`).
* **Phase 3**: Compare preprocessing pipelines (Wavelet vs Savitzky-Golay vs Adaptive).
* **Phase 4**: Train and evaluate the multi-scale **InceptionTime** model.
* **Phase 5**: Train and evaluate the **Minor-Axis Attention** model.
* **Phase 6**: Train and evaluate the **Hybrid** configuration.
* **Phase 7**: Optimize parameters using **Optuna** HPO.
* **Phase 8**: Build an ensemble of the best models.
* **Phase 9**: Rank all models against each other on the shared `evaluation/leaderboard.py`.
* **Phase 10**: Swap the production model with a challenger only if the benchmark shows improvements.

---

## Quick Start

### 1. Configure the Run
Open `ramanuj_model/config.py` and configure the settings:
- Toggle `ARCHITECTURE = "baseline"` (or `"inceptiontime"`, `"minor_axis_attention"`, `"hybrid"`).
- Select `PREPROCESSING = "combined"` (or `"baseline"`, `"wavelet"`, `"sg_filter"`, `"adaptive"`).

### 2. Standard Training Run
Start a standard training run (this automatically logs results and triggers auto-benchmarking if `datasets/test/` has data):
```bash
.\venv\Scripts\python.exe ramanuj_model/train.py
```

### 3. Hyperparameter Search (Optuna HPO)
Start a multi-trial Optuna study to optimize learning rate, optimizer, dropout, and batch size:
```bash
.\venv\Scripts\python.exe ramanuj_model/train.py --optuna
```

### 4. Evaluate a Model
```bash
.\venv\Scripts\python.exe ramanuj_model/evaluate.py ramanuj_model/saved_models/baseline_model.h5 datasets/validation/
```

### 5. Prediction
```bash
.\venv\Scripts\python.exe ramanuj_model/predict.py ramanuj_model/saved_models/baseline_model.h5 path/to/light_curve.csv
```
