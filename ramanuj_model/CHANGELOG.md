# Changelog: Ramanuj Model Workspace

All notable changes to the personal research workspace will be documented in this file.

## [1.0.0] - 2026-06-28

### Added
- Modular workspace structure under `ramanuj_model/`.
- Dynamic model factory pattern under `model.py` and `architectures/`.
- Preprocessing wrapper pipeline supporting baseline, wavelet, Savitzky-Golay, and adaptive strategies.
- NumPy-based augmentations (shift, scale, noise).
- Early stopping, learning rate scheduler, checkpoint, and TensorBoard logging callbacks.
- Focal loss and BCE loss function configurations.
- Dynamic data loaders supporting JSON, NumPy, and CSV.
- Automated evaluation and inference CLI tools (`evaluate.py`, `predict.py`).
- Auto-benchmarking integration triggering `evaluation/benchmark.py` post-training.
- Incremental experiment directory compiler (`experiment.py`).
- Optuna hyperparameter optimization study pipeline (`trainer.py`).
- Documentation files (`README.md`, `ROADMAP.md`).
