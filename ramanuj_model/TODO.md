# Workspace Todo & Action Items

## Active Task List

### Ingestion & Data Setup
- [ ] Place training light curves in `datasets/train/`
- [ ] Place validation light curves in `datasets/validation/`
- [ ] Place test light curves in `datasets/test/`

### Baseline Training
- [ ] Run `python ramanuj_model/train.py` using `baseline` architecture and `combined` preprocessing
- [ ] Verify that model weights, configurations, and metric logs are successfully saved in `experiments/001_.../`
- [ ] Verify that the run is logged on the shared leaderboard in `experiments/leaderboard.csv`

### Architectural Iterations
- [ ] Switch config to `inceptiontime` and train
- [ ] Switch config to `minor_axis_attention` and train
- [ ] Switch config to `hybrid` and train

### Hyperparameter Search
- [ ] Run `python ramanuj_model/train.py --optuna` to tune baseline parameters
- [ ] Incorporate optimal parameters back into `config.py`
