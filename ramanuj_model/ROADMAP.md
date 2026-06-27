# STELSION Research Roadmap: Ramanuj Model

Detailed roadmap outlining development phases, architectural targets, and comparison steps.

## Phase Overview

### Phase 1: Research Workspace setup
* **Status**: Completed
* **Objective**: Standardize directories, configs, dataset loader, metrics, callbacks, and automatic benchmark logging.

### Phase 2: Reproduce Production Baseline
* **Status**: Active
* **Objective**: Replicate the production secondary model CNN architecture inside the isolated workspace and verify leaderboard output compatibility.

### Phase 3: Preprocessing Experiments
* **Status**: Planned
* **Objective**: Compare baseline median-normalized, wavelet-denoised, Savitzky-Golay smoothed, and adaptive configurations.

### Phase 4: InceptionTime Architecture
* **Status**: Planned
* **Objective**: Integrate multi-scale kernel lengths (e.g. 3, 5, 9) to capture ingress/egress transits over varying temporal horizons.

### Phase 5: Spatial & Channel Attention
* **Status**: Planned
* **Objective**: Implement minor-axis spatial self-attention blocks to weigh spatial transit dimensions.

### Phase 6: Hybrid Structures
* **Status**: Planned
* **Objective**: Combine Baseline + InceptionTime + Spatial Attention layers.

### Phase 7: Optuna HPO Search
* **Status**: Planned
* **Objective**: Run automated trials over learning rate, dropout, optimization method, and batch sizes.

### Phase 8: Model Ensembling
* **Status**: Planned
* **Objective**: Average predictions across top-ranked workspace configurations.
