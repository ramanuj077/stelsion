import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")

# General settings
SEED = 42
TF_GPU_MEMORY_GROWTH = True

# Dataset settings
# Selected preprocessing pipeline: "baseline", "wavelet", "sg_filter", "adaptive", "combined"
PREPROCESSING = "combined"
INPUT_SHAPE = (9, 800, 1)
SEGMENT_LENGTH = 7200  # product of input shape dimensions

# Augmentation settings
ENABLE_AUGMENTATION = True
ROLL_FRACTION = 0.25
NOISE_STD = 0.01
SCALE_MIN = 0.7
SCALE_MAX = 1.3

# Model training settings
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.001
OPTIMIZER = "adam"       # "adam", "sgd", "rmsprop"
LOSS = "bce"             # "bce", "focal_loss"
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3

# Active architecture selection
# Options: "baseline", "inceptiontime", "minor_axis_attention", "hybrid"
ARCHITECTURE = "baseline"

# Hyperparameter optimization (Optuna)
OPTUNA_TRIALS = 15
