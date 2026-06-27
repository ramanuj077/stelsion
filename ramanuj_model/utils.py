import os
import random
import sys
import subprocess
import numpy as np
import tensorflow as tf

def set_seed(seed=42):
    """Sets standard seeds to ensure reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # TF reproducibility flag (if available)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def setup_gpu():
    """Configures TensorFlow to dynamically allocate memory on the GPU."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs configured with dynamic memory growth: {len(gpus)}")
    except Exception as e:
        print(f"Warning during GPU configuration: {e}")

def get_git_info():
    """Extracts git commit hash and user config info."""
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
    except Exception:
        git_commit = "N/A"
        
    try:
        git_user = subprocess.check_output(["git", "config", "user.name"]).decode("utf-8").strip()
    except Exception:
        git_user = "Unknown"
        
    return git_commit, git_user

def get_system_metadata():
    """Gathers system metadata for logging."""
    git_commit, git_user = get_git_info()
    return {
        "python_version": sys.version.split()[0],
        "tensorflow_version": tf.__version__,
        "git_commit": git_commit,
        "developer": git_user
    }
