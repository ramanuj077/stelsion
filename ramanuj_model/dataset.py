import os
import json
import numpy as np
from ramanuj_model.preprocessing import apply_preprocessing
from ramanuj_model.augmentation import augment_light_curve
import ramanuj_model.config as config

def load_data_from_dir(directory: str, preprocessing_name: str = "combined", segment_len: int = 7200):
    """
    Loads dataset files from the specified folder.
    Supports JSON, CSV, and NumPy arrays.
    """
    if not os.path.exists(directory):
        return None, None
        
    x_list, y_list = [], []
    
    # 1. Try loading JSON files first
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    for f in files:
        filepath = os.path.join(directory, f)
        with open(filepath, "r") as file_obj:
            try:
                data = json.load(file_obj)
                if "flux" in data and "label" in data:
                    x_list.append(data["flux"])
                    y_list.append(data["label"])
            except Exception:
                pass
                
    # 2. Try loading NumPy arrays if JSON not present
    if not x_list:
        try:
            npy_x_candidates = [f for f in os.listdir(directory) if "x_" in f or "X_" in f]
            npy_y_candidates = [f for f in os.listdir(directory) if "y_" in f or "Y_" in f]
            if npy_x_candidates and npy_y_candidates:
                x_arr = np.load(os.path.join(directory, npy_x_candidates[0]))
                y_arr = np.load(os.path.join(directory, npy_y_candidates[0]))
                # Convert back to list to allow uniform pipeline processing
                x_list = x_arr.tolist()
                y_list = y_arr.tolist()
        except Exception:
            pass
            
    if not x_list:
        return None, None
        
    # Preprocess all curves
    processed_x = []
    for curve in x_list:
        # Preprocess using the chosen config pipeline
        p_curve = apply_preprocessing(curve, pipeline_name=preprocessing_name)
        # Pad or segment to match expected flat segment length
        curr_len = len(p_curve)
        if curr_len > segment_len:
            start = (curr_len - segment_len) // 2
            p_curve = p_curve[start:start+segment_len]
        elif curr_len < segment_len:
            p_curve = np.pad(p_curve, (0, segment_len - curr_len), mode='edge')
        processed_x.append(p_curve)
        
    return np.array(processed_x, dtype=np.float32), np.array(y_list, dtype=np.float32)

def prepare_datasets():
    """
    Loads all data, applies augmentations on the training set,
    and returns balanced splits ready for model input.
    """
    # Load dataset splits
    train_data = load_data_from_dir("datasets/train", config.PREPROCESSING, config.SEGMENT_LENGTH)
    val_data = load_data_from_dir("datasets/validation", config.PREPROCESSING, config.SEGMENT_LENGTH)
    test_data = load_data_from_dir("datasets/test", config.PREPROCESSING, config.SEGMENT_LENGTH)
    
    if train_data[0] is None:
        raise ValueError(
            "No training dataset found in 'datasets/train/'. "
            "Please copy train light curves (.json, .csv, or .npy) first."
        )
        
    x_train, y_train = train_data
    x_val, y_val = val_data if val_data[0] is not None else (None, None)
    x_test, y_test = test_data if test_data[0] is not None else (None, None)
    
    # Apply augmentations on training positive transits
    if config.ENABLE_AUGMENTATION:
        x_aug, y_aug = [], []
        for curve, label in zip(x_train, y_train):
            if label == 1:
                aug_curves = augment_light_curve(
                    curve,
                    roll_frac=config.ROLL_FRACTION,
                    noise_std=config.NOISE_STD,
                    scale_min=config.SCALE_MIN,
                    scale_max=config.SCALE_MAX
                )
                for ac in aug_curves:
                    x_aug.append(ac)
                    y_aug.append(1.0)
            else:
                x_aug.append(curve)
                y_aug.append(0.0)
        x_train = np.array(x_aug, dtype=np.float32)
        y_train = np.array(y_aug, dtype=np.float32)
        
    # Oversample minority class to balance training set 1:1
    c0 = np.where(y_train == 0)[0]
    c1 = np.where(y_train == 1)[0]
    if len(c0) > 0 and len(c1) > 0 and len(c0) != len(c1):
        target_size = max(len(c0), len(c1))
        if len(c0) < target_size:
            extra = np.random.choice(c0, size=target_size - len(c0), replace=True)
            c0 = np.concatenate([c0, extra])
        if len(c1) < target_size:
            extra = np.random.choice(c1, size=target_size - len(c1), replace=True)
            c1 = np.concatenate([c1, extra])
        balanced_idx = np.concatenate([c0, c1])
        np.random.shuffle(balanced_idx)
        x_train = x_train[balanced_idx]
        y_train = y_train[balanced_idx]
        
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
