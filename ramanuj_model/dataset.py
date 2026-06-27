import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from astropy.timeseries import BoxLeastSquares
from ramanuj_model.preprocessing import apply_preprocessing_pipeline
import ramanuj_model.config as config

def run_bls_candidate_generation(time: np.ndarray, flux: np.ndarray):
    """
    Runs Box Least Squares (BLS) periodogram search on a light curve
    to estimate candidate transit parameters.
    """
    try:
        min_p = config.BLS_MIN_PERIOD
        max_p = min(config.BLS_MAX_PERIOD, (np.max(time) - np.min(time)) / 2.0)
        if max_p <= min_p:
            max_p = min_p + 5.0
            
        bls = BoxLeastSquares(time, flux)
        durations = np.linspace(0.04, 0.2, 5)
        period_grid = np.linspace(min_p, max_p, 1000)
        power_results = bls.power(period_grid, durations)
        
        best_idx = np.argmax(power_results.power)
        period = float(power_results.period[best_idx])
        t0 = float(power_results.transit_time[best_idx])
        duration = float(power_results.duration[best_idx])
        depth = float(power_results.depth[best_idx])
    except Exception:
        # Fallback values
        period = 10.0
        t0 = np.min(time) + 2.0
        duration = 0.1
        depth = 0.01
        
    return period, t0, duration, depth

def extract_folded_views(time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration: float):
    """
    Generates three representation views from the folded light curve:
    1. Global View (2000 bins)
    2. Local View (200 bins)
    3. Orbit Matrix (10 x 500 = 5000 elements)
    """
    # 1. Fold light curve and center transit at phase 0.5
    phase = ((time - t0) / period + 0.5) % 1.0
    sort_idx = np.argsort(phase)
    folded_phase = phase[sort_idx]
    folded_flux = flux[sort_idx]
    
    # 2. Global View: interpolate complete orbit to 2000 points
    global_grid = np.linspace(0.0, 1.0, config.GLOBAL_VIEW_LEN)
    global_view = np.interp(global_grid, folded_phase, folded_flux)
    
    # 3. Local View: extract transit window (4 * duration) centered at phase 0.5
    local_width = 4.0 * (duration / period)
    local_start = max(0.0, 0.5 - local_width / 2.0)
    local_end = min(1.0, 0.5 + local_width / 2.0)
    
    local_grid = np.linspace(local_start, local_end, config.LOCAL_VIEW_LEN)
    local_view = np.interp(local_grid, folded_phase, folded_flux)
    
    # 4. Orbit Matrix: segment by individual orbits (shape 10 x 500 = 5000 elements)
    num_orbits = 10
    num_bins = 500
    t_start, t_end = np.min(time), np.max(time)
    
    start_orb = int(np.floor((t_start - t0) / period))
    end_orb = int(np.ceil((t_end - t0) / period))
    
    orbits = []
    for i in range(start_orb, end_orb):
        orb_t_start = t0 + (i - 0.5) * period
        orb_t_end = t0 + (i + 0.5) * period
        
        mask = (time >= orb_t_start) & (time < orb_t_end)
        t_orb = time[mask]
        f_orb = flux[mask]
        
        if len(t_orb) > 5:
            p_orb = (t_orb - orb_t_start) / period
            s_idx = np.argsort(p_orb)
            
            grid_orb = np.linspace(0.0, 1.0, num_bins)
            binned = np.interp(grid_orb, p_orb[s_idx], f_orb[s_idx])
            orbits.append(binned)
            
    if len(orbits) == 0:
        orbits = [np.zeros(num_bins) for _ in range(num_orbits)]
        
    if len(orbits) < num_orbits:
        padding = [np.zeros(num_bins) for _ in range(num_orbits - len(orbits))]
        orbits.extend(padding)
    elif len(orbits) > num_orbits:
        orbits = orbits[:num_orbits]
        
    orbit_matrix = np.array(orbits, dtype=np.float32)
    
    return global_view, local_view, orbit_matrix

def pack_views(global_view, local_view, orbit_matrix):
    """
    Concatenates global, local, and matrix views into a single flat array
    of length 7200 and reshapes it to (9, 800, 1).
    """
    packed = np.concatenate([
        global_view.flatten(),
        local_view.flatten(),
        orbit_matrix.flatten()
    ])
    return packed.reshape(9, 800, 1)

def prepare_datasets():
    """
    Loads all .npz Kepler light curves listed in dataset_index.csv.
    Applies adaptive preprocessing, runs BLS periodograms, folds, splits,
    balances, and exports test files for shared auto-benchmarking.
    """
    index_path = "dataset_index.csv"
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"dataset_index.csv not found. Run dataset_fetcher.py first.")
        
    df = pd.read_csv(index_path)
    # Remove duplicates by KIC ID (kepid)
    df = df.drop_duplicates(subset=["kepid"])
    
    x_packed = []
    labels = []
    kepids = []
    
    print("Ingesting and processing Kepler NPZ light curves...")
    for _, row in df.iterrows():
        file_path = row["file"]
        kepid = row["kepid"]
        lbl_str = str(row["label"]).strip().lower()
        
        # Map labels: confirmed planet or candidate -> 1.0, others -> 0.0
        label = 1.0 if lbl_str in ["transit", "candidate"] else 0.0
        
        if not os.path.exists(file_path):
            continue
            
        try:
            npz = np.load(file_path)
            time = npz["time"]
            flux = npz["flux"]
            
            # Remove NaNs from raw input
            nan_mask = np.isnan(time) | np.isnan(flux)
            time = time[~nan_mask]
            flux = flux[~nan_mask]
            
            if len(flux) < 50:
                continue
                
            # Apply configurable preprocessing
            clean_flux = apply_preprocessing_pipeline(flux)
            
            # Run BLS candidate generation
            period, t0, duration, depth = run_bls_candidate_generation(time, clean_flux)
            
            # Fold and extract representation views
            g_view, l_view, o_matrix = extract_folded_views(time, clean_flux, period, t0, duration)
            
            # Pack views into the single (9, 800, 1) shape expected by the model
            packed_sample = pack_views(g_view, l_view, o_matrix)
            
            x_packed.append(packed_sample)
            labels.append(label)
            kepids.append(kepid)
        except Exception as e:
            print(f"Warning: Corrupted sample {kepid} skipped: {e}")
            
    # Convert lists to NumPy arrays
    x_packed = np.array(x_packed, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    kepids = np.array(kepids, dtype=np.int32)
    
    # Stratified Splits (70% Train, 15% Validation, 15% Test)
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=config.SEED, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=config.SEED, stratify=labels[temp_idx]
    )
    
    # Extract splits
    train_split = (x_packed[train_idx], labels[train_idx])
    val_split = (x_packed[val_idx], labels[val_idx])
    test_split = (x_packed[test_idx], labels[test_idx])
    
    # In-memory logging of dataset details
    print("\n" + "=" * 50)
    print("DATASET PIPELINE INGESTION LOG")
    print("=" * 50)
    print(f"Total Unique Ingested Samples: {len(labels)}")
    print(f"Class Distribution: {np.sum(labels == 1.0)} Transits, {np.sum(labels == 0.0)} Non-Transits")
    print(f"Train set size:       {len(train_idx)} samples")
    print(f"Validation set size:  {len(val_idx)} samples")
    print(f"Test set size:        {len(test_idx)} samples")
    print("=" * 50 + "\n")
    
    # Save test dataset split to datasets/test/ for shared benchmark evaluation
    os.makedirs("datasets/test", exist_ok=True)
    for idx in test_idx:
        k_id = kepids[idx]
        lbl = int(labels[idx])
        flat_flux = x_packed[idx].flatten()
        test_file = os.path.join("datasets/test", f"KIC_{k_id}.json")
        with open(test_file, "w") as f:
            json.dump({
                "flux": flat_flux.tolist(),
                "label": lbl,
                "candidate": f"KIC {k_id}"
            }, f)
            
    # Apply balancing to training set (oversample minority class)
    tr_x, tr_y = train_split
    c0 = np.where(tr_y == 0.0)[0]
    c1 = np.where(tr_y == 1.0)[0]
    
    if len(c0) > 0 and len(c1) > 0 and len(c0) != len(c1):
        target = max(len(c0), len(c1))
        if len(c0) < target:
            extra = np.random.choice(c0, size=target - len(c0), replace=True)
            c0 = np.concatenate([c0, extra])
        if len(c1) < target:
            extra = np.random.choice(c1, size=target - len(c1), replace=True)
            c1 = np.concatenate([c1, extra])
            
        balanced_indices = np.concatenate([c0, c1])
        np.random.shuffle(balanced_indices)
        
        train_split = (tr_x[balanced_indices], tr_y[balanced_indices])
        
    return train_split, val_split, test_split
