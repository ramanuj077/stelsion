import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Add parent directory to path to import filters
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from debo_model.debo_preprocessing import preprocess_light_curve

# --- 1. Dual-Input Data Preparation ---
def get_dual_inputs(index_path="debo_model/dataset_index.csv", metadata_path="koi_cumulative_labeled.csv"):
    """
    Loads light curves, phase-folds them using Kepler metadata, and creates
    both Global (2000 pts) and Local (200 pts) views.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index {index_path} not found.")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata {metadata_path} not found.")
        
    df_index = pd.read_csv(index_path)
    df_meta = pd.read_csv(metadata_path, comment='#')
    # Drop duplicates of kepid to ensure the index map is unique
    df_meta_unique = df_meta.drop_duplicates(subset="kepid")
    
    # Map kepid to period and t0
    meta_map = df_meta_unique.set_index("kepid")[["koi_period", "koi_time0bk"]].to_dict(orient="index")
    
    x_global = []
    x_local = []
    y = []
    
    print("Processing light curves into Global and Local views...")
    for idx, row in df_index.iterrows():
        filepath = row["file"]
        if not os.path.exists(filepath):
            filepath = os.path.join("..", filepath)
            if not os.path.exists(filepath):
                continue
                
        data = np.load(filepath)
        time_arr = data["time"]
        flux_arr = data["flux"]
        
        # Clean and normalize
        flux_clean = preprocess_light_curve(flux_arr)
        
        # Retrieve period and epoch for phase-folding
        kepid = row["kepid"]
        meta = meta_map.get(kepid, {"koi_period": None, "koi_time0bk": None})
        period = meta.get("koi_period")
        t0 = meta.get("koi_time0bk")
        
        if pd.isna(period) or pd.isna(t0) or period is None or t0 is None:
            # Fallback if no metadata: center on the minimum dip
            center_idx = np.argmin(flux_clean)
        else:
            # Calculate phase and find the transit center
            phases = ((time_arr - t0) % period) / period
            # Find the phase closest to 0.0 (the transit center)
            center_idx = np.argmin(np.minimum(phases, 1.0 - phases))
            
        # 1. Create Global View: align/pad/crop to 2000 points
        curr_len = len(flux_clean)
        if curr_len > 2000:
            start = (curr_len - 2000) // 2
            global_flux = flux_clean[start:start+2000]
        else:
            global_flux = np.pad(flux_clean, (0, 2000 - curr_len), mode='edge')
            
        # 2. Create Local View: 200 points centered on the transit/dip
        local_half = 100
        # Wrap index boundary safely
        start_idx = max(0, center_idx - local_half)
        end_idx = min(len(flux_clean), center_idx + local_half)
        
        local_flux = flux_clean[start_idx:end_idx]
        # Pad local view if it falls off edges
        if len(local_flux) < 200:
            pad_width = 200 - len(local_flux)
            local_flux = np.pad(local_flux, (0, pad_width), mode='edge')
            
        label = 1.0 if row["label"] == "transit" else 0.0
        
        x_global.append(global_flux)
        x_local.append(local_flux)
        y.append(label)
        
    return (np.array(x_global, dtype=np.float32), 
            np.array(x_local, dtype=np.float32), 
            np.array(y, dtype=np.float32))

# --- 2. Custom AstroNet Model Definition ---
def build_astronet(global_len=2000, local_len=200, freeze_first_two=True):
    """
    Implements a custom TF2/Keras dual-view AstroNet architecture.
    Optionally freezes the first 2 convolutional layers in both branches.
    """
    # Global Branch
    input_global = tf.keras.Input(shape=(global_len, 1), name="global_input")
    g_conv1 = tf.keras.layers.Conv1D(16, kernel_size=7, activation='relu', padding='same', name="global_conv1")(input_global)
    g_conv2 = tf.keras.layers.Conv1D(16, kernel_size=7, activation='relu', padding='same', name="global_conv2")(g_conv1)
    g_pool = tf.keras.layers.MaxPool1D(2)(g_conv2)
    
    g_conv3 = tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same', name="global_conv3")(g_pool)
    g_conv4 = tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same', name="global_conv4")(g_conv3)
    g_pool2 = tf.keras.layers.MaxPool1D(2)(g_conv4)
    g_flat = tf.keras.layers.Flatten()(g_pool2)
    
    # Local Branch
    input_local = tf.keras.Input(shape=(local_len, 1), name="local_input")
    l_conv1 = tf.keras.layers.Conv1D(8, kernel_size=5, activation='relu', padding='same', name="local_conv1")(input_local)
    l_conv2 = tf.keras.layers.Conv1D(8, kernel_size=5, activation='relu', padding='same', name="local_conv2")(l_conv1)
    l_pool = tf.keras.layers.MaxPool1D(2)(l_conv2)
    
    l_conv3 = tf.keras.layers.Conv1D(16, kernel_size=5, activation='relu', padding='same', name="local_conv3")(l_pool)
    l_flat = tf.keras.layers.Flatten()(l_conv3)
    
    # Merge and Fully Connected
    merged = tf.keras.layers.concatenate([g_flat, l_flat])
    fc1 = tf.keras.layers.Dense(256, activation='relu')(merged)
    drop1 = tf.keras.layers.Dropout(0.3)(fc1)
    fc2 = tf.keras.layers.Dense(128, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(0.3)(fc2)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)
    
    model = tf.keras.Model(inputs=[input_global, input_local], outputs=output, name="AstroNet_Custom")
    
    # Freeze the first 2 convolutional layers in each branch if requested
    if freeze_first_two:
        print("\nFreezing the first 2 convolutional layers in both branches...")
        model.get_layer("global_conv1").trainable = False
        model.get_layer("global_conv2").trainable = False
        model.get_layer("local_conv1").trainable = False
        model.get_layer("local_conv2").trainable = False
        
    return model

# --- 3. Main Experiment Execution ---
def main():
    print("=" * 70)
    print("             AstroNet DUAL-VIEW FINE-TUNING EXPERIMENT             ")
    print("=" * 70)
    
    # 1. Load data
    try:
        x_g, x_l, y = get_dual_inputs()
    except Exception as e:
        print(f"Error loading inputs: {e}")
        return
        
    print(f"Loaded {len(y)} samples successfully.")
    
    # 2. Reshape and Split
    x_g = np.expand_dims(x_g, axis=-1)
    x_l = np.expand_dims(x_l, axis=-1)
    
    # Train/Val/Test split
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.15, random_state=42, stratify=y)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.176, random_state=42, stratify=y[train_idx]) # ~15% of total
    
    x_g_train, x_g_val, x_g_test = x_g[train_idx], x_g[val_idx], x_g[test_idx]
    x_l_train, x_l_val, x_l_test = x_l[train_idx], x_l[val_idx], x_l[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    
    # 3. Build Model
    model = build_astronet(global_len=2000, local_len=200, freeze_first_two=False)
    model.summary()
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]
    # Calculate class weights to handle imbalance (Transit is ~28%)
    neg_count = np.sum(y_train == 0.0)
    pos_count = np.sum(y_train == 1.0)
    total = len(y_train)
    class_weight = {
        0: (total / (2.0 * neg_count)) if neg_count > 0 else 1.0,
        1: (total / (2.0 * pos_count)) if pos_count > 0 else 1.0
    }
    print(f"Class Weights -> 0: {class_weight[0]:.2f}, 1: {class_weight[1]:.2f}")
    
    # 4. Train/Fine-Tune
    print("\nTraining the fine-tuned AstroNet model...")
    history = model.fit(
        [x_g_train, x_l_train], y_train,
        validation_data=([x_g_val, x_l_val], y_val),
        epochs=15,
        batch_size=16,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    # 5. Evaluate on Test set
    print("\nEvaluating on Test Set...")
    y_pred_prob = model.predict([x_g_test, x_l_test], verbose=0).flatten()
    print(f"Prediction Probabilities -> Min: {np.min(y_pred_prob):.4f}, Max: {np.max(y_pred_prob):.4f}, Mean: {np.mean(y_pred_prob):.4f}")
    y_pred = (y_pred_prob >= 0.5).astype(int)
    y_true = y_test.astype(int)
    
    # Calculate classification metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    test_loss, test_acc = model.evaluate([x_g_test, x_l_test], y_test, verbose=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    print("=" * 60)
    print("                  FINAL TEST EVALUATION METRICS                  ")
    print("=" * 60)
    print(f"Test Loss:            {test_loss:.4f}")
    print(f"Test Accuracy:        {test_acc*100:.2f}%")
    print(f"Precision:            {precision*100:.2f}%")
    print(f"Recall (Sensitivity): {recall*100:.2f}%")
    print(f"F1-Score:             {f1*100:.2f}%")
    print("=" * 60)
    
    # Print Confusion Matrix Table
    print("\nConfusion Matrix / Classification Table:")
    print(f"{'':<20} | {'Predicted Non-Planet (0)':<25} | {'Predicted Planet (1)':<25}")
    print("-" * 78)
    print(f"{'Actual Non-Planet (0)':<20} | {f'TN: {tn}':<25} | {f'FP: {fp}':<25}")
    print(f"{'Actual Planet (1)':<20} | {f'FN: {fn}':<25} | {f'TP: {tp}':<25}")
    print("-" * 78)
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Non-Planet", "Planet"], zero_division=0))

if __name__ == "__main__":
    main()
