import sys
import os
# Add parent directory to sys.path to enable absolute package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from debo_model.dataset import prepare_splits
from debo_model.model import DeboDetectorNet

def main():
    print("=" * 60)
    print("      DeboModel: EXOPLANET DETECTION TRAINING WORKSPACE      ")
    print("=" * 60)
    
    # 1. Load splits
    print("\n[Step 1/4] Loading and Preprocessing Kepler dataset splits...")
    try:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_splits(
            index_path="debo_model/dataset_index.csv",
            target_len=2000,
            test_size=0.15,
            val_size=0.15,
            seed=42
        )
        print("Dataset loaded successfully:")
        print(f" - Train shapes: X={x_train.shape}, y={y_train.shape}")
        print(f" - Val shapes:   X={x_val.shape}, y={y_val.shape}")
        print(f" - Test shapes:  X={x_test.shape}, y={y_test.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check if dataset_fetcher.py is finished and downloaded the .npz files.")
        return

    # 2. Build model
    print("\n[Step 2/4] Initializing DeboDetectorNet (1D CNN + BiGRU)...")
    model = DeboDetectorNet(input_len=2000, dropout=0.3)
    
    # Build on dummy input
    dummy_input = tf.zeros((1, 2000, 1))
    _ = model(dummy_input)
    model.summary()

    # 3. Compile and callbacks
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    
    # Checkpoints
    checkpoint_dir = "debo_model/saved_models"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "debo_detector_best.weights.h5")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
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

    # 4. Fit Model
    print("\n[Step 3/4] Starting training loop...")
    epochs = 15
    batch_size = 16
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    # 5. Evaluate on Test set
    print("\n[Step 4/4] Evaluating best model on test dataset...")
    # Load best weights
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print("Loaded best weights from checkpoint.")
        
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    # Calculate predictions
    y_pred_prob = model.predict(x_test, verbose=0).flatten()
    print(f"Prediction Probabilities -> Min: {np.min(y_pred_prob):.4f}, Max: {np.max(y_pred_prob):.4f}, Mean: {np.mean(y_pred_prob):.4f}")
    y_pred = (y_pred_prob >= 0.5).astype(int)
    y_true = y_test.astype(int)
    
    # Calculate classification metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    print("-" * 50)
    print(f"Test Loss:            {test_loss:.4f}")
    print(f"Test Accuracy:        {test_acc*100:.2f}%")
    print(f"Precision:            {precision*100:.2f}%")
    print(f"Recall (Sensitivity): {recall*100:.2f}%")
    print(f"F1-Score:             {f1*100:.2f}%")
    print("-" * 50)
    
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
    print("Training process completed!")

if __name__ == "__main__":
    main()
