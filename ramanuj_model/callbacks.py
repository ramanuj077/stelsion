import os
import tensorflow as tf

def get_callbacks(experiment_dir, patience_early_stopping=8, patience_reduce_lr=3):
    """
    Creates and returns standard TensorFlow callbacks:
    - EarlyStopping: prevents overfitting by stopping when validation loss stops improving.
    - ReduceLROnPlateau: decreases learning rate dynamically when loss plateaus.
    - ModelCheckpoint: saves best weights to disk.
    - TensorBoard: logs training metrics and charts.
    """
    os.makedirs(experiment_dir, exist_ok=True)
    checkpoint_path = os.path.join(experiment_dir, "model.h5")
    log_dir = os.path.join(experiment_dir, "logs")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience_early_stopping,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_reduce_lr,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Safely add TensorBoard callback
    try:
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
        callbacks.append(tb_callback)
    except Exception as e:
        print(f"Warning: TensorBoard not available in environment. Skipping TensorBoard logging. Error: {e}")
        
    return callbacks
