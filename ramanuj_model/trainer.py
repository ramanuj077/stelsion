import os
import sys
import time
import tensorflow as tf
import optuna
import numpy as np
from ramanuj_model.model import get_model
from ramanuj_model.dataset import prepare_datasets
from ramanuj_model.losses import get_loss
from ramanuj_model.callbacks import get_callbacks
from ramanuj_model.experiment import create_experiment_dir, save_experiment_results
from ramanuj_model.utils import get_system_metadata, set_seed, setup_gpu
from evaluation.metrics import calculate_metrics
import ramanuj_model.config as config

def train_model(config_overrides: dict = None):
    """
    Main training loop for STELSION Research Model V2.
    """
    set_seed(config.SEED)
    setup_gpu()
    
    # 1. Load train/val/test splits (Phase 1)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_datasets()
    
    # 2. Build model
    model = get_model(**(config_overrides or {}))
    
    # 3. Compile with learning rate, optimizer, loss function, and gradient clipping
    lr = config.LEARNING_RATE
    if config_overrides and "learning_rate" in config_overrides:
        lr = config_overrides["learning_rate"]
        
    optimizer_name = config.OPTIMIZER
    if config_overrides and "optimizer" in config_overrides:
        optimizer_name = config_overrides["optimizer"]
        
    # Support gradient clipping
    if optimizer_name == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
        
    loss_fn = get_loss(config.LOSS)
    model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
    model.summary()
    
    # 4. Compute class weights to address balance
    c0 = np.sum(y_train == 0.0)
    c1 = np.sum(y_train == 1.0)
    total = len(y_train)
    class_weights = {0.0: total / (2.0 * c0), 1.0: total / (2.0 * c1)} if (c0 > 0 and c1 > 0) else None
    
    # 5. Callbacks & Directory setup
    exp_dir = create_experiment_dir()
    callbacks = get_callbacks(exp_dir, patience_early_stopping=15, patience_reduce_lr=5)
    
    # 6. Fit
    epochs = config.EPOCHS
    if config_overrides and "epochs" in config_overrides:
        epochs = config_overrides["epochs"]
        
    batch_size = config.BATCH_SIZE
    if config_overrides and "batch_size" in config_overrides:
        batch_size = config_overrides["batch_size"]
        
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val) if x_val is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # 7. Evaluate on validation set
    val_metrics = {}
    y_val_prob = None
    if x_val is not None:
        y_val_prob = model.predict(x_val).flatten()
        y_val_pred = (y_val_prob >= 0.5).astype(int)
        val_metrics = calculate_metrics(y_val, y_val_pred, y_val_prob)
        
    # 8. Save results
    save_experiment_results(
        exp_dir, model, history, val_metrics,
        config_overrides=config_overrides,
        y_val_true=y_val, y_val_prob=y_val_prob
    )
    
    # 9. Save copy to saved_models directory (Isolated inside ramanuj_model)
    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
    final_model_path = os.path.join(config.SAVED_MODELS_DIR, f"{config.ARCHITECTURE}_model.h5")
    
    # Re-compile with standard binary_crossentropy to ensure it loads without custom loss functions
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.save(final_model_path)
    
    return final_model_path, val_metrics

def run_optuna_study():
    """
    Optimizes validation F1 score across several trials.
    """
    (x_train, y_train), (x_val, y_val), _ = prepare_datasets()
    if x_val is None:
        raise ValueError("Optuna study requires a validation dataset.")
        
    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout_rate", 0.1, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
        
        model = get_model(dropout_rate=dropout)
        
        if optimizer_name == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        else:
            opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, clipnorm=1.0)
            
        model.compile(optimizer=opt, loss=get_loss(config.LOSS), metrics=['accuracy'])
        
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=5,
            batch_size=batch_size,
            callbacks=[es],
            verbose=0
        )
        
        y_val_prob = model.predict(x_val).flatten()
        y_val_pred = (y_val_prob >= 0.5).astype(int)
        metrics = calculate_metrics(y_val, y_val_pred, y_val_prob)
        return metrics["f1_score"]
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.OPTUNA_TRIALS)
    
    print("\n" + "=" * 50)
    print("OPTUNA OPTIMIZATION STUDY RESULTS")
    print("=" * 50)
    print(f"Best validation F1 Score: {study.best_trial.value:.4f}")
    print("Best parameters:")
    for k, v in study.best_trial.params.items():
        print(f"  - {k}: {v}")
    print("=" * 50)
    
    return study.best_trial.params
