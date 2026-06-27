import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_baseline_model(input_shape=(9, 800, 1), l2_reg=1e-4, dropout_rate_fc1=0.3, dropout_rate_fc2=0.2):
    """
    Builds the baseline secondary Conv2D architecture, matching the production model.
    """
    reg = regularizers.l2(l2_reg)
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, kernel_size=(3, 16), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(32, kernel_size=(3, 5), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(64, kernel_size=(3, 5), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(64, kernel_size=(3, 5), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate_fc1),
        layers.Dense(256, kernel_regularizer=reg),
        layers.ReLU(),
        layers.Dropout(dropout_rate_fc2),
        layers.Dense(128, kernel_regularizer=reg),
        layers.ReLU(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_local_branch(input_shape=(200, 1), l2_reg=1e-4):
    """
    Builds a 1D CNN Branch with residual connections to process the local transit view (200 samples).
    """
    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv1D(16, kernel_size=5, padding='same', kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Block 2
    shortcut = x
    x = layers.Conv1D(32, kernel_size=5, padding='same', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(32, kernel_size=5, padding='same', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    
    # Project shortcut
    project_shortcut = layers.Conv1D(32, kernel_size=1, padding='same', kernel_regularizer=reg)(shortcut)
    x = layers.add([x, project_shortcut])
    x = layers.ReLU()(x)
    
    # Pool
    x = layers.GlobalAveragePooling1D()(x)
    return models.Model(inputs=inputs, outputs=x, name="local_branch")
