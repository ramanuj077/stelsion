import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_baseline_model(input_shape=(9, 800, 1), l2_reg=1e-4, dropout_rate_fc1=0.3, dropout_rate_fc2=0.2):
    """
    Builds the baseline secondary Conv2D architecture, matching the production model:
    Input (9x800x1) -> Conv2D(16, 3x16) -> Conv2D(32, 3x5) -> Conv2D(64, 3x5) 
                    -> Conv2D(64, 3x5) -> GlobalAveragePooling2D -> FC(256) -> FC(128) -> Out(1)
    """
    reg = regularizers.l2(l2_reg)
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Conv 1
        layers.Conv2D(16, kernel_size=(3, 16), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # Conv 2
        layers.Conv2D(32, kernel_size=(3, 5), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # Conv 3
        layers.Conv2D(64, kernel_size=(3, 5), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # Conv 4
        layers.Conv2D(64, kernel_size=(3, 5), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate_fc1),
        
        # FC 256
        layers.Dense(256, kernel_regularizer=reg),
        layers.ReLU(),
        layers.Dropout(dropout_rate_fc2),
        
        # FC 128
        layers.Dense(128, regularizer=reg),
        layers.ReLU(),
        
        # Output Sigmoid
        layers.Dense(1, activation='sigmoid')
    ])
    return model
