import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from ramanuj_model.architectures.inceptiontime import build_inception_module
from ramanuj_model.architectures.attention_blocks import SpatialAttention2D

def build_hybrid_model(input_shape=(9, 800, 1), l2_reg=1e-4, dropout_rate=0.3):
    """
    Builds a hybrid neural network combining baseline CNN feature extraction,
    parallel InceptionTime blocks, and minor-axis spatial self-attention layers.
    """
    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=input_shape)
    
    # 1. Baseline feature extraction
    x = layers.Conv2D(16, kernel_size=(3, 16), padding='same', kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 2. Multi-scale Inception block
    x = build_inception_module(x, 32, l2_reg)
    
    # 3. Spatial attention map weighting
    x = SpatialAttention2D(kernel_size=7)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)
