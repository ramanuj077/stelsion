import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from ramanuj_model.architectures.attention_blocks import SpatialAttention2D, ChannelAttention2D

def build_minor_axis_attention_model(input_shape=(9, 800, 1), l2_reg=1e-4, dropout_rate=0.3):
    """
    Builds a CNN architecture incorporating spatial and channel self-attention blocks
    to isolate and weigh exoplanet transit features along the minor axis.
    """
    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=input_shape)
    
    # Conv Block 1 + Channel Attention
    x = layers.Conv2D(16, kernel_size=(3, 16), padding='same', kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = ChannelAttention2D(ratio=4)(x)
    
    # Conv Block 2 + Spatial Attention
    x = layers.Conv2D(32, kernel_size=(3, 5), padding='same', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = SpatialAttention2D(kernel_size=7)(x)
    
    # Conv Block 3 + Hybrid Attention
    x = layers.Conv2D(64, kernel_size=(3, 5), padding='same', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = ChannelAttention2D(ratio=8)(x)
    x = SpatialAttention2D(kernel_size=7)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)
