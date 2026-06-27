import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from ramanuj_model.architectures.attention_blocks import SpatialAttention2D, ChannelAttention2D

def build_minor_axis_attention_model(input_shape=(9, 800, 1), l2_reg=1e-4, dropout_rate=0.3):
    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(16, kernel_size=(3, 16), padding='same', kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = ChannelAttention2D(ratio=4)(x)
    
    x = layers.Conv2D(32, kernel_size=(3, 5), padding='same', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = SpatialAttention2D(kernel_size=7)(x)
    
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

def build_matrix_branch(input_shape=(9, 800, 1), l2_reg=1e-4):
    """
    Builds the 2D CNN branch with channel and spatial attention modules
    to process the folded orbit matrix (shape 9x800x1).
    """
    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(16, kernel_size=(3, 16), padding='same', kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = ChannelAttention2D(ratio=4)(x)
    
    x = layers.Conv2D(32, kernel_size=(3, 5), padding='same', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = SpatialAttention2D(kernel_size=7)(x)
    
    x = layers.Conv2D(64, kernel_size=(3, 5), padding='same', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = ChannelAttention2D(ratio=8)(x)
    x = SpatialAttention2D(kernel_size=7)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    return models.Model(inputs=inputs, outputs=x, name="matrix_branch")
