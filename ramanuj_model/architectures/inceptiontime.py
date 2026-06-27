import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_inception_module(inputs, filters, l2_reg=1e-4):
    """
    Builds a single Inception module with parallel convolutional filters
    of different kernel sizes (3x3, 3x5, 3x9) to capture varying temporal signatures.
    """
    reg = regularizers.l2(l2_reg)
    
    # Bottleneck 1x1 convolution
    bottleneck = layers.Conv2D(
        filters, kernel_size=(1, 1), padding='same', activation='linear', 
        kernel_regularizer=reg, use_bias=False
    )(inputs)
    
    # Parallel branches
    conv3 = layers.Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=reg)(bottleneck)
    conv5 = layers.Conv2D(filters, kernel_size=(3, 5), padding='same', activation='relu', kernel_regularizer=reg)(bottleneck)
    conv9 = layers.Conv2D(filters, kernel_size=(3, 9), padding='same', activation='relu', kernel_regularizer=reg)(bottleneck)
    
    # Max pooling branch
    pool = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    conv_pool = layers.Conv2D(filters, kernel_size=(1, 1), padding='same', activation='relu', kernel_regularizer=reg)(pool)
    
    # Concatenate all branches along the channel axis
    concat = layers.Concatenate(axis=-1)([conv3, conv5, conv9, conv_pool])
    bn = layers.BatchNormalization()(concat)
    return layers.Activation('relu')(bn)

def build_inceptiontime_model(input_shape=(9, 800, 1), l2_reg=1e-4, dropout_rate=0.3):
    """
    Builds the full InceptionTime network by stacking Inception modules.
    """
    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=input_shape)
    
    x = build_inception_module(inputs, 16, l2_reg)
    x = build_inception_module(x, 32, l2_reg)
    x = build_inception_module(x, 64, l2_reg)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)
