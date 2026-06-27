import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from ramanuj_model.architectures.inceptiontime import build_global_inception_module_1d

def build_channel_attention_block(inputs, ratio=8):
    """
    Standard serializable Squeeze-and-Excitation channel attention block.
    """
    channels = inputs.shape[-1]
    
    # Shared MLP layers
    mlp_1 = layers.Dense(channels // ratio, activation='relu', use_bias=False)
    mlp_2 = layers.Dense(channels, use_bias=False)
    
    avg_pool = layers.GlobalAveragePooling2D()(inputs)
    max_pool = layers.GlobalMaxPooling2D()(inputs)
    
    avg_out = mlp_2(mlp_1(avg_pool))
    max_out = mlp_2(mlp_1(max_pool))
    
    # Combined attention vector
    channel_attention = layers.Activation('sigmoid')(layers.add([avg_out, max_out]))
    channel_attention = layers.Reshape((1, 1, channels))(channel_attention)
    
    return layers.Multiply()([inputs, channel_attention])

def build_spatial_attention_block(inputs, kernel_size=7):
    """
    Standard serializable spatial attention block using Conv2D layers.
    Uses a standard learnable 1x1 convolution for channel pooling to avoid
    custom constant initializer serialization issues.
    """
    # Learnable channel aggregation using 1x1 convolution
    avg_out = layers.Conv2D(
        filters=1,
        kernel_size=(1, 1),
        padding='same',
        activation='linear',
        use_bias=False
    )(inputs)
    
    # Apply 2D convolution for spatial feature mapping
    spatial_attention = layers.Conv2D(
        filters=1,
        kernel_size=(kernel_size, kernel_size),
        padding='same',
        activation='sigmoid',
        use_bias=False
    )(avg_out)
    
    return layers.Multiply()([inputs, spatial_attention])

def build_hybrid_model(input_shape=(9, 800, 1), l2_reg=1e-4, dropout_rate=0.3):
    """
    Builds the STELSION Research Model V2 using Keras Functional API with standard layers.
    Uses LayerNormalization instead of BatchNormalization to ensure stable training and inference
    on small-sample sizes.
    """
    reg = regularizers.l2(l2_reg)
    
    # 1. Single input (compatible with benchmark.py)
    inputs = layers.Input(shape=input_shape, name="input_1")
    
    # Flatten input to (7200, 1) to prepare for Cropping1D
    flat_3d = layers.Reshape((7200, 1), name="flat_3d")(inputs)
    
    # 2. Slice and reshape views using Cropping1D (fully serializable without Lambda)
    x_global = layers.Cropping1D(cropping=(0, 5200), name="crop_global")(flat_3d)
    x_local = layers.Cropping1D(cropping=(2000, 5000), name="crop_local")(flat_3d)
    x_matrix_flat = layers.Cropping1D(cropping=(2200, 0), name="crop_matrix")(flat_3d)
    x_matrix = layers.Reshape((10, 500, 1), name="reshape_matrix")(x_matrix_flat)
    
    # ==========================================
    # --- BRANCH 1: Global View (InceptionTime) ---
    # ==========================================
    g = build_global_inception_module_1d(x_global, 16, l2_reg)
    g = build_global_inception_module_1d(g, 32, l2_reg)
    g_out = layers.GlobalAveragePooling1D(name="gap_global")(g)
    
    # ==========================================
    # --- BRANCH 2: Local View (1D CNN) ---
    # ==========================================
    l = layers.Conv1D(16, kernel_size=5, padding='same', kernel_regularizer=reg)(x_local)
    l = layers.LayerNormalization()(l)
    l = layers.ReLU()(l)
    
    # Residual block
    l_shortcut = l
    l = layers.Conv1D(32, kernel_size=5, padding='same', kernel_regularizer=reg)(l)
    l = layers.LayerNormalization()(l)
    l = layers.ReLU()(l)
    l = layers.Conv1D(32, kernel_size=5, padding='same', kernel_regularizer=reg)(l)
    l = layers.LayerNormalization()(l)
    
    l_project = layers.Conv1D(32, kernel_size=1, padding='same', kernel_regularizer=reg)(l_shortcut)
    l = layers.add([l, l_project])
    l = layers.ReLU()(l)
    l_out = layers.GlobalAveragePooling1D(name="gap_local")(l)
    
    # ==========================================
    # --- BRANCH 3: Orbit Matrix (2D CNN + Attention) ---
    # ==========================================
    m = layers.Conv2D(16, kernel_size=(3, 16), padding='same', kernel_regularizer=reg)(x_matrix)
    m = layers.LayerNormalization()(m)
    m = layers.ReLU()(m)
    m = build_channel_attention_block(m, ratio=4)
    
    m = layers.Conv2D(32, kernel_size=(3, 5), padding='same', kernel_regularizer=reg)(m)
    m = layers.LayerNormalization()(m)
    m = layers.ReLU()(m)
    m = build_spatial_attention_block(m, kernel_size=7)
    
    m = layers.Conv2D(64, kernel_size=(3, 5), padding='same', kernel_regularizer=reg)(m)
    m = layers.LayerNormalization()(m)
    m = layers.ReLU()(m)
    m = build_channel_attention_block(m, ratio=8)
    m = build_spatial_attention_block(m, kernel_size=7)
    m_out = layers.GlobalAveragePooling2D(name="gap_matrix")(m)
    
    # ==========================================
    # --- FEATURE FUSION & CLASSIFIER ---
    # ==========================================
    fused = layers.Concatenate(axis=-1)([g_out, l_out, m_out])
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(fused)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1, activation='sigmoid', name="sigmoid_output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="StelsionV2Model")
    return model
