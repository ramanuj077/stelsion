import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Add parent directory to path to import filters
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from debo_model.debo_preprocessing import preprocess_light_curve
from debo_model.astronet_experiment import get_dual_inputs

# --- 1. Transformer Encoder Layer ---
class TransformerEncoderBlock(layers.Layer):
    """
    Standard Transformer Encoder block in Keras.
    """
    def __init__(self, num_heads, key_dim, ff_dim, dropout=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        embed_dim = input_shape[-1]
        self.mha = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.ffn = tf.keras.Sequential([
            layers.Dense(self.ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        super(TransformerEncoderBlock, self).build(input_shape)
        
    def call(self, inputs, training=None):
        # Self-Attention
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-Forward Network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# --- 2. Build Stage 5 CNN-Transformer Hybrid Model ---
def build_stage5_model(global_len=2000, local_len=200, num_heads=4, key_dim=16, ff_dim=64):
    """
    Implements Stage 5 Architecture:
    Global branch: Conv1D -> Conv1D -> Transformer -> Global Feature Vector
    Local branch: Conv1D -> Conv1D -> Transformer -> Local Feature Vector
    Features are merged and classified.
    """
    # ================= GLOBAL BRANCH =================
    input_global = layers.Input(shape=(global_len, 1), name="global_input")
    
    # 2x Conv1D layers
    g_conv1 = layers.Conv1D(32, kernel_size=7, activation='relu', padding='same', name="global_conv1")(input_global)
    g_conv2 = layers.Conv1D(32, kernel_size=7, activation='relu', padding='same', name="global_conv2")(g_conv1)
    
    # Transformer Encoder (Directly following Conv1D layers)
    g_trans = TransformerEncoderBlock(num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim, name="global_transformer")(g_conv2)
    
    # Global Feature Vector extraction
    g_feat = layers.GlobalAveragePooling1D(name="global_feature_vector")(g_trans)
    
    # ================= LOCAL BRANCH =================
    input_local = layers.Input(shape=(local_len, 1), name="local_input")
    
    # 2x Conv1D layers
    l_conv1 = layers.Conv1D(16, kernel_size=5, activation='relu', padding='same', name="local_conv1")(input_local)
    l_conv2 = layers.Conv1D(16, kernel_size=5, activation='relu', padding='same', name="local_conv2")(l_conv1)
    
    # Transformer Encoder (Directly following Conv1D layers)
    l_trans = TransformerEncoderBlock(num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim, name="local_transformer")(l_conv2)
    
    # Local Feature Vector extraction
    l_feat = layers.GlobalAveragePooling1D(name="local_feature_vector")(l_trans)
    
    # ================= FEATURE MERGE & HEAD =================
    merged = layers.concatenate([g_feat, l_feat], name="concatenate_features")
    
    fc1 = layers.Dense(128, activation='relu')(merged)
    drop1 = layers.Dropout(0.3)(fc1)
    fc2 = layers.Dense(64, activation='relu')(drop1)
    drop2 = layers.Dropout(0.3)(fc2)
    output = layers.Dense(1, activation='sigmoid')(drop2)
    
    model = Model(inputs=[input_global, input_local], outputs=output, name="CNN_Transformer_Hybrid")
    return model

# --- 3. Main Experiment Execution ---
def main():
    print("=" * 70)
    print("             Stage 5 CNN-TRANSFORMER HYBRID PIPELINE             ")
    print("=" * 70)
    
    # 1. Load data
    try:
        x_g, x_l, y = get_dual_inputs()
    except Exception as e:
        print(f"Error loading inputs: {e}")
        return
        
    print(f"Loaded {len(y)} samples successfully.")
    
    # 2. Reshape and Split
    x_g = np.expand_dims(x_g, axis=-1)
    x_l = np.expand_dims(x_l, axis=-1)
    
    # Split
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.15, random_state=42, stratify=y)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.176, random_state=42, stratify=y[train_idx])
    
    x_g_train, x_g_val, x_g_test = x_g[train_idx], x_g[val_idx], x_g[test_idx]
    x_l_train, x_l_val, x_l_test = x_l[train_idx], x_l[val_idx], x_l[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    
    # 3. Build Model
    model = build_stage5_model()
    model.summary()
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
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
    
    # 4. Train Model
    print("\nTraining the CNN-Transformer Hybrid model...")
    history = model.fit(
        [x_g_train, x_l_train], y_train,
        validation_data=([x_g_val, x_l_val], y_val),
        epochs=15,
        batch_size=16,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    # 5. Evaluate on Test set
    print("\nEvaluating on Test Set...")
    y_pred_prob = model.predict([x_g_test, x_l_test], verbose=0).flatten()
    print(f"Prediction Probabilities -> Min: {np.min(y_pred_prob):.4f}, Max: {np.max(y_pred_prob):.4f}, Mean: {np.mean(y_pred_prob):.4f}")
    y_pred = (y_pred_prob >= 0.5).astype(int)
    y_true = y_test.astype(int)
    
    # Calculate classification metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    test_loss, test_acc = model.evaluate([x_g_test, x_l_test], y_test, verbose=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    print("=" * 60)
    print("                  FINAL TEST EVALUATION METRICS                  ")
    print("=" * 60)
    print(f"Test Loss:            {test_loss:.4f}")
    print(f"Test Accuracy:        {test_acc*100:.2f}%")
    print(f"Precision:            {precision*100:.2f}%")
    print(f"Recall (Sensitivity): {recall*100:.2f}%")
    print(f"F1-Score:             {f1*100:.2f}%")
    print("=" * 60)
    
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

if __name__ == "__main__":
    main()
