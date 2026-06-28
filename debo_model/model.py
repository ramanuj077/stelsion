import tensorflow as tf
from tensorflow.keras import layers, models

class ResidualBlock1D(layers.Layer):
    """
    Self-contained 1D Residual block.
    """
    def __init__(self, filters, stride=1, dropout=0.2, **kwargs):
        super(ResidualBlock1D, self).__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.dropout_rate = dropout

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.conv1 = layers.Conv1D(self.filters, kernel_size=5, strides=self.stride, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv1D(self.filters, kernel_size=5, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.dropout = layers.Dropout(self.dropout_rate)
        self.relu2 = layers.ReLU()
        
        if self.stride != 1 or in_channels != self.filters:
            self.shortcut_conv = layers.Conv1D(self.filters, kernel_size=1, strides=self.stride, padding='same', use_bias=False)
            self.shortcut_bn = layers.BatchNormalization()
            
        super(ResidualBlock1D, self).build(input_shape)

    def call(self, x, training=None):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu1(out)
        out = self.dropout(out, training=training)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        
        if hasattr(self, 'shortcut_conv'):
            shortcut = self.shortcut_conv(x)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = x
            
        out += shortcut
        out = self.relu2(out)
        return out

class DeboDetectorNet(models.Model):
    """
    A custom neural network combining 1D CNN, Residual blocks, and Bidirectional GRUs
    for robust Kepler light curve classification.
    """
    def __init__(self, input_len=2000, dropout=0.3, **kwargs):
        super(DeboDetectorNet, self).__init__(**kwargs)
        self.input_len = input_len
        
        # 1. Feature extraction layer
        self.conv1 = layers.Conv1D(32, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.maxpool = layers.MaxPool1D(pool_size=3, strides=2, padding='same')
        
        # 2. Residual representations
        self.res1 = ResidualBlock1D(64, stride=2, dropout=dropout)
        self.res2 = ResidualBlock1D(128, stride=2, dropout=dropout)
        
        # 3. Recurrent Layer for temporal dependency (Transit period/duration)
        self.bigru = layers.Bidirectional(layers.GRU(64, return_sequences=False))
        
        # 4. Dense heads
        self.fc1 = layers.Dense(64, activation='relu')
        self.dropout = layers.Dropout(dropout)
        self.fc2 = layers.Dense(1, activation='sigmoid')

    def call(self, x, training=None):
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=-1)
            
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.maxpool(x)
        
        x = self.res1(x, training=training)
        x = self.res2(x, training=training)
        
        x = self.bigru(x, training=training)
        
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        logits = self.fc2(x)
        
        return logits
