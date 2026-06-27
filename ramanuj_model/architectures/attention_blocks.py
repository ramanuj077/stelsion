import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package="RamanujAttention")
class SpatialAttention2D(layers.Layer):
    """
    Computes spatial attention map using average and max pooling across channels,
    followed by a standard 2D convolution and sigmoid scaling.
    """
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention2D, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = None
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding='same',
            activation='sigmoid',
            use_bias=False
        )
        super(SpatialAttention2D, self).build(input_shape)
        
    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_out, max_out], axis=-1)
        attention = self.conv(concat)
        return inputs * attention
        
    def get_config(self):
        config = super(SpatialAttention2D, self).get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


@tf.keras.utils.register_keras_serializable(package="RamanujAttention")
class ChannelAttention2D(layers.Layer):
    """
    Computes channel attention map using squeeze-and-excitation block.
    """
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention2D, self).__init__(**kwargs)
        self.ratio = ratio
        self.shared_mlp_1 = None
        self.shared_mlp_2 = None
        
    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_mlp_1 = layers.Dense(channel // self.ratio, activation='relu', use_bias=False)
        self.shared_mlp_2 = layers.Dense(channel, use_bias=False)
        super(ChannelAttention2D, self).build(input_shape)
        
    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2])
        max_pool = tf.reduce_max(inputs, axis=[1, 2])
        
        avg_out = self.shared_mlp_2(self.shared_mlp_1(avg_pool))
        max_out = self.shared_mlp_2(self.shared_mlp_1(max_pool))
        
        attention = tf.sigmoid(avg_out + max_out)[:, tf.newaxis, tf.newaxis, :]
        return inputs * attention

    def get_config(self):
        config = super(ChannelAttention2D, self).get_config()
        config.update({"ratio": self.ratio})
        return config
