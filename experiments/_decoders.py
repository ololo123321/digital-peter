"""
Все декодеры должны шарить общий интерфейс:
tf.Tensor([batch, time, hidden_in], tf.float32) -> decoder.call -> tf.Tensor([batch, time, hidden_out], tf.float32)
"""
import tensorflow as tf
from layers import TransformerLayer


class DecoderBaseline(tf.keras.layers.Layer):
    def __init__(self, cell_name="GRU", num_layers=2, cell_dim=256, dropout=0.2, skip_conn=False):
        super().__init__()

        cell_cls = getattr(tf.keras.layers, cell_name)
        self.layers = [
            tf.keras.layers.Bidirectional(cell_cls(cell_dim, return_sequences=True, dropout=dropout))
            for _ in range(num_layers)
        ]
        self.skip_conn = skip_conn

    def call(self, x):
        """
        x - tf.Tensor of shape [N, T, D]
        """
        for layer in self.layers:
            if self.skip_conn:
                x += layer(x)
            else:
                x = layer(x)
        return x


class DecoderTransformer(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, head_dim, dff, dropout):
        super().__init__()

        self.dense_in = tf.keras.layers.Dense(num_heads * head_dim)
        self.transformer_layers = [
            TransformerLayer(
                num_heads=num_heads,
                head_dim=head_dim,
                dff=dff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ]

    def call(self, x):
        x = self.dense_in(x)
        for layer in self.transformer_layers:
            x = layer(x)
        return x
