"""
Все энкодеры должны шарить общий интерфейс:
tf.Tensor([N, H, W, C], tf.float32) -> encoder.call -> tf.Tensor([N, frames, hidden], tf.float32)
"""
import tensorflow as tf
import numpy as np
from layers import FullGatedConv2D, DepthWiseConvBlock, SpatialTransformer, TransformerLayer


# image encoders

class EncoderFlor(tf.keras.layers.Layer):
    """
    https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/network/model.py#L424
    """

    def __init__(self):
        super().__init__()

        # 1
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same",
                                            kernel_initializer="he_uniform")
        self.prelu1 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn1 = tf.keras.layers.BatchNormalization(renorm=True)
        self.fgconv1 = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")

        # 2
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                            kernel_initializer="he_uniform")
        self.prelu2 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn2 = tf.keras.layers.BatchNormalization(renorm=True)
        self.fgconv2 = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")

        # 3
        self.conv3 = tf.keras.layers.Conv2D(filters=40, kernel_size=(2, 4), strides=(2, 4), padding="same",
                                            kernel_initializer="he_uniform")
        self.prelu3 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn3 = tf.keras.layers.BatchNormalization(renorm=True)
        self.fgconv3 = FullGatedConv2D(filters=40, kernel_size=(3, 3), padding="same",
                                       kernel_constraint=tf.keras.constraints.MaxNorm(4, [0, 1, 2]))
        self.dropout3 = tf.keras.layers.Dropout(rate=0.2)

        # 4
        self.conv4 = tf.keras.layers.Conv2D(filters=48, kernel_size=(2, 4), strides=(1, 1), padding="same",
                                            kernel_initializer="he_uniform")
        self.prelu4 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn4 = tf.keras.layers.BatchNormalization(renorm=True)
        self.fgconv4 = FullGatedConv2D(filters=48, kernel_size=(3, 3), padding="same",
                                       kernel_constraint=tf.keras.constraints.MaxNorm(4, [0, 1, 2]))
        self.dropout4 = tf.keras.layers.Dropout(rate=0.2)

        # 5
        self.conv5 = tf.keras.layers.Conv2D(filters=56, kernel_size=(2, 4), strides=(2, 4), padding="same",
                                            kernel_initializer="he_uniform")
        self.prelu5 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn5 = tf.keras.layers.BatchNormalization(renorm=True)
        self.fgconv5 = FullGatedConv2D(filters=56, kernel_size=(3, 3), padding="same",
                                       kernel_constraint=tf.keras.constraints.MaxNorm(4, [0, 1, 2]))
        self.dropout5 = tf.keras.layers.Dropout(rate=0.2)

        # 6
        self.conv6 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                            kernel_initializer="he_uniform")
        self.prelu6 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn6 = tf.keras.layers.BatchNormalization(renorm=True)
        self.pool6 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")

    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.bn1(x, training=training)
        x = self.fgconv1(x)

        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.bn2(x, training=training)
        x = self.fgconv2(x)

        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.bn3(x, training=training)
        x = self.fgconv3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.prelu4(x)
        x = self.bn4(x, training=training)
        x = self.fgconv4(x)
        x = self.dropout4(x)

        x = self.conv5(x)
        x = self.prelu5(x)
        x = self.bn5(x, training=training)
        x = self.fgconv5(x)
        x = self.dropout5(x)

        x = self.conv6(x)
        x = self.prelu6(x)
        x = self.bn6(x, training=training)
        x = self.pool6(x)

        x_shape = x.get_shape()
        x = tf.keras.layers.Reshape((x_shape[1], x_shape[2] * x_shape[3]))(x)
        return x


class EncoderFlorV2(tf.keras.layers.Layer):
    """
    https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/network/model.py#L424
    """

    def __init__(self):
        super().__init__()

        # 1
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same",
                                            kernel_initializer="he_uniform")
        self.prelu1 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn1 = tf.keras.layers.BatchNormalization(renorm=True)
        self.fgconv1 = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")

        # 2
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                            kernel_initializer="he_uniform")
        self.prelu2 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn2 = tf.keras.layers.BatchNormalization(renorm=True)
        self.fgconv2 = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")

        # 3
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 4), strides=(2, 4), padding="same",
                                            kernel_initializer="he_uniform")
        self.prelu3 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn3 = tf.keras.layers.BatchNormalization(renorm=True)
        self.fgconv3 = FullGatedConv2D(filters=64, kernel_size=(3, 3), padding="same",
                                       kernel_constraint=tf.keras.constraints.MaxNorm(4, [0, 1, 2]))
        self.dropout3 = tf.keras.layers.Dropout(rate=0.2)

        # 4
        self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 4), strides=(1, 1), padding="same",
                                            kernel_initializer="he_uniform")
        self.prelu4 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn4 = tf.keras.layers.BatchNormalization(renorm=True)
        self.fgconv4 = FullGatedConv2D(filters=128, kernel_size=(3, 3), padding="same",
                                       kernel_constraint=tf.keras.constraints.MaxNorm(4, [0, 1, 2]))
        self.dropout4 = tf.keras.layers.Dropout(rate=0.2)

        # 5
        self.conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(2, 4), strides=(2, 4), padding="same",
                                            kernel_initializer="he_uniform")
        self.prelu5 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn5 = tf.keras.layers.BatchNormalization(renorm=True)
        self.fgconv5 = FullGatedConv2D(filters=256, kernel_size=(3, 3), padding="same",
                                       kernel_constraint=tf.keras.constraints.MaxNorm(4, [0, 1, 2]))
        self.dropout5 = tf.keras.layers.Dropout(rate=0.2)

        # 6
        self.conv6 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                            kernel_initializer="he_uniform")
        self.prelu6 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn6 = tf.keras.layers.BatchNormalization(renorm=True)
        self.pool6 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")

    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.bn1(x, training=training)
        x = self.fgconv1(x)

        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.bn2(x, training=training)
        x = self.fgconv2(x)

        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.bn3(x, training=training)
        x = self.fgconv3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.prelu4(x)
        x = self.bn4(x, training=training)
        x = self.fgconv4(x)
        x = self.dropout4(x)

        x = self.conv5(x)
        x = self.prelu5(x)
        x = self.bn5(x, training=training)
        x = self.fgconv5(x)
        x = self.dropout5(x)

        x = self.conv6(x)
        x = self.prelu6(x)
        x = self.bn6(x, training=training)
        x = self.pool6(x)

        x_shape = x.get_shape()
        x = tf.keras.layers.Reshape((x_shape[1], x_shape[2] * x_shape[3]))(x)
        return x


class EncoderFlorV3(tf.keras.layers.Layer):
    """
    https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/network/model.py#L424
    """

    def __init__(self):
        super().__init__()

        # 1
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same",
                                            kernel_initializer="he_uniform")
        self.prelu1 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn1 = tf.keras.layers.BatchNormalization(renorm=True)
        self.fgconv1 = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")

        # 2
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                            kernel_initializer="he_uniform")
        self.prelu2 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn2 = tf.keras.layers.BatchNormalization(renorm=True)
        self.fgconv2 = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")

        # 3
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 4), strides=(2, 4), padding="same",
                                            kernel_initializer="he_uniform")
        self.prelu3 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn3 = tf.keras.layers.BatchNormalization(renorm=True)
        self.fgconv3 = FullGatedConv2D(filters=64, kernel_size=(3, 3), padding="same",
                                       kernel_constraint=tf.keras.constraints.MaxNorm(4, [0, 1, 2]))
        self.dropout3 = tf.keras.layers.Dropout(rate=0.2)

        # 4
        self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 4), strides=(1, 1), padding="same",
                                            kernel_initializer="he_uniform")
        self.prelu4 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn4 = tf.keras.layers.BatchNormalization(renorm=True)
        self.fgconv4 = FullGatedConv2D(filters=128, kernel_size=(3, 3), padding="same",
                                       kernel_constraint=tf.keras.constraints.MaxNorm(4, [0, 1, 2]))
        self.dropout4 = tf.keras.layers.Dropout(rate=0.2)

        # 5
        self.conv5 = tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 4), strides=(2, 4), padding="same",
                                            kernel_initializer="he_uniform")
        self.prelu5 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn5 = tf.keras.layers.BatchNormalization(renorm=True)
        self.fgconv5 = FullGatedConv2D(filters=128, kernel_size=(3, 3), padding="same",
                                       kernel_constraint=tf.keras.constraints.MaxNorm(4, [0, 1, 2]))
        self.dropout5 = tf.keras.layers.Dropout(rate=0.2)

        # 6
        self.conv6 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                            kernel_initializer="he_uniform")
        self.prelu6 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn6 = tf.keras.layers.BatchNormalization(renorm=True)
        self.pool6 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")

    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.bn1(x, training=training)
        x = self.fgconv1(x)

        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.bn2(x, training=training)
        x = self.fgconv2(x)

        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.bn3(x, training=training)
        x = self.fgconv3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.prelu4(x)
        x = self.bn4(x, training=training)
        x = self.fgconv4(x)
        x = self.dropout4(x)

        x = self.conv5(x)
        x = self.prelu5(x)
        x = self.bn5(x, training=training)
        x = self.fgconv5(x)
        x = self.dropout5(x)

        x = self.conv6(x)
        x = self.prelu6(x)
        x = self.bn6(x, training=training)
        x = self.pool6(x)

        x_shape = x.get_shape()
        x = tf.keras.layers.Reshape((x_shape[1], x_shape[2] * x_shape[3]))(x)
        return x


class EncoderBaseline(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(4, 2), strides=2)

        self.conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(4, 2), strides=2)

        self.conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')

        self.conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(4, 1), padding='same')

        self.conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        self.bn5 = tf.keras.layers.BatchNormalization()

        self.conv6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        self.bn6 = tf.keras.layers.BatchNormalization()
        self.pool6 = tf.keras.layers.MaxPool2D(pool_size=(4, 1), padding='same')

        self.conv7 = tf.keras.layers.Conv2D(512, (2, 2), activation='relu')

    def call(self, x):
        """
        x - tf.Tensor of shape [N, 128, 1024, 3]
        return: x - tf.Tensor of shape [N, 255, 512]
        """
        x = self.conv1(x)  # [128, 1024, 64]
        x = self.pool1(x)  # [63, 512, 64]
        x = self.conv2(x)  # [63, 512, 128]
        x = self.pool2(x)  # [30, 256, 128]
        x = self.conv3(x)  # [30, 256, 256]
        x = self.conv4(x)  # [30, 256, 256]
        x = self.pool4(x)  # [8, 256, 256]
        x = self.conv5(x)  # [8, 265, 512]
        x = self.bn5(x)
        x = self.conv6(x)  # [8, 256, 512]
        x = self.bn6(x)
        x = self.pool6(x)  # [2, 256, 512]
        x = self.conv7(x)  # [1, 255, 512]
        x = tf.squeeze(x, [1])  # [255, 512]
        return x


class EncoderDepthWiseConv(tf.keras.layers.Layer):
    def __init__(self, img_shape, use_stn=False, num_units=128, dropout=0.4):
        super().__init__()
        self.img_shape = img_shape

        self.use_stn = use_stn
        if use_stn:
            self.spatial_transformer = SpatialTransformer(sampling_size=img_shape[:2])
        else:
            self.spatial_transformer = None

        self.zero_pad = tf.keras.layers.ZeroPadding2D(padding=(2, 2))
        self.dw_conv1 = DepthWiseConvBlock((3, 3), 64, dropout=0.1, pooling=None)
        self.dw_conv2 = DepthWiseConvBlock((3, 3), 128, dropout=0.1, pooling=None)
        self.dw_conv3 = DepthWiseConvBlock((3, 3), 256, dropout=0.1, pooling=(2, 2))
        self.dw_conv4 = DepthWiseConvBlock((3, 3), 256, dropout=0.1, pooling=None)
        self.dw_conv5 = DepthWiseConvBlock((3, 3), 512, dropout=0.1, pooling=(1, 2))
        self.dw_conv6 = DepthWiseConvBlock((3, 3), 512, dropout=0.1, pooling=None)
        self.dw_conv7 = DepthWiseConvBlock((3, 3), 512, dropout=0.1, pooling=None)

        self.reshape = tf.keras.layers.Reshape(target_shape=self._infer_shape())
        self.dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=None):
        if self.use_stn:
            x = self.spatial_transformer(x)
        x = self.zero_pad(x)
        x = self.dw_conv1(x)
        x = self.dw_conv2(x)
        x = self.dw_conv3(x)
        x = self.dw_conv4(x)
        x = self.dw_conv5(x)
        x = self.dw_conv6(x)
        x = self.dw_conv7(x)
        x = self.reshape(x)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        return x

    def _infer_shape(self):
        num_pools_h = 1
        num_pools_w = 2
        h = (self.img_shape[0] + 4) // (2 ** num_pools_h)
        w = ((self.img_shape[1] + 4) // (2 ** num_pools_w)) * 512
        return h, w


# frame encoders

class DecoderBaseline(tf.keras.layers.Layer):
    # TODO: переименовать в BiRNNEncoder, а также поправить все места, где есть зависимость от имени данного слоя
    def __init__(self, cell_name="GRU", num_layers=2, cell_dim=256, dropout=0.2, add_skip_connections=False, add_projection_in=False):
        super().__init__()

        cell_cls = getattr(tf.keras.layers, cell_name)
        self.layers = [
            tf.keras.layers.Bidirectional(cell_cls(cell_dim, return_sequences=True, dropout=dropout))
            for _ in range(num_layers)
        ]
        
        self.add_skip_connections = add_skip_connections
        self.add_projection_in = add_projection_in

        if self.add_projection_in:
            self.dense_in = tf.keras.layers.Dense(cell_dim * 2)

    def call(self, x, training=None, mask=None):
        """
        x - tf.Tensor of shape [N, T, D]
        """
        if self.add_projection_in:
            x = self.dense_in(x)

        for layer in self.layers:
            x1 = layer(x, training=training, mask=mask)
            if self.add_skip_connections:
                x += x1
            else:
                x = x1
        return x


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, head_dim, dff, dropout):
        super().__init__()

        self.d_model = num_heads * head_dim
        self.dropout = dropout
        self.dense_in = tf.keras.layers.Dense(self.d_model)
        self.transformer_layers = [
            TransformerLayer(
                num_heads=num_heads,
                head_dim=head_dim,
                dff=dff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ]

        self.pos_encoding = None
        self._set_positional_encoding()

    def call(self, x, training=None):
        seq_len = tf.shape(x)[1]
        x = self.dense_in(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = tf.keras.layers.Dropout(self.dropout)(x, training=training)
        for layer in self.transformer_layers:
            x = layer(x)
        return x

    def _set_positional_encoding(self):
        maxlen = 10000
        pos = np.arange(maxlen)[:, None]
        i = np.arange(self.d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(self.d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[None, :, :]  # [1, maxlen, d_model]
        self.pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)  # [1, maxlen, d_model]
