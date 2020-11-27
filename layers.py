import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np


class FullGatedConv2D(tf.keras.layers.Conv2D):
    """Gated Convolutional Class"""

    def __init__(self, filters, **kwargs):
        super().__init__(filters=filters * 2, **kwargs)
        self.nb_filters = filters

    def call(self, inputs):
        """Apply gated convolution"""
        output = super().call(inputs)
        linear = tf.keras.layers.Activation("linear")(output[:, :, :, :self.nb_filters])
        sigmoid = tf.keras.layers.Activation("sigmoid")(output[:, :, :, self.nb_filters:])
        return tf.keras.layers.Multiply()([linear, sigmoid])

    def compute_output_shape(self, input_shape):
        """Compute shape of layer output"""
        output_shape = super().compute_output_shape(input_shape)
        return tuple(output_shape[:3]) + (self.nb_filters,)

    def get_config(self):
        """Return the config of the layer"""
        config = super().get_config()
        config['nb_filters'] = self.nb_filters
        del config['filters']
        return config


class TransformerModel(tf.keras.layers.Layer):
    def __init__(self, vocab_size, num_layers=6, num_heads=8, head_dim=64, dff=1024, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = num_heads * head_dim
        self.dropout = dropout
        self.emb = tf.keras.layers.Embedding(vocab_size, self.d_model)
        self.dec_layers = [
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

    def call(self, x, training=None, **kwargs):
        seq_len = tf.shape(x)[1]
        mask = self._build_mask(x, seq_len)
        x = self.emb(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = tf.keras.layers.Dropout(self.dropout)(x, training=training)
        for dec in self.dec_layers:
            x = dec(x, training=training, mask=mask)
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

    @staticmethod
    def _build_mask(x, seq_len):
        padding_mask = tf.cast(tf.math.equal(x, 0), tf.float32)  # [N, T]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)  # [seq_len, seq_len]
        combined_mask = padding_mask[:, None, None, :] + look_ahead_mask[None, None, :, :]  # [N, 1, T, T]
        return combined_mask


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_dim, dff, dropout):
        super().__init__()
        d_model = num_heads * head_dim
        self.mha = MHA(num_heads, head_dim)
        self.dense_ff = tf.keras.layers.Dense(dff, activation=tf.nn.relu)
        self.dense_model = tf.keras.layers.Dense(d_model)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask=None, training=None):
        x1 = self.mha(x, mask=mask)
        x1 = self.dropout1(x1, training=training)
        x = self.ln1(x + x1)
        x1 = self.dense_ff(x)
        x1 = self.dense_model(x1)
        x1 = self.dropout2(x1, training=training)
        x = self.ln2(x + x1)
        return x


class MHA(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dense_input = tf.keras.layers.Dense(num_heads * head_dim * 3)

    def call(self, x, mask=None):
        """
        https://arxiv.org/abs/1706.03762
        :param x: tf.Tensor of shape [N, T, H * D]
        :param mask: tf.Tensor of shape [N, T]
        :return: tf.Tensor of shape [N, T, H * D]
        """
        batch_size = tf.shape(x)[0]
        qkv = self.dense_input(x)  # [N, T, H * D * 3]
        qkv = tf.reshape(qkv, [batch_size, -1, self.num_heads, self.head_dim, 3])  # [N, T, H, D, 3]
        qkv = tf.transpose(qkv, [4, 0, 2, 1, 3])  # [3, N, H, T, D]
        #         q, k, v = tf.unstack(qkv)  # 3 * [N, H, T, D]
        qkv = tf.unstack(qkv)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        logits = tf.matmul(q, k, transpose_b=True)  # [N, H, T, T]
        logits /= self.head_dim ** 0.5  # [N, H, T, T]

        if mask is not None:
            logits += (mask * -1e9)

        w = tf.nn.softmax(logits, axis=-1)  # [N, H, T, T] (k-axis)
        x = tf.matmul(w, v)  # [N, H, T, D]
        x = tf.transpose(x, [0, 2, 1, 3])  # [N, T, H, D]
        x = tf.reshape(x, [batch_size, -1, self.num_heads * self.head_dim])  # [N, T, D * H]
        return x


class BiRNNLayer(tf.keras.layers.Layer):
    """
    fixed
    """
    def __init__(
            self,
            cell_name,
            cell_dim,
            num_layers,
            dropout,
            skip_connections=True,
            stateful=False
    ):
        super().__init__()

        self.layers_fw = []
        self.layers_bw = []
        cell_cls = getattr(tf.keras.layers, cell_name)
        for _ in range(num_layers):
            layer = cell_cls(cell_dim, return_sequences=True, dropout=dropout, stateful=stateful)
            self.layers_fw.append(layer)
            layer = cell_cls(cell_dim, return_sequences=True, dropout=dropout, stateful=stateful)
            self.layers_bw.append(layer)

        self.skip_connections = skip_connections

    def call(self, x, char_ids, training=None):
        """
        char_ids прокидываются для того, чтобы явно не прокидывать длины последовательностей,
        чтоб можно было использовать один генератор данных и для rnn, и для трансформера
        """
        x_fw = x
        for layer in self.layers_fw:
            if self.skip_connections:
                x_fw += layer(x_fw, training=training)
            else:
                x_fw = layer(x_fw, training=training)

        sequence_mask = tf.math.not_equal(char_ids, 0)
        seq_lengths = tf.reduce_sum(tf.cast(sequence_mask, tf.int32), axis=1)
        x_bw = tf.reverse_sequence(x, seq_lengths=seq_lengths, seq_axis=1, batch_axis=0)
        for layer in self.layers_bw:
            if self.skip_connections:
                x_bw += layer(x_bw, training=training)
            else:
                x_bw = layer(x_bw, training=training)

        x_bw = tf.reverse_sequence(x_bw, seq_lengths=seq_lengths, seq_axis=1, batch_axis=0)
        x = tf.concat([x_fw[:, :-2, :], x_bw[:, 2:, :]], axis=-1)
        return x


class DepthWiseConvBlock(tf.keras.layers.Layer):
    def __init__(self, depth_wise_conv_kernel_size=(3, 3), point_wise_conv_num_filters=64, dropout=0.1, pooling=None):
        super().__init__()
        self.conv1 = tf.keras.layers.DepthwiseConv2D(depth_wise_conv_kernel_size, padding='same', strides=(1, 1), depth_multiplier=1, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu1 = tf.keras.layers.ReLU(max_value=6.0)

        self.conv2 = tf.keras.layers.Conv2D(point_wise_conv_num_filters, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu2 = tf.keras.layers.ReLU(max_value=6.0)

        if pooling is not None:
            self.pool = tf.keras.layers.MaxPooling2D(pooling)
        else:
            self.pool = None

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        if self.pool is not None:
            x = self.pool(x)

        x = self.dropout(x, training=training)
        return x


class BiLinearInterpolation(tf.keras.layers.Layer):
    """Performs bilinear interpolation as a keras layer
    References
    ----------
    [1]  Spatial Transformer Networks, Max Jaderberg, et al.
    [2]  https://github.com/skaae/transformer_network
    [3]  https://github.com/EderSantana/seya
    """

    def __init__(self, output_size=(100, 32), **kwargs):
        self.output_size = output_size
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        height, width = self.output_size
        num_channels = input_shapes[0][-1]
        return None, height, width, num_channels

    def call(self, tensors, mask=None):
        X, transformation = tensors
        output = self._transform(X, transformation, self.output_size)
        return output

    @staticmethod
    def _interpolate(image, sampled_grids, output_size):

        batch_size = K.shape(image)[0]
        height = K.shape(image)[1]
        width = K.shape(image)[2]
        num_channels = K.shape(image)[3]

        x = K.cast(K.flatten(sampled_grids[:, 0:1, :]), dtype='float32')
        y = K.cast(K.flatten(sampled_grids[:, 1:2, :]), dtype='float32')

        x = .5 * (x + 1.0) * K.cast(width, dtype='float32')
        y = .5 * (y + 1.0) * K.cast(height, dtype='float32')

        x0 = K.cast(x, 'int32')
        x1 = x0 + 1
        y0 = K.cast(y, 'int32')
        y1 = y0 + 1

        max_x = int(K.int_shape(image)[2] - 1)
        max_y = int(K.int_shape(image)[1] - 1)

        x0 = K.clip(x0, 0, max_x)
        x1 = K.clip(x1, 0, max_x)
        y0 = K.clip(y0, 0, max_y)
        y1 = K.clip(y1, 0, max_y)

        pixels_batch = K.arange(0, batch_size) * (height * width)
        pixels_batch = K.expand_dims(pixels_batch, axis=-1)
        flat_output_size = output_size[0] * output_size[1]
        base = K.repeat_elements(pixels_batch, flat_output_size, axis=1)
        base = K.flatten(base)

        # base_y0 = base + (y0 * width)
        base_y0 = y0 * width
        base_y0 = base + base_y0
        # base_y1 = base + (y1 * width)
        base_y1 = y1 * width
        base_y1 = base_y1 + base

        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = K.reshape(image, shape=(-1, num_channels))
        flat_image = K.cast(flat_image, dtype='float32')
        pixel_values_a = K.gather(flat_image, indices_a)
        pixel_values_b = K.gather(flat_image, indices_b)
        pixel_values_c = K.gather(flat_image, indices_c)
        pixel_values_d = K.gather(flat_image, indices_d)

        x0 = K.cast(x0, 'float32')
        x1 = K.cast(x1, 'float32')
        y0 = K.cast(y0, 'float32')
        y1 = K.cast(y1, 'float32')

        area_a = K.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = K.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = K.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = K.expand_dims(((x - x0) * (y - y0)), 1)

        values_a = area_a * pixel_values_a
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d
        return values_a + values_b + values_c + values_d

    @staticmethod
    def _make_regular_grids(batch_size, height, width):
        # making a single regular grid
        x_linspace = tf.linspace(-1., 1., width)
        y_linspace = tf.linspace(-1., 1., height)
        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = K.flatten(x_coordinates)
        y_coordinates = K.flatten(y_coordinates)
        ones = K.ones_like(x_coordinates)
        grid = K.concatenate([x_coordinates, y_coordinates, ones], 0)

        # repeating grids for each batch
        grid = K.flatten(grid)
        grids = K.tile(grid, K.stack([batch_size]))
        return K.reshape(grids, (batch_size, 3, height * width))

    def _transform(self, X, affine_transformation, output_size):
        batch_size, num_channels = K.shape(X)[0], K.shape(X)[3]
        transformations = K.reshape(affine_transformation,
                                    shape=(batch_size, 2, 3))
        # transformations = K.cast(affine_transformation[:, 0:2, :], 'float32')
        regular_grids = self._make_regular_grids(batch_size, *output_size)
        sampled_grids = K.batch_dot(transformations, regular_grids)
        interpolated_image = self._interpolate(X, sampled_grids, output_size)
        new_shape = (batch_size, output_size[0], output_size[1], num_channels)
        interpolated_image = K.reshape(interpolated_image, new_shape)
        return interpolated_image

    def get_config(self):
        config = super().get_config()
        config['output_size'] = self.output_size
        return config


class SpatialTransformer(tf.keras.layers.Layer):
    def __init__(self, sampling_size):
        super().__init__()

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.conv1 = tf.keras.layers.Conv2D(20, kernel_size=(5, 5))

        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(20, kernel_size=(5, 5))

        self.flatten = tf.keras.layers.Flatten()
        self.dense3 = tf.keras.layers.Dense(50)  # TODO: почему 50
        self.relu = tf.keras.layers.Activation('relu')

        self.dense4 = tf.keras.layers.Dense(6, weights=self._get_initial_weights(50))  # TODO: почему 6
        self.bilinear_interpolation = BiLinearInterpolation(output_size=sampling_size)

    def call(self, img, **kwargs):
        x = self.pool1(img)
        x = self.conv1(x)

        x = self.pool2(x)
        x = self.conv2(x)

        x = self.flatten(x)
        x = self.dense3(x)
        x = self.relu(x)

        x = self.dense4(x)
        x = self.bilinear_interpolation([img, x])
        return x

    @staticmethod
    def _get_initial_weights(output_size):
        b = np.zeros((2, 3), dtype='float32')
        b[0, 0] = 1
        b[1, 1] = 1
        W = np.zeros((output_size, 6), dtype='float32')
        weights = [W, b.flatten()]
        return weights


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, num_layers=6, num_heads=8, head_dim=64, dff=1024, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = num_heads * head_dim
        self.dropout = dropout
        self.emb = tf.keras.layers.Embedding(vocab_size, self.d_model)
        self.dec_layers = [
            TransformerDecoderLayer(
                num_heads=num_heads,
                head_dim=head_dim,
                dff=dff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ]

        self.pos_encoding = None
        self._set_positional_encoding()

    def call(self, x, enc_output, training=None, **kwargs):
        # вход декодера на уровне символов -> нужно маскировать впереди стойщие символы + паддинги:
        dec_combined_mask = self._build_combined_mask(x)

        # выход энкодера на уровне фреймов -> нечего маскировать
        enc_padding_mask = None

        x = self.emb(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :tf.shape(x)[1], :]
        x = tf.keras.layers.Dropout(self.dropout)(x, training=training)
        for dec in self.dec_layers:
            x = dec(
                x,
                enc_output=enc_output,
                training=training,
                dec_combined_mask=dec_combined_mask,
                enc_padding_mask=enc_padding_mask
            )
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

    @staticmethod
    def _build_combined_mask(x):
        """
        x - tf.Tensor of dtype tf.int32 and shape [N, T_dst]
        """
        seq_len = tf.shape(x)[1]
        padding_mask = tf.cast(tf.math.equal(x, 0), tf.float32)  # [N, T]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)  # [T, T]
        combined_mask = padding_mask[:, None, None, :] + look_ahead_mask[None, None, :, :]  # [N, 1, T, T]
        return combined_mask


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_dim, dff, dropout):
        super().__init__()
        d_model = num_heads * head_dim

        self.mha1 = MHA2(num_heads, head_dim)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.mha2 = MHA2(num_heads, head_dim)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dense_ff = tf.keras.layers.Dense(dff, activation=tf.nn.relu)
        self.dense_model = tf.keras.layers.Dense(d_model)
        self.dropout3 = tf.keras.layers.Dropout(dropout)
        self.ln3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, enc_output, training=None, dec_combined_mask=None, enc_padding_mask=None):
        x1 = self.mha1(q=x, k=x, v=x, mask=dec_combined_mask)
        x1 = self.dropout1(x1, training=training)
        x = self.ln1(x + x1)

        x1 = self.mha2(q=x, k=enc_output, v=enc_output, mask=enc_padding_mask)
        x1 = self.dropout2(x1, training=training)
        x = self.ln2(x + x1)

        x1 = self.dense_ff(x)
        x1 = self.dense_model(x1)
        x1 = self.dropout3(x1, training=training)
        x = self.ln3(x + x1)
        return x


class MHA2(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.dense_q = tf.keras.layers.Dense(num_heads * head_dim)
        self.dense_k = tf.keras.layers.Dense(num_heads * head_dim)
        self.dense_v = tf.keras.layers.Dense(num_heads * head_dim)

    def call(self, q, k, v, mask=None):
        """
        if q = k = v, then `mask` - tf.Tensor of shape [N, 1, T_dst, T_dst] (combined = look_ahead + padding_dst)
        if q = k, then `mask` - tf.Tensor of shape [N, 1, 1, T_src] (padding_src)
        """
        batch_size = tf.shape(q)[0]

        q = self.dense_q(q)  # [N, T_dst, H * D]
        k = self.dense_k(k)  # [N, T_src, H * D]
        v = self.dense_v(v)  # [N, T_src, H * D]

        q = tf.reshape(q, [batch_size, -1, self.num_heads, self.head_dim])  # [N, T_dst, H, D]
        k = tf.reshape(k, [batch_size, -1, self.num_heads, self.head_dim])  # [N, T_src, H, D]
        v = tf.reshape(v, [batch_size, -1, self.num_heads, self.head_dim])  # [N, T_src, H, D]

        q = tf.transpose(q, [0, 2, 1, 3])  # [N, H, T_dst, D]
        k = tf.transpose(k, [0, 2, 3, 1])  # [N, H, D, T_src]
        v = tf.transpose(v, [0, 2, 1, 3])  # [N, H, T_src, D]

        logits = tf.matmul(q, k)  # [N, H, T_dst, T_src]
        logits /= self.head_dim ** 0.5  # [N, H, T_dst, T_src]

        if mask is not None:
            logits += (mask * -1e9)

        w = tf.nn.softmax(logits, axis=-1)  # [N, H, T_dst, T_src] (k-axis)
        x = tf.matmul(w, v)  # [N, H, T_dst, D]
        x = tf.transpose(x, [0, 2, 1, 3])  # [N, T_dst, H, D]
        x = tf.reshape(x, [batch_size, -1, self.num_heads * self.head_dim])  # [N, T_dst, D * H]
        return x


class TopConvLayerV1(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_initializer="he_uniform"
        )
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn = tf.keras.layers.BatchNormalization(renorm=True)
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")

    def call(self, x, training=None):
        """
        x - tf.Tensor of shape [N, 128, 16, 512]
        """
        x = self.conv(x)
        x = self.prelu(x)
        x = self.bn(x, training=training)
        x = self.pool(x)
        return x
