# import tensorflow as tf
import tensorflow.compat.v1 as tf


class Encoder(tf.keras.layers.Layer):
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

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=(4, 1), padding='same')

        self.conv7 = tf.keras.layers.Conv2D(512, (2, 2), activation='relu')

    def call(self, x):
        """
        x - tf.Tensor of shape [N, 128, 1024, 3]
        return: x - tf.Tensor of shape [N, 255, 512]
        """
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.bn1(x)
        x = self.conv6(x)
        x = self.bn2(x)
        x = self.pool5(x)
        x = self.conv7(x)
        x = tf.squeeze(x, [1])
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size):
        """
        vocab_size - число символов С УЧЁТОМ пустого символа
        """
        super().__init__()

        gru_cell1 = tf.keras.layers.GRU(256, return_sequences=True, dropout=0.2)
        self.gru_layer1 = tf.keras.layers.Bidirectional(gru_cell1)

        gru_cell2 = tf.keras.layers.GRU(256, return_sequences=True, dropout=0.2)
        self.gru_layer2 = tf.keras.layers.Bidirectional(gru_cell2)

        self.fc = tf.keras.layers.Dense(vocab_size + 1)

    def call(self, x):
        """
        x - tf.Tensor of shape [N, T, D]
        """
        x = self.gru_layer1(x)
        x = self.gru_layer2(x)
        x = self.fc(x)  # [N, T, vocab_size]
        return x


class Model:
    def __init__(self):
        self.images_ph = None
        self.char_indices_ph = None
        self.char_values_ph = None
        self.sequence_len_ph = None
        self.logits_len_ph = None
        self.dense_shape_ph = None

        self.loss = None
        self.train_op = None

    def build(self):
        self.images_ph = tf.placeholder(tf.float32, shape=[None, 128, 1024, 3])
        self.char_indices_ph = tf.placeholder(tf.int32, shape=[None, None])
        self.char_values_ph = tf.placeholder(tf.int32, shape=[None])
        self.sequence_len_ph = tf.placeholder(tf.int32, shape=[None])
        self.logits_len_ph = tf.placeholder(tf.int32, shape=[None])
        self.dense_shape_ph = tf.placeholder(tf.int32, shape=[2])  # [batch_size, max_len_batch]

        encoder = Encoder()
        decoder = Decoder(vocab_size=len(char2id))

        x = encoder(self.images_ph)  # [N, T, D]
        x = decoder(x)  # [N, T, V]
        x = tf.transpose(x, [1, 0, 2])  # [T, N, V]
        labels = tf.SparseTensor(
            indices=tf.cast(self.char_indices_ph, tf.int64),
            values=self.char_values_ph,
            dense_shape=tf.cast(self.dense_shape_ph, tf,int64)
        )
        loss = tf.nn.ctc_loss(labels=labels, inputs=x, sequence_length=self.sequence_len_ph)
        self.loss = tf.reduce_mean(loss)

        opt = tf.train.AdamOptimizer()
        self.train_op = opt.minimize(self.loss)

    # def fit(self,
    #         train_data: List,
    #         eval_data: List = None,
    #         batch_size: int = 32,
    #         epochs: int = 7,
    #         update_plot_step: int = 5
    #         ):
    #     global_step = 0
    #     train_loss = []
    #     eval_loss = []
    #
    #     for epoch in range(1, epochs+1):
    #         print(f"epoch: {epoch}")
    #         examples_epoch = self._get_shuffled_data(train_data, seed=epoch)
    #
    #         for start in range(0, len(examples_epoch), batch_size):
    #             end = start + batch_size
    #             examples_batch = examples_epoch[start:end]
    #             loss = self._train_step(examples_batch)
    #             train_loss.append(loss)
    #
    #             if global_step % update_plot_step == 0:
    #                 self._plot(train_loss, eval_loss)
    #
    #             global_step += 1
    #
    #         if eval_data:
    #             print(f"evaluation on random examples starts; global_step: {global_step}")
    #             eval_loss_epoch = []
    #             for start in range(0, len(eval_data), batch_size):
    #                 end = start + batch_size
    #                 examples_batch = eval_data[start:end]
    #                 loss = self._eval_step(examples_batch)
    #                 eval_loss_epoch.append(loss)
    #             loss = np.mean(eval_loss_epoch)
    #             eval_loss.append(loss)
    #             print(f"eval loss: {loss}")
    #
    #     self._plot(train_loss, eval_loss)
    #
    #     return train_loss, eval_loss


train_step_signature = [
    tf.TensorSpec(shape=(None, 128, 1024, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None,), dtype=tf.int32),
    tf.TensorSpec(shape=(None,), dtype=tf.int32),
]

# train_step_signature = [
#     tf.TensorSpec(shape=(BATCH_SIZE, 128, 1024, 3), dtype=tf.float32),
#     tf.TensorSpec(shape=(BATCH_SIZE, MAXLEN), dtype=tf.int32),
#     tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.int32),
#     tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.int32),
# ]


@tf.function(input_signature=train_step_signature)
def train_step(images, labels, sequence_len, logit_length):
    indices = tf.cast(tf.where(tf.sequence_mask(sequence_len)), tf.int64)
    values = tf.gather_nd(labels, indices)
    dense_shape = tf.cast(tf.shape(labels), tf.int64)
    labels_sparse = tf.sparse.SparseTensor(indices, values, dense_shape)
    with tf.GradientTape() as tape:
        x = encoder(images)
        logits = decoder(x)
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/ctc_loss
        # Using time_major = True (default) is a bit more efficient
        # because it avoids transposes at the beginning of the ctc_loss calculation.
        logits = tf.transpose(logits, [1, 0, 2])
        loss = tf.nn.ctc_loss(
            # labels=labels,  # [batch_size, max_label_seq_length]
            labels=labels_sparse,
            logits=logits,  # [frames, batch_size, num_labels]
            label_length=sequence_len,  # [batch_size]
            logit_length=logit_length,   # [batch_size]
            logits_time_major=True,  # дефолтное значение
            # unique=None,  # пока хз, что это
            blank_index=BLANK_INDEX  # по дефолту 0, но решил явно прокинуть BLANK_INDEX = 0
            # blank_index=0
        )
        loss = tf.reduce_mean(loss)
    tvars = encoder.trainable_variables + decoder.trainable_variables
    grads = tape.gradient(loss, tvars)
    optimizer.apply_gradients(zip(grads, tvars))
    return loss


def train(num_epochs):
    for epoch in range(num_epochs):
        start = time.time()
        total_loss = 0
        num_steps = 0
        for batch, inputs in enumerate(ds_train):
            loss = train_step(*inputs)
            loss = loss.numpy()
            total_loss += loss
            num_steps += 1
            if batch % 10 == 0:
                print(f'Epoch {epoch} Batch {batch} Loss {loss}')
        loss_epoch = total_loss / num_steps
        loss_plot.append(loss_epoch)

        #         if epoch % 5 == 0:
        #             ckpt_manager.save()

        print(f'Epoch {epoch + 1} Loss {loss_epoch}')
        print(f'Time taken for 1 epoch {time.time() - start} sec')
        print()
