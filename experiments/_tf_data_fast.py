import tensorflow as tf
from functools import partial

AUTOTUNE = tf.data.experimental.AUTOTUNE


def io_wrapper(**kwargs):
    """
    При обучении и валидации каждая функция обработка датасета должна
    шарить общий интерфейс: (x, y) -> f -> (x, y).
    При этом x - четвёрка:
    * картинка - для энкодера
    * символы (включая pad) - для ctc_loss
    * число символов (не включая pad) - для ctc_loss
    * число логитов - для ctc_loss
    При инференсе нам нужны только изображения, следовательно интерфейс функций обработки
    датасета следующий: img -> f -> img.
    Данный декоратор требуется для использования функций-обработчиков изображений
    в режимах обучения и валидации с использоваванием tf.data.Dataset API.
    """
    def wrapper(func):
        def wrapped(x, y):
            img, labels, logits_len, labels_len = x
            img = func(img, **kwargs)
            x = img, labels, logits_len, labels_len
            return x, y
        return wrapped
    return wrapper


class DataPipelineBuilder:
    def __init__(
            self,
            num_frames,
            num_channels=3,
            target_height=128,
            target_width=1024,
            max_delta_stretch=0.3,
            max_delta_brightness=0.1,
            rot90=False
    ):
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.target_height = target_height
        self.target_width = target_width
        assert 0 <= max_delta_stretch < 1
        self.max_delta_stretch = max_delta_stretch
        self.max_delta_brightness = max_delta_brightness
        self.rot90 = rot90

    def build_train_dataset(self, examples, batch_size=16, buffer=100):
        """
        Yields tuples:
        * img - tf.Tensor of shape [N, H, W, C] - images
        * labels - tf.Tensor of shape [N, T] - captions
        * logits_len - tf.Tensor of shape [N] - number of frames from image
        * labels_len - tf.Tensor of shape [N] - number of non-pad elements in labels
        """
        # взятие необходимого контента из примеров
        # files_img, labels, logits_len, labels_len = map(list, zip(*examples))
        files_img, labels, logits_len, labels_len = [], [], [], []
        for x in examples:
            files_img.append(x.img)
            labels.append(x.char_ids)
            logits_len.append(x.logits_len)
            labels_len.append(len(x.text))

        ds = tf.data.Dataset.from_tensor_slices((files_img, labels, logits_len, labels_len))

        # перемешивание
        ds = ds.shuffle(buffer)

        # чтение изображений
        ds = ds.map(lambda *args: read_fn_dev(*args, channels=self.num_channels), num_parallel_calls=AUTOTUNE)

        # кэширование детерминированной части датасета: она не особо большая
        ds = ds.cache()

        # приведение к фиксированной высоте (с изменением пропорций)
        wrapper = io_wrapper(target_height=self.target_height, target_width=self.target_width,
                             max_delta_stretch=self.max_delta_stretch, max_k_left=1.0)
        stretch_fn = wrapper(stretch_and_resize_to_target_dim)
        ds = ds.map(stretch_fn, num_parallel_calls=AUTOTUNE)

        # создание батчей до преобразований, чтоб каждое преобразование применялось один раз ко всему батчу
        ds = ds.batch(batch_size)

        # нормирование на [0, 1]
        scale_fn = io_wrapper()(scale)
        ds = ds.map(scale_fn)

        # яркость
        wrapper = io_wrapper(max_delta_brightness=self.max_delta_brightness)
        brightness_fn = wrapper(random_brightness)
        ds = ds.map(brightness_fn, num_parallel_calls=AUTOTUNE)

        # rotate (if necessary)
        if self.rot90:
            rot_fn = io_wrapper()(rot90)
            ds = ds.map(rot_fn, num_parallel_calls=AUTOTUNE)

        # repeat, prefetch
        ds = ds.repeat()
        ds = ds.prefetch(AUTOTUNE)
        return ds

    def build_valid_dataset(self, examples, batch_size=16):
        """
        Yields the same structures as build_train_dataset
        """
        # взятие необходимого контента из примеров
        # files_img, labels, logits_len, labels_len = map(list, zip(*examples))
        files_img, labels, logits_len, labels_len = [], [], [], []
        for x in examples:
            files_img.append(x.img)
            labels.append(x.char_ids)
            logits_len.append(x.logits_len)
            labels_len.append(len(x.text))

        ds = tf.data.Dataset.from_tensor_slices((files_img, labels, logits_len, labels_len))

        # чтение изображений
        ds = ds.map(lambda *args: read_fn_dev(*args, channels=self.num_channels), num_parallel_calls=AUTOTUNE)

        # батчи
        ds = ds.batch(batch_size)

        # приведение к фиксированной высоте (без изменения пропорций)
        wrapper = io_wrapper(target_height=self.target_height, max_width=self.target_width, max_delta_stretch=0.0)
        stretch_fn = wrapper(stretch_and_cast_to_target_height)
        ds = ds.map(stretch_fn, num_parallel_calls=AUTOTUNE)

        # нормирование на [0, 1]
        scale_fn = io_wrapper()(scale)
        ds = ds.map(scale_fn)

        # padding
        pad_fn = io_wrapper(target_width=self.target_width, max_k_left=0.0)(pad)
        ds = ds.map(pad_fn, num_parallel_calls=AUTOTUNE)

        # rotate (if necessary)
        if self.rot90:
            rot_fn = io_wrapper()(rot90)
            ds = ds.map(rot_fn, num_parallel_calls=AUTOTUNE)

        # prefetch
        ds = ds.prefetch(AUTOTUNE)
        return ds

    def build_test_dataset(self, files_img, batch_size=16):
        """
        Yields tf.Tensor of shape [N, H, W, C] - images
        """
        ds = tf.data.Dataset.from_tensor_slices(files_img)

        # чтение изображений
        read_fn = partial(read_fn_inference, channels=self.num_channels)
        ds = ds.map(read_fn, num_parallel_calls=AUTOTUNE)

        # батчи
        ds = ds.batch(batch_size)

        # приведение к фиксированной высоте (без изменения пропорций)
        stretch_fn = partial(stretch_and_cast_to_target_height, target_height=self.target_height,
                             max_width=self.target_width, max_delta_stretch=0.0)
        ds = ds.map(stretch_fn, num_parallel_calls=AUTOTUNE)

        # нормирование на [0, 1]
        ds = ds.map(scale)

        # padding
        pad_fn = partial(pad, target_width=self.target_width, max_k_left=0.0)
        ds = ds.map(pad_fn, num_parallel_calls=AUTOTUNE)

        # rotate (if necessary)
        if self.rot90:
            ds = ds.map(rot90, num_parallel_calls=AUTOTUNE)

        # prefetch
        ds = ds.prefetch(AUTOTUNE)
        return ds


def read_fn_dev(path, labels, logits_len, labels_len, channels):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=channels)
    x = img, labels, logits_len, labels_len
    y = tf.zeros_like(logits_len)
    # y = None  # в этом случае при обучении вылезет ошибка, что градиенты не удаётся посчитать
    return x, y


def read_fn_inference(path, channels):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=channels)
    return img


def stretch_and_resize_to_target_dim(img, target_height, target_width, max_delta_stretch=0.0, max_k_left=0.0):
    """
    Приведение изображения к целевой высоте, а также растяжение / сжатие по оси x в k раз
    """
    shape_src = tf.shape(img)
    height_src = shape_src[0]
    width_src = shape_src[1]

    # если высота изображения превышает длину в 2 раза, то повернуть на 90 градусов 1 раз
    # и поменять height_src и width_src местами. иначе - ничего не делать
    k = tf.cast(height_src > width_src * 2, tf.int32)
    img = tf.image.rot90(img, k=k)
    height_src_new = width_src * k + height_src * (1 - k)
    width_src = height_src * k + width_src * (1 - k)
    height_src = height_src_new

    height_dst_float = tf.cast(target_height, tf.float32)
    height_src_float = tf.cast(height_src, tf.float32)
    width_src_float = tf.cast(width_src, tf.float32)
    k = tf.random.uniform(shape=(), minval=1.0 - max_delta_stretch, maxval=1.0 + max_delta_stretch)
    width_dst = width_src_float * height_dst_float / height_src_float * k
    width_dst = tf.cast(width_dst, tf.int32)
    width_dst = tf.minimum(width_dst, target_width)

    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [target_height, width_dst])

    k_left = tf.random.uniform(shape=(), minval=0.0, maxval=max_k_left)
    left = tf.cast(tf.cast(target_width - width_dst, tf.float32) * k_left, tf.int32)
    right = target_width - width_dst - left
    # constant_values = 1.0, т.к. изображение уже нормировано на [0, 1]
    img = tf.pad(img, [[0, 0], [left, right], [0, 0]], constant_values=255.0)
    return img


def scale(img):
    return img / 255.0


def random_brightness(img, max_delta_brightness):
    img = tf.image.random_brightness(img, max_delta=max_delta_brightness)
    img = tf.clip_by_value(img, 0, 1)
    return img


def rot90(img):
    return tf.image.rot90(img, k=3)
