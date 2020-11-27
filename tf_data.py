from functools import partial
from typing import Union

import tensorflow as tf

from iterators import HTRIterator, JointIterator, LMIterator
from image import (
    random_shear_img,
    random_brightness,
    random_rotate_img,
    rot90,
    erode,
    stretch_and_cast_to_target_height,
    distort_elastic_cv2,
    scale,
    pad
)


AUTOTUNE = tf.data.experimental.AUTOTUNE


def io_wrapper(**kwargs):
    """
    При обучении и валидации каждая функция обработка датасета должна
    шарить общий интерфейс: (x, y) -> f -> (x, y).
    В HTR x - это:
    * картинка - для энкодера
    * символы (включая pad) - для ctc_loss
    * число символов (не включая pad) - для ctc_loss
    * число логитов - для ctc_loss
    В HTR + LM x - это:
    * картинка - для энкодера
    * символы - для декодера
    При инференсе нам нужны только изображения, следовательно интерфейс функций обработки
    датасета следующий: img -> f -> img.
    Данный декоратор требуется для использования функций-обработчиков изображений
    в режимах обучения и валидации с использоваванием tf.data.Dataset API.
    """
    def wrapper(func):
        def wrapped(x, y):
            img = x[0]
            img = func(img, **kwargs)
            x = (img, *x[1:])
            return x, y
        return wrapped
    return wrapper


def py_func_wrapper(img, func, **kwargs):
    """
    Каждая питоновская функция обработки изображений должна быть обёрнута
    в tf.py_function, чтобы быть операцией в вычислительном графе
    """
    img = tf.py_function(partial(func, **kwargs), [img], Tout=tf.float32)
    return img


class DataPipelineBuilderHTR:
    """
    Пример конфигурации аугментаций:

    augmentations = {
        'width_translation': {
            'enabled': True,
            'max_k_left': 1.0
        },
        'elastic': {
            'enabled': True,
            'alpha': 750,
            'sigma': 20
        },
        'brightness': {
            'enabled': True,
            'max_delta': 0.1
        },
        'stretching': {
            'enabled': True,
            'max_delta': 0.3
        },
        'shearing': {
            'enabled': False,
            'x_range': (-0.3, 0.3),
            'y_range': (0, 0),
            'border_mode': cv2.BORDER_CONSTANT,
            'border_value: 0'
        },
        'rotation': {
            'enabled': False,
            'angle_range': (-3, 3),
            'border_mode': cv2.BORDER_CONSTANT,
            'border_value: 0'
        }
    }
    """
    def __init__(
            self,
            num_channels=1,
            target_height=128,
            target_width=1024,
            resnet50=False,
            rot90=True,
            erode=False,
            augmentations: dict = None
    ):
        self.num_channels = num_channels
        self.target_height = target_height
        self.target_width = target_width
        self.resnet50 = resnet50
        self.rot90 = rot90
        self.erode = erode
        self.augmentations = augmentations

        self.scale = not resnet50

    def build_train_dataset(
            self, it: Union[HTRIterator, JointIterator],
            batch_size: int = 16,
            buffer: int = 100,
            drop_remainder: bool = False
    ) -> tf.data.Dataset:
        """
        Yields tuples:
        * img - tf.Tensor of shape [N, H, W, C] - images
        * labels - tf.Tensor of shape [N, T] - captions
        * logits_len - tf.Tensor of shape [N] - number of frames from image
        * labels_len - tf.Tensor of shape [N] - number of non-pad elements in labels
        """
        # чтение примеров
        ds = tf.data.Dataset.from_generator(
            lambda: iter(it),
            output_types=it.output_types,
            output_shapes=it.output_shapes
        )

        # перемешивание
        ds = ds.shuffle(buffer)

        # чтение изображений
        ds = ds.map(lambda *args: read_fn_joint_dev(*args, channels=self.num_channels), num_parallel_calls=AUTOTUNE)

        # shearing
        if self._is_enabled("shearing"):
            wrapper = io_wrapper(
                func=random_shear_img,
                x_range=self.augmentations["shearing"]["x_range"],
                y_range=self.augmentations["shearing"]["y_range"],
                num_channels=self.num_channels,
                border_mode=self.augmentations["shearing"]["border_mode"],
                border_value=self.augmentations["shearing"]["border_value"]
            )
            shear_fn = wrapper(py_func_wrapper)
            ds = ds.map(shear_fn, num_parallel_calls=AUTOTUNE)

        # random rotate
        if self._is_enabled("rotation"):
            wrapper = io_wrapper(
                func=random_rotate_img,
                angle_range=self.augmentations["rotation"]["angle_range"],
                num_channels=self.num_channels,
                border_mode=self.augmentations["rotation"]["border_mode"],
                border_value=self.augmentations["rotation"]["border_value"]
            )
            rotation_fn = wrapper(py_func_wrapper)
            ds = ds.map(rotation_fn, num_parallel_calls=AUTOTUNE)

        # приведение к фиксированной высоте (с изменением пропорций)
        if self.augmentations is not None:
            if self.augmentations["stretching"]["enabled"]:
                max_delta_stretch = self.augmentations["stretching"]["max_delta"]
            else:
                max_delta_stretch = 0.0
        else:
            max_delta_stretch = 0.0
        wrapper = io_wrapper(
            target_height=self.target_height,
            max_width=self.target_width,
            max_delta_stretch=max_delta_stretch
        )
        stretch_fn = wrapper(stretch_and_cast_to_target_height)
        ds = ds.map(stretch_fn, num_parallel_calls=AUTOTUNE)

        # нормирование на [0, 1]
        if self.scale:
            scale_fn = io_wrapper()(scale)
            ds = ds.map(scale_fn)

        # яркость
        if self._is_enabled('brightness'):
            wrapper = io_wrapper(max_delta_brightness=self.augmentations["brightness"]["max_delta"], clip=self.scale)
            brightness_fn = wrapper(random_brightness)
            ds = ds.map(brightness_fn, num_parallel_calls=AUTOTUNE)

        # distort elastic
        if self._is_enabled('elastic'):
            wrapper = io_wrapper(
                func=distort_elastic_cv2,
                alpha=self.augmentations["elastic"]["alpha"],
                sigma=self.augmentations["elastic"]["sigma"],
                num_channels=self.num_channels
            )
            distort_fn = wrapper(py_func_wrapper)
            ds = ds.map(distort_fn, num_parallel_calls=AUTOTUNE)

        # padding
        if self.augmentations is not None:
            if self.augmentations["width_translation"]["enabled"]:
                max_k_left = self.augmentations["width_translation"]["max_k_left"]
            else:
                max_k_left = 0.0
        else:
            max_k_left = 0.0
        if self.scale:
            value = 1.0
        else:
            value = 255.0
        pad_fn = io_wrapper(target_width=self.target_width, max_k_left=max_k_left, value=value)(pad)
        ds = ds.map(pad_fn, num_parallel_calls=AUTOTUNE)

        # rotate (if necessary)
        if self.rot90:
            rot_fn = io_wrapper()(rot90)
            ds = ds.map(rot_fn, num_parallel_calls=AUTOTUNE)

        # batch
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)

        # erode (only batch-level)
        if self.erode:
            erode_fn = io_wrapper()(erode)
            ds = ds.map(erode_fn, num_parallel_calls=AUTOTUNE)

        # prefetch
        ds = ds.repeat()
        ds = ds.prefetch(AUTOTUNE)
        return ds

    def build_valid_dataset(
            self,
            it: Union[HTRIterator, JointIterator],
            batch_size: int = 16,
            drop_remainder: bool = False
    ) -> tf.data.Dataset:
        """
        Yields the same structures as build_train_dataset
        """
        # чтение примеров
        ds = tf.data.Dataset.from_generator(
            lambda: iter(it),
            output_types=it.output_types,
            output_shapes=it.output_shapes
        )

        # чтение изображений
        ds = ds.map(lambda *args: read_fn_joint_dev(*args, channels=self.num_channels), num_parallel_calls=AUTOTUNE)

        # приведение к фиксированной высоте (без изменения пропорций)
        wrapper = io_wrapper(target_height=self.target_height, max_width=self.target_width, max_delta_stretch=0.0)
        stretch_fn = wrapper(stretch_and_cast_to_target_height)
        ds = ds.map(stretch_fn, num_parallel_calls=AUTOTUNE)

        # нормирование на [0, 1]
        if self.scale:
            scale_fn = io_wrapper()(scale)
            ds = ds.map(scale_fn)

        # padding
        if self.scale:
            value = 1.0
        else:
            value = 255.0
        pad_fn = io_wrapper(target_width=self.target_width, max_k_left=0.0, value=value)(pad)
        ds = ds.map(pad_fn, num_parallel_calls=AUTOTUNE)

        # rotate (if necessary)
        if self.rot90:
            rot_fn = io_wrapper()(rot90)
            ds = ds.map(rot_fn, num_parallel_calls=AUTOTUNE)

        # batch, prefetch
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)

        # erode (only batch-level)
        if self.erode:
            erode_fn = io_wrapper()(erode)
            ds = ds.map(erode_fn, num_parallel_calls=AUTOTUNE)

        ds = ds.prefetch(AUTOTUNE)
        return ds

    def build_test_dataset(
            self, it: Union[HTRIterator, JointIterator],
            batch_size: int = 16
    ) -> tf.data.Dataset:
        """
        Yields tf.Tensor of shape [N, H, W, C] - images
        """
        ds = tf.data.Dataset.from_generator(
            lambda: iter(it),
            output_types=it.output_types,
            output_shapes=it.output_shapes
        )

        # чтение изображений
        read_fn = partial(read_fn_htr_inference, channels=self.num_channels)
        ds = ds.map(read_fn, num_parallel_calls=AUTOTUNE)

        # приведение к фиксированной высоте (без изменения пропорций)
        stretch_fn = partial(stretch_and_cast_to_target_height, target_height=self.target_height,
                             max_width=self.target_width, max_delta_stretch=0.0)
        ds = ds.map(stretch_fn, num_parallel_calls=AUTOTUNE)

        # нормирование на [0, 1]
        if self.scale:
            ds = ds.map(scale)

        # padding
        if self.scale:
            value = 1.0
        else:
            value = 255.0
        pad_fn = partial(pad, target_width=self.target_width, max_k_left=0.0, value=value)
        ds = ds.map(pad_fn, num_parallel_calls=AUTOTUNE)

        # rotate (if necessary)
        if self.rot90:
            ds = ds.map(rot90, num_parallel_calls=AUTOTUNE)

        # batch
        ds = ds.batch(batch_size)

        # erode
        if self.erode:
            ds = ds.map(erode, num_parallel_calls=AUTOTUNE)

        # prefetch
        ds = ds.prefetch(AUTOTUNE)
        return ds

    def _is_enabled(self, aug_name):
        return (self.augmentations is not None) and self.augmentations[aug_name]['enabled']


class DataPipelineBuilderLM:
    @staticmethod
    def build_train_dataset(
            it: LMIterator,
            batch_size: int,
            buffer: int = 1000,
            drop_remainder: bool = False
    ) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_generator(lambda: iter(it), output_types=it.output_types)
        ds = ds.shuffle(buffer)
        ds = ds.repeat()
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    @staticmethod
    def build_valid_dataset(it: LMIterator, batch_size: int, drop_remainder: bool = False) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_generator(lambda: iter(it), output_types=it.output_types)
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    def build_test_dataset(self, it: LMIterator, batch_size: int, drop_remainder: bool = False) -> tf.data.Dataset:
        return self.build_valid_dataset(it=it, batch_size=batch_size, drop_remainder=drop_remainder)


def build_joint_decoder_inference_ds(it: JointIterator, batch_size: int) -> tf.data.Dataset:
    """
    Yields tuples:
    * img_path
    * char_ids
    * char_ids_next
    """
    ds = tf.data.Dataset.from_generator(lambda: iter(it), output_types=it.output_types, output_shapes=it.output_shapes)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def read_fn_joint_dev(path, char_ids, next_char_ids, channels):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=channels)
    x = img, char_ids
    y = next_char_ids
    return x, y


def read_fn_joint_inference(path, char_ids, next_char_ids, channels):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=channels)
    x = img, char_ids
    return x


def read_fn_htr_inference(path, channels):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=channels)
    return img
