from typing import Tuple, Union
import tensorflow as tf
import numpy as np

# костыль, чтоб в образ не засовывать opencv и чтоб в то же время код не падал на предикте
# TODO: написать как-то нормально
try:
    import cv2
except ImportError:  # этот модуль нужен только для аугментаций при обучении
    class cv2:
        BORDER_CONSTANT = 0
        BORDER_REPLICATE = 1


def stretch_and_cast_to_target_height(img, target_height, max_width, max_delta_stretch=0.0):
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
    width_dst = tf.minimum(width_dst, max_width)

    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [target_height, width_dst])
    return img


def scale(img: tf.Tensor) -> tf.Tensor:
    return img / 255.0


def subtract_mean_for_resnet50(img: tf.Tensor) -> tf.Tensor:
    """
    Нормализация для resnet50
    :param img: tf.Tensor of dtype tf.float32 and shape [N, H, W, C]
    :return:
    """
    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
    mean = mean[None, None, None, :]
    img -= mean
    return img


def random_brightness(img: tf.Tensor, max_delta_brightness: float, clip: bool) -> tf.Tensor:
    img = tf.image.random_brightness(img, max_delta=max_delta_brightness)
    if clip:
        img = tf.clip_by_value(img, 0, 1)
    return img


def pad(img: tf.Tensor, target_width: int, max_k_left: float, value: int) -> tf.Tensor:
    k_left = tf.random.uniform(shape=(), minval=0.0, maxval=max_k_left)
    width_src = tf.shape(img)[1]
    left = tf.cast(tf.cast(target_width - width_src, tf.float32) * k_left, tf.int32)
    right = target_width - width_src - left
    # constant_values = 1.0, т.к. изображение уже нормировано на [0, 1]
    image = tf.pad(img, [[0, 0], [left, right], [0, 0]], constant_values=value)
    return image


def erode(img: tf.Tensor) -> tf.Tensor:
    value = tf.cast(img, tf.float32)
    filters = tf.ones((3, 3, 1), dtype=tf.float32)
    img = tf.nn.erosion2d(
        value=value,
        filters=filters,
        strides=[1, 1, 1, 1],
        padding='SAME',
        data_format='NHWC',
        dilations=[1, 1, 1, 1]
    )
    return img


def rot90(img: tf.Tensor) -> tf.Tensor:
    return tf.image.rot90(img, k=3)


def distort_elastic_cv2(
        img: Union[tf.Tensor, np.ndarray],
        alpha: int = 80,
        sigma: int = 20,
        num_channels: int = 1,
        random_state: int = None
) -> np.ndarray:
    """
    Elastic deformation of images as per [Simard2003].
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    h = img.shape[0]
    w = img.shape[1]

    # Downscaling the random grid and then upsizing post filter
    # improves performance. Approx 3x for scale of 4, diminishing returns after.
    grid_scale = 4
    alpha //= grid_scale  # Does scaling these make sense? seems to provide
    sigma //= grid_scale  # more similar end result when scaling grid used.
    grid_shape = (h // grid_scale, w // grid_scale)

    blur_size = int(4 * sigma) | 1
    rand_x = cv2.GaussianBlur(
        (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
        ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    rand_y = cv2.GaussianBlur(
        (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
        ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    if grid_scale > 1:
        rand_x = cv2.resize(rand_x, (w, h))
        rand_y = cv2.resize(rand_y, (w, h))

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_x = (grid_x + rand_x).astype(np.float32)
    grid_y = (grid_y + rand_y).astype(np.float32)

    img = img.numpy() if isinstance(img, tf.Tensor) else img
    distorted_img = cv2.remap(img, grid_x, grid_y, borderMode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR)

    if num_channels == 1:
        distorted_img = distorted_img[:, :, None]

    return distorted_img


def random_shear_img(
        img: Union[tf.Tensor, np.ndarray],
        x_range: Tuple[float, float] = (-0.1, 0.1),
        y_range: Tuple[float, float] = (0.0, 0.0),
        num_channels: int = 1,
        border_mode: int = cv2.BORDER_CONSTANT,
        border_value: int = 0
) -> np.ndarray:
    """
    https://www.thepythoncode.com/article/image-transformations-using-opencv-in-python#Shearing_yaxis
    """
    h = img.shape[0]
    w = img.shape[1]
    random_state = np.random.RandomState(None)

    shx = random_state.uniform(*x_range)
    shy = random_state.uniform(*y_range)

    h_new = int(h * (1 + abs(shy)))
    w_new = int(w * (1 + abs(shx)))

    up_down = (h_new - h) // 2
    left_right = (w_new - w) // 2

    img = img.numpy() if isinstance(img, tf.Tensor) else img
    img = cv2.copyMakeBorder(img, up_down, up_down, left_right, left_right, borderType=border_mode, value=border_value)

    M = np.float32([
        [1, shx, 0],
        [shy, 1, 0],
    ])

    img = cv2.warpAffine(img, M, (w_new, h_new), borderMode=border_mode, borderValue=border_value)

    if num_channels == 1:
        img = img[..., None]

    return img


def random_rotate_img(
        img: Union[tf.Tensor, np.ndarray],
        angle_range: Tuple[int, int] = (-5, 5),
        num_channels: int = 1,
        border_mode: int = cv2.BORDER_REPLICATE,
        border_value: int = 0
) -> np.ndarray:
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_BorderTypes.html
    """

    h = img.shape[0]
    w = img.shape[1]
    random_state = np.random.RandomState(None)

    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (w / 2, h / 2)

    angle = random_state.uniform(*angle_range)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    img = img.numpy() if isinstance(img, tf.Tensor) else img
    rotated_mat = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h), borderMode=border_mode, borderValue=border_value)

    if num_channels == 1:
        rotated_mat = rotated_mat[..., None]

    return rotated_mat
