import os
import random
import cv2
from matplotlib import pyplot as plt


class Example:
    img_format = 'jpg'
    text_format = 'txt'

    def __init__(self, name=None, img=None, text=None):
        self.name = name
        self.img = img
        self.text = text

        self.char_ids = None
        self.img_processed = None

    def __repr__(self):
        class_name = self.__class__.__name__
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items() if k != 'img')
        return f'{class_name}({params})'

    @classmethod
    def load(cls, name, img_dir, text_dir):
        img = plt.imread(os.path.join(img_dir, name + "." + cls.img_format))
        text = open(os.path.join(text_dir, name + "." + cls.text_format), encoding='utf8').readline()
        return cls(name=name, img=img, text=text)


class Tokenizer:
    def __init__(self):
        pass

    def encode(self, text):
        pass

    def decode(self, char_ids):
        pass


def load_examples(data_dir, n=None):
    img_dir = os.path.join(data_dir, "images")
    text_dir = os.path.join(data_dir, "words")
    names = [x.split(".")[0] for x in os.listdir(text_dir)[:n]]
    examples = [Example.load(name, img_dir=img_dir, text_dir=text_dir) for name in tqdm.tqdm(names)]
    return examples


def rotate_image(img, angle):
    """
    бывает так, что изображения в выборке могут быть повёрнуты на 90 градусов по часовой стрелке:
    67 / 6196 изображений обучающей выборки таковы, что высота в 2 раза превышает длину, что странно.
    очевидно, что на таких объектах модель будет предсказывать полный рандом.
    https://stackoverflow.com/a/37347070
    """
    # angle in degrees

    h, w, _ = img.shape
    image_center = w / 2, h / 2

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    rotated_mat = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def aug_image(img):
    """
    https://github.com/githubharald/SimpleHTR/blob/master/src/SamplePreprocessor.py#L17
    растяжение / сжатие по оси X
    """
    h = img.shape[0]
    w = img.shape[1]
    stretch = random.random() - 0.5  # -0.5 .. +0.5
    wStretched = max(int(w * (1 + stretch)), 1)  # random width, but at least 1
    img = cv2.resize(img, (wStretched, h))  # stretch horizontally by factor 0.5 .. 1.5
    return img


def process_image_v1(img, aug=False):
    """
    * если высота гораздо больше ширины (напирмер, в 2 раза), то повернуть изображение на 90 градусов влево
    * сделать все изображения высоты 128 и ширины не больше 1024
    * нормировать на [-1, 1]
    """
    if img.shape[0] > img.shape[1] * 2:
        img = rotate_image(img, 90)

    if aug:
        img = aug_image(img)

    w, h, _ = img.shape

    new_w = 128
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h, _ = img.shape

    img = img.astype('float32')

    if w < 128:
        add_zeros = np.full((128 - w, h, 3), 255)
        img = np.concatenate((img, add_zeros))
        w, h, _ = img.shape

    if h < 1024:
        add_zeros = np.full((w, 1024 - h, 3), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h, _ = img.shape

    if h > 1024 or w > 128:
        dim = (1024, 128)
        img = cv2.resize(img, dim)

    img = cv2.subtract(255, img)

    img = img / 255

    return img


def process_image_v2(img, aug=False):
    h, w, _ = img.shape

    # rotation. было замечено, что некоторые изображение поставлены на бок.
    # их достаточно повернуть на 90 градусов против часовой стрелки
    if h > w * 2:
        img = rotate_img(img, 90)

    if aug:
        img = aug_image(img)

    # binarization. по итогу будет один канал
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    T = threshold_yen(img, nbins=256)
    img = (img > T).astype("uint8") * 255

    # erosion. нужно разобраться
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)

    img = img.astype('float32')

    # resize
    ht = 128
    wt = 1024
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    w_new = max(min(wt, int(w / f)), 1)
    h_new = max(min(ht, int(h / f)), 1)
    img = cv2.resize(img, (w_new, h_new))
    target = np.full((ht, wt), 255)
    target[0:h_new, 0:w_new] = img
    img = target

    # нормирование
    m, s = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img

    return img
