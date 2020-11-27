import os
import random
import json
import re
from collections import namedtuple, defaultdict
from typing import List, Tuple, Dict
from functools import wraps
from datetime import datetime

import tensorflow as tf
import editdistance


class Example:
    def __init__(
            self,
            name: str = None,
            img: str = None,
            text: str = None,
            char_ids: List[int] = None,
            logits_len: int = None,
            augmentations: List[str] = None,
            p_aug: float = 0.9,
            candidates: List[str] = None
    ):
        self.name = name
        self._img = img
        self.text = text
        self.char_ids = char_ids
        self.logits_len = logits_len
        self.augmentations = augmentations
        self.p_aug = p_aug
        self.candidates = candidates

    @property
    def img(self) -> str:
        """путь к картинке"""
        if self.augmentations is None:
            return self._img
        else:
            if random.random() < self.p_aug:
                return random.choice(self.augmentations)
            else:
                return self._img


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)  # встречаются лишние пробелы
    text = text.strip()  # удаление пробелов по краям
    text = re.sub(r'\bps\b|\bрs\b|\bр s\b', 'p s', text)  # 1) <p латинская>s, 2) <р русская>s, 3) <р русская> s -> <p латинская> s
    text = text.replace('c', 'с')  # латинская -> русская
    text = text.replace('і', 'i')  # кирилическая -> латинская
    return text


def split_dataset(input_dir: str, output_dir: str):
    """
    Получение единого разбиения на {train, valid, test} для всех экспериментов
    """
    images_dir = os.path.join(input_dir, "images")
    img_names = {x.split(".")[0] for x in os.listdir(images_dir)}
    print("num images:", len(img_names))

    texts_dir = os.path.join(input_dir, "words")
    text_names = {x.split(".")[0] for x in os.listdir(texts_dir)}
    print("num texts:", len(text_names))

    names = list(img_names & text_names)
    print("num common names:", len(names))

    n_total = len(names)
    rng = random.Random(228)
    indices = rng.sample(list(range(n_total)), n_total)
    n_train = int(n_total * 0.7)
    n_valid = int(n_total * 0.1)
    print("train size:", n_train)
    print("valid size:", n_valid)
    print("test_size:", n_total - n_train - n_valid)

    with open(os.path.join(output_dir, "train_names.txt"), "w") as f:
        for i in indices[:n_train]:
            f.write(names[i] + "\n")

    with open(os.path.join(output_dir, "valid_names.txt"), "w") as f:
        for i in indices[n_train:n_train + n_valid]:
            f.write(names[i] + "\n")

    with open(os.path.join(output_dir, "test_names.txt"), "w") as f:
        for i in indices[n_train + n_valid:]:
            f.write(names[i] + "\n")


def load_examples(data_dir: str, partitions_dir: str, logits_len: int) -> Tuple[List[Example], List[Example], List[Example]]:
    examples_train = []
    examples_valid = []
    examples_test = []
    partitions = ["train", "valid", "test"]
    for part in partitions:
        with open(os.path.join(partitions_dir, f"{part}_names.txt")) as f:
            for name in f:
                name = name.strip()
                text = open(os.path.join(data_dir, "words", name + ".txt")).readline()
                text = clean_text(text)
                x = Example(
                    name=name,
                    img=os.path.join(data_dir, "images", name + ".jpg"),
                    text=text,
                    logits_len=logits_len
                )
                if part == "train":
                    examples_train.append(x)
                elif part == "valid":
                    examples_valid.append(x)
                elif part == "test":
                    examples_test.append(x)
    return examples_train, examples_valid, examples_test


HTRMetrics = namedtuple("HTRMetrics", ["cer", "wer", "string_acc"])


def evaluate(true_texts: List[str], pred_texts: List[str], top_k: int = 10) -> HTRMetrics:
    numCharErr = 0
    numCharTotal = 0
    numStringOK = 0
    numStringTotal = 0

    word_eds, word_true_lens = [], []

    verbose = top_k > 0

    if verbose:
        print('Ground truth -> Recognized')

    for i, (true, pred) in enumerate(zip(true_texts, pred_texts)):
        numStringOK += 1 if true == pred else 0
        numStringTotal += 1
        dist = editdistance.eval(pred, true)
        numCharErr += dist
        numCharTotal += len(true)

        pred_words = pred.split()
        true_words = true.split()
        word_eds.append(editdistance.eval(pred_words, true_words))
        word_true_lens.append(len(true_words))

        if verbose and i < top_k:
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + true + '"', '->', '"' + pred + '"')

    charErrorRate = numCharErr / numCharTotal
    wordErrorRate = sum(word_eds) / sum(word_true_lens)
    stringAccuracy = numStringOK / numStringTotal
    if verbose:
        print('cer: %f%%; wer: %f%%; string_acc: %f%%.' % \
              (charErrorRate * 100.0, wordErrorRate * 100.0, stringAccuracy * 100.0))
    return HTRMetrics(cer=charErrorRate, wer=wordErrorRate, string_acc=stringAccuracy)


class CharEncoder:
    PAD_ID = 0
    UNK = '@'

    def __init__(
            self,
            char2id: Dict[str, int] = None,
            maxlen: int = 80,
            min_freq: int = 2,
            pad: bool = True
    ):
        """
        min_freq - если символ встречается меньше min_freq раз в обучающей выборке,
        то ему присваивается символ UNK.
        """
        self.maxlen = maxlen
        self.min_freq = min_freq
        self.pad = pad

        self._char2id = char2id
        if char2id is not None:
            self._id2char = {v: k for k, v in char2id.items()}

    def fit_transform(self, examples: List[Example]):
        self.fit(examples)
        self.transform(examples)

    def fit(self, examples: List[Example]):
        chars = set()
        counts = defaultdict(int)
        for x in examples:
            for c in x.text:
                chars.add(c)
                counts[c] += 1

        self._id2char = {0: '¶'}
        idx = 1  # 0 зарезервирован под PAD
        for c in chars:
            if counts[c] >= self.min_freq:
                self._id2char[idx] = c
                idx += 1
        self._id2char[idx] = self.UNK

        self._char2id = {v: k for k, v in self._id2char.items()}

    def transform(self, examples: List[Example]):
        for x in examples:
            x.char_ids = self.encode(x.text)

    def encode(self, line: str) -> List[int]:
        id_unk = self.get_id(self.UNK)
        char_ids = [self._char2id.get(c, id_unk) for c in line]
        if self.pad:
            char_ids += [self.PAD_ID] * (self.maxlen - len(line))
        return char_ids

    def decode(self, char_ids: List[int]) -> str:
        # 0 - pad
        # -1 - blank
        return ''.join(self._id2char[i] for i in char_ids if i > 0)

    def get_id(self, char: str) -> int:
        return self._char2id[char]

    def get_char(self, idx: int) -> str:
        return self._id2char[idx]

    @property
    def vocab_size(self) -> int:
        return len(self._char2id)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self._char2id, f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            char2id = json.load(f)
        return cls(char2id=char2id)


def log(func):
    @wraps(func)
    def logged(self, *args, **kwargs):
        if self.verbose:
            print(f"{func.__name__} started.")
        t0 = datetime.now()
        res = func(self, *args, **kwargs)
        time_elapsed = datetime.now() - t0
        if self.verbose:
            print(f"{func.__name__} finished. Time elapsed: {time_elapsed}")
        return res
    return logged


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps):
        super().__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps
        }


class LMDirection:
    FORWARD = "fw"
    BIDIRECTIONAL = "bi"


def ce_masked_loss(y_true, y_pred):
    """
    кастомный лосс, т.к. по временному измерению входной и
    выходной тензоры не совпадают в случае двусторонней языковой модели
    """
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
    loss = loss_obj(y_true, y_pred)
    mask = tf.cast(tf.math.not_equal(y_true, 0), tf.float32)
    loss *= mask
    loss_mean = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss_mean


def get_optimizer(noam_scheme=True, d_model=None, warmup_steps=4000, lr=1e-3):
    if noam_scheme:
        learning_rate = CustomSchedule(d_model=d_model, warmup_steps=warmup_steps)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9
        )
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    return optimizer


def get_seq_scores(
        prob: tf.Tensor,
        char_ids_next: tf.Tensor,
        eos: bool = False,
        pad_id: int = 0,
        eos_id: int = None
) -> tf.Tensor:
    """
    Логарифмические вероятности последовательностей
    :param prob: tf.Tensor of type tf.float32 and shape [N, T, V]
    :param char_ids_next: tf.Tensor of type tf.int32 as shape [N, T]
    :param eos: bool, нужно ли учитывать вероятность символа окончания последовательности
    :param pad_id: int, индекс pad символа
    :param eos_id: int, индекс символа окончания последовательности
    :return:
    """
    vocab_size = tf.shape(prob)[-1]
    y_oh = tf.one_hot(char_ids_next, vocab_size)
    prob_oh = prob * y_oh
    prob_target = tf.reduce_sum(prob_oh, axis=-1)
    log_prob_target = tf.math.log(prob_target)
    log_prob_target *= tf.cast(tf.math.not_equal(char_ids_next, pad_id), tf.float32)
    if not eos:
        log_prob_target *= tf.cast(tf.math.not_equal(char_ids_next, eos_id), tf.float32)
    seq_scores = tf.reduce_sum(log_prob_target, axis=1)
    return seq_scores
