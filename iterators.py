from typing import Tuple, List
import random
from abc import ABC, abstractmethod
import tensorflow as tf

from utils import LMDirection


class ExamplesIterator(ABC):
    def __init__(self, examples):
        self.examples = examples

    @abstractmethod
    def __iter__(self): ...

    @property
    @abstractmethod
    def output_types(self): ...

    @property
    @abstractmethod
    def output_shapes(self): ...


class HTRIterator(ExamplesIterator):
    def __init__(self, examples, test_mode: bool):
        super().__init__(examples)
        self.test_mode = test_mode  # train или valid

    def __iter__(self):
        for example in self.examples:
            if self.test_mode:
                x = example.img
                yield x
            else:
                x = example.img, example.char_ids, example.logits_len, len(example.text)
                y = tf.zeros_like(example.logits_len)
                yield x, y

    @property
    def output_types(self):
        if self.test_mode:
            return tf.string
        else:
            return tf.string, tf.int32, tf.int32, tf.int32

    @property
    def output_shapes(self):
        if self.test_mode:
            return tf.TensorShape([])
        else:
            return tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([])


class LMIterator(ExamplesIterator):
    BOS = '<bos>'
    EOS = '<eos>'
    UNK = '<unk>'

    def __init__(
            self,
            examples,
            direction: str,
            char2id: dict,
            add_seq_borders: bool,
            training: bool,
            p_aug: float = 0.5,
            min_seq_len: int = 1,
            pad: bool = False,
            maxlen: int = 100
    ):
        super().__init__(examples)
        assert direction in {LMDirection.FORWARD, LMDirection.BIDIRECTIONAL}
        self.direction = direction
        self.char2id = char2id
        self.add_seq_borders = add_seq_borders
        self.training = training
        self.p_aug = p_aug
        self.min_seq_len = min_seq_len
        self.pad = pad
        self.maxlen = maxlen

    def __iter__(self):
        if self.training:
            while True:
                x = random.choice(self.examples)
                text = x.text
                if random.random() < self.p_aug:
                    tokens = text.split()
                    i = random.randint(0, len(tokens) - self.min_seq_len)  # TODO: может быть меньше нуля!
                    j = random.randint(i + self.min_seq_len, len(tokens))
                    text = ' '.join(tokens[i:j])
                yield self._text2inputs(text)
        else:
            for x in self.examples:
                yield self._text2inputs(x.text)

    @property
    def output_types(self):
        return tf.int32, tf.int32

    @property
    def output_shapes(self):
        return tf.TensorShape([None]), tf.TensorShape([None])

    def _text2inputs(self, text: str) -> Tuple[List[int], List[int]]:
        if self.direction == LMDirection.FORWARD:
            x = [self._get_char_id(x) for x in text[:-1]]
            y = [self._get_char_id(x) for x in text[1:]]
            if self.add_seq_borders:
                x = [self.char2id[self.BOS]] + x + [self._get_char_id(text[-1])]
                y = [self._get_char_id(text[0])] + y + [self.char2id[self.EOS]]
            if self.pad:
                x += [0] * (self.maxlen - len(x))
                y += [0] * (self.maxlen - len(y))
        elif self.direction == LMDirection.BIDIRECTIONAL:
            y = [self._get_char_id(x) for x in text]
            x = [self.char2id[self.BOS]] + y + [self.char2id[self.EOS]]
            if self.pad:
                x += [0] * (self.maxlen - len(x))
                y += [0] * (self.maxlen - len(y) - 2)
        else:
            raise ValueError(f'expected direction "{LMDirection.FORWARD}" or "{LMDirection.BIDIRECTIONAL}", '
                             f'got {self.direction}')

        return x, y

    def _get_char_id(self, char: str) -> int:
        return self.char2id.get(char, self.char2id[self.UNK])


class JointIterator(LMIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        for x in self.examples:
            char_ids_x, char_ids_y = self._text2inputs(x.text)
            yield x.img, char_ids_x, char_ids_y

    @property
    def output_types(self):
        return tf.string, tf.int32, tf.int32

    @property
    def output_shapes(self):
        return tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([None])
