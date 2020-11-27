import sys
from typing import List, Dict, Type, Tuple
from itertools import chain
from functools import partial
from multiprocessing import Pool

import numpy as np
import kenlm

from utils import Example
from models import BaseLanguageModel
sys.path.append('/app')  # если запускать в контейнере, то в этой папке лежит скомпиленный код для декодирования
from ctc_pyx.ctc import ctc_beam_search, BeamState


def lm_score(
        candidates: List[List[str]],
        log_prob_ctc: np.ndarray,
        lm: Type[BaseLanguageModel],
        ctc_weight: float
) -> List[str]:
    """
    Псевдокод:
    [[foo1, bar1], [foo2, bar2], [foo3, bar3]] ->
    [[foo1, foo2, foo3], [bar1, bar2, bar3]] ->
    [foo1, foo2, foo3, bar1, bar2, bar3] ->
    lm.predict ->
    [p(foo1), p(foo2), p(foo3), p(bar1), p(bar2), p(bar3)] ->
    [[p(foo1), p(foo2), p(foo3)], [p(bar1), p(bar2), p(bar3)]] (log_prob_lm) ->
    scores = log_prob_ctc * k_ctc + log_prob_lm * (1 - k_ctc)

    * Т.к. кандидаты, полученные от ctc могут быть произвольными, возможен случай одного пробельного символа.
    * Т.к. мы удаляем лишние пробелы, данные последовательности препащаются в пустые, то приводит к падению
    языковой модели.
    * Поэтому для получения вероятностей последовательности от языковой модели заменим пустые строки
    на какую-то маловероятную, чтоб модель выдавала низкие вероятности на таких последовательностях
    """
    top_paths = len(candidates)
    improbable_seq = '000000'

    def process_input_text(text):
        if text == "":
            return improbable_seq
        return text

    def process_output_text(text):
        if text == improbable_seq:
            return ""
        return text

    candidates_flat = list(map(process_input_text, chain(*zip(*candidates))))
    examples = [Example(text=text) for text in candidates_flat]
    log_prob_lm = lm.predict(examples)  # [num_examples * top_paths]
    log_prob_lm = log_prob_lm.reshape((-1, top_paths))
    scores = log_prob_ctc * ctc_weight + log_prob_lm * (1 - ctc_weight)  # [num_examples, top_paths]
    indices = scores.argmax(1)  # [num_examples]
    best_candidates = list(map(process_output_text, (
        candidates_flat[top_paths * id_example + id_path] for id_example, id_path in enumerate(indices)
    )))
    return best_candidates


def get_beam_states(char_prob: np.ndarray, **kwargs) -> List[BeamState]:
    f = partial(ctc_beam_search, **kwargs)
    with Pool() as p:
        states = p.map(f, char_prob)
    return states


def get_ctc_prob_and_candidates(
        states: List[BeamState],
        beam_width: int,
        id2char: Dict[int, str]
) -> Tuple[List[List[str]], np.ndarray]:
    candidates = []
    log_prob_ctc = []
    for state in states:
        ci = []
        si = []
        for entry in state.sorted_entries[:beam_width]:
            text = ''.join(id2char[i] for i in entry.labeling)
            ci.append(text)
            si.append(entry.score)
        ci = ci + [ci[-1]] * (beam_width - len(ci))
        si = si + [si[-1]] * (beam_width - len(si))
        candidates.append(ci)
        log_prob_ctc.append(si)
    log_prob_ctc = np.array(log_prob_ctc)
    log_prob_ctc = np.log(log_prob_ctc + 1e-100)
    return candidates, log_prob_ctc


class LanguageModelKen:
    def __init__(self, lm: kenlm.Model):
        self.lm = lm

    def __call__(self, text: str) -> float:
        text_processed = self.process_text(text)
        p_log = self.lm.score(text_processed, bos=True, eos=False)
        return 10 ** p_log

    @staticmethod
    def process_text(text: str) -> str:
        return ' '.join(text.replace(' ', '|'))
