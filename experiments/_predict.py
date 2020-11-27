import os
import glob
import json
import re
import argparse
from itertools import chain

import kenlm
import tensorflow as tf

import sys; sys.path.insert(0, "./src")
import models
from utils import Example, CharEncoder, save_predictions, clean_text
from postprocessing import LanguageModelKen, get_beam_states, get_ctc_prob_and_candidates


INPUT_DIR = '/data'
OUTPUT_DIR = '/output'

# BEAM_WIDTH = 10
# BEAM_WIDTH = 25
BEAM_WIDTH = 30
# BEAM_WIDTH = 100
# BEAM_WIDTH = 500

ALPHA = 0.6
BETA = 4
W_CTC = 70
W_BIRNN = 30

IMPROBABLE_SEQ = '000000'


def build_and_restore(model_cls, model_dir):
    tf.keras.backend.clear_session()
    model = model_cls.load(model_dir)
    model.build()
    model.restore(model_dir)
    return model


def process_input_text(text):
    text = clean_text(text)
    if text == "":
        return IMPROBABLE_SEQ
    return text


def process_output_text(text):
    if text == IMPROBABLE_SEQ:
        return ""
    return text


def main():
    # load images
    images = glob.glob(os.path.join(INPUT_DIR, "*"))

    # build htr model
    htr_model = build_and_restore(model_cls=models.HTRModel, model_dir="./htr")

    # inference of htr model
    examples = [Example(img=img) for img in images]
    char_prob = htr_model.predict_v2(examples, batch_size=64)

    # ctc decoding with prefix beam search and kneser-ney smoothing LM
    id2char = {v: k for k, v in htr_model.char2id.items()}

    # kenlm
    kenlm_path = "./lm/kneser-ney/old_slavic_texts_wo_petr_letters_char_level_5.arpa"
    lm = LanguageModelKen(lm=kenlm.Model(kenlm_path))

    states = get_beam_states(
        char_prob,
        classes=id2char,
        lm=lm,
        beam_width=BEAM_WIDTH,
        alpha=ALPHA,
        beta=BETA,
        min_char_prob=0.001
    )
    candidates, log_prob_ctc = get_ctc_prob_and_candidates(states, beam_width=BEAM_WIDTH, id2char=id2char)

    # birnn
    candidates_flat = list(map(process_input_text, chain(*candidates)))
    examples = [Example(text=text) for text in candidates_flat]

    lm_birnn = build_and_restore(model_cls=models.BiRNNLanguageModel, model_dir="./lm/birnn")
    log_prob_birnn = lm_birnn.predict(examples, batch_size=256)  # [num_examples * beam_width]
    log_prob_birnn = log_prob_birnn.reshape((-1, BEAM_WIDTH))

    # weighting
    scores = log_prob_ctc * W_CTC + log_prob_birnn * W_BIRNN  # [num_examples, beam_width]
    indices = scores.argmax(1)  # [num_examples]
    texts = list(map(process_output_text, (
        candidates_flat[BEAM_WIDTH * id_example + id_path] for id_example, id_path in enumerate(indices)
    )))

    # saving predictions
    save_predictions(output_dir=OUTPUT_DIR, images=images, texts=texts)


if __name__ == '__main__':
    main()
