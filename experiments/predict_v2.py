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
HTR_MODEL_DIR = './htr'
LM_DIR = './lm'


def build_and_restore(model_cls, model_dir):
    model = model_cls.load(model_dir)
    model.build()
    model.restore(os.path.join(model_dir, "model.hdf5"))
    return model


def main():
    # load images
    images = glob.glob(os.path.join(INPUT_DIR, "*"))

    # build htr model
    htr_model = build_and_restore(models.HTRModel, HTR_MODEL_DIR)

    # inference of htr model
    examples = [Example(img=img) for img in images]
    char_prob = htr_model.predict_v2(examples)

    # ctc decoding with prefix beam search and kneser-ney smoothing LM
    id2char = {v: k for k, v in htr_model.char2id.items()}
    beam_width = 10

    # kenlm
    kenlm_path = "./lm/kneser-ney/old_slavic_texts_wo_petr_letters_char_level_5.arpa"
    lm = LanguageModelKen(lm=kenlm.Model(kenlm_path))

    states = get_beam_states(
        char_prob,
        classes=id2char,
        lm=lm,
        beam_width=beam_width,
        alpha=1.0,
        beta=2.0,
        min_char_prob=0.001
    )
    candidates, log_prob_ctc = get_ctc_prob_and_candidates(states, beam_width=beam_width, id2char=id2char)
    texts = [clean_text(x[0]) for x in candidates]

    # saving predictions
    save_predictions(output_dir=OUTPUT_DIR, images=images, texts=texts)


if __name__ == '__main__':
    main()
