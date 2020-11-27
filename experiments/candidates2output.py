import os
import glob
import json
import re
import argparse
from itertools import chain

import numpy as np
import tensorflow as tf

import sys; sys.path.insert(0, "./src")
import models
from utils import Example, CharEncoder, save_predictions, clean_text
from postprocessing import lm_score


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
    # load intermediate results
    log_prob_ctc = np.load("log_prob_ctc.npy")
    candidates = json.load(open("candidates.json"))

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

    tf.keras.backend.clear_session()
    lm_dir_birnn = "./lm/birnn"
    lm_birnn = build_and_restore(models.BiRNNLanguageModel, lm_dir_birnn)
    log_prob_birnn = lm_birnn.predict(examples)  # [num_examples * top_paths]
    log_prob_birnn = log_prob_birnn.reshape((-1, top_paths))

    tf.keras.backend.clear_session()
    lm_dir_transformer = "./lm/transformer"
    lm_transformer = build_and_restore(models.TransformerLanguageModel, lm_dir_transformer)
    log_prob_transformer = lm_transformer.predict(examples)  # [num_examples * top_paths]
    log_prob_transformer = log_prob_transformer.reshape((-1, top_paths))

    w_ctc = 50
    w_birnn = 30
    w_transformer = 20
    scores = log_prob_ctc * w_ctc + log_prob_birnn * w_birnn + log_prob_transformer * w_transformer  # [num_examples, top_paths]
    indices = scores.argmax(1)  # [num_examples]
    texts = list(map(process_output_text, (
        candidates_flat[top_paths * id_example + id_path] for id_example, id_path in enumerate(indices)
    )))

    # texts = lm_score(candidates=candidates, log_prob_ctc=log_prob_ctc, lm=lm, ctc_weight=postprocessing_config["ctc_weight"])

    # saving predictions
    images = glob.glob(os.path.join(INPUT_DIR, "*"))
    save_predictions(output_dir=OUTPUT_DIR, images=images, texts=texts)


if __name__ == '__main__':
    main()
