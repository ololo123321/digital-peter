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
    # load images
    images = glob.glob(os.path.join(INPUT_DIR, "*"))

    # build htr model
    htr_model = build_and_restore(models.HTRModel, HTR_MODEL_DIR)

    # inference of htr model
    examples = [Example(img=img) for img in images]
    decoded, log_prob_ctc = htr_model.predict(examples)
    log_prob_ctc = log_prob_ctc.numpy()

    # ids -> chars
    char_encoder = CharEncoder(char2id=htr_model.char2id)
    candidates = []
    for x in decoded:
        candidates_i = []
        for char_ids in x.numpy():
            text = char_encoder.decode(char_ids)
            text = clean_text(text)
            candidates_i.append(text)
        candidates.append(candidates_i)

    # save intermediate results
    np.save("log_prob_ctc.npy", log_prob_ctc)

    with open("candidates.json", "w") as f:
        json.dump(candidates, f)


if __name__ == '__main__':
    main()
