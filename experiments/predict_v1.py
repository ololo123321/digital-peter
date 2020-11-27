import os
import glob
import json
import re
import argparse
from itertools import chain

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

    # postprocessing
    # TODO: сделать нормально с этими параметрами
    # postprocessing_config_path = './postprocessing_config.json'  # {"greedy": False, "lm_name": "", "ctc_weight": 0.65}
    # postprocessing_config = json.load(open(postprocessing_config_path))
    # if postprocessing_config["greedy"]:
    greedy = False
    if greedy:
        texts = candidates[0]
    else:
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
        log_prob_birnn = lm_birnn.predict(examples, batch_size=256)  # [num_examples * top_paths]
        log_prob_birnn = log_prob_birnn.reshape((-1, top_paths))

        tf.keras.backend.clear_session()
        lm_dir_transformer = "./lm/transformer"
        lm_transformer = build_and_restore(models.TransformerLanguageModel, lm_dir_transformer)
        log_prob_transformer = lm_transformer.predict(examples, batch_size=256)  # [num_examples * top_paths]
        log_prob_transformer = log_prob_transformer.reshape((-1, top_paths))

        w_ctc = 50
        w_birnn = 30
        w_transformer = 20
        scores = log_prob_ctc * w_ctc + log_prob_birnn * w_birnn + log_prob_transformer * w_transformer  # [num_examples, top_paths]
        indices = scores.argmax(1)  # [num_examples]
        texts = list(map(process_output_text, (
            candidates_flat[top_paths * id_example + id_path] for id_example, id_path in enumerate(indices)
        )))

    # saving predictions
    save_predictions(output_dir=OUTPUT_DIR, images=images, texts=texts)


if __name__ == '__main__':
    main()
