import os
import glob
import json
import re
import argparse
import time
from itertools import chain
from multiprocessing import Process, Manager

import tensorflow as tf

import sys; sys.path.insert(0, "./src")
import models
from utils import Example, CharEncoder, save_predictions, clean_text
from postprocessing import lm_score


INPUT_DIR = '/data'
OUTPUT_DIR = '/output'
HTR_MODEL_DIR = './htr'
LM_DIR = './lm'


class HTRModelV2(models.HTRModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, examples, **kwargs):
        """
        Пусть num_texts = len(texts). Тогда данная функция возвращает следующее:
        * decoded - список длины top_paths.
        decoded[i] - tf.Tensor of type tf.int32 and shape [num_texts, maxlen].
        decoded[i][j, k] - айдишник k-ого символа j-ой последовательности i-ых "лучшей путей", полученных
        с помощью алгоритма beam search.
        * log_prob - tf.Tensor of type tf.float32 and shape [num_texts, top_paths].
        log_prob[i, j] - логарифм вероятности j-ой лучшей последовательности примера i
        """
        ds_test = self.data_pipeline.build_test_dataset(examples)
        char_prob = self.dec_model.predict(ds_test)  # [N, T, V]
        return char_prob


def build_restore_predict_htr(input_data, res):
    print("build_restore_predict starts")
    model_dir = './htr'
    model = HTRModelV2.load(model_dir)
    model.build()
    print("model build")
    model.restore(os.path.join(model_dir, "model.hdf5"))
    res["res"] = model.predict(input_data)
    print("prediction done")


def build_restore_predict_bigru(input_data, res):
    print("build_restore_predict starts")
    model_dir = "./lm/birnn"
    model = models.BiRNNLanguageModel.load(model_dir)
    model.build()
    print("model build")
    model.restore(os.path.join(model_dir, "model.hdf5"))
    res["res"] = model.predict(input_data)
    print("prediction done")


def build_restore_predict_transformer(input_data, res):
    print("build_restore_predict starts")
    model_dir = "./lm/transformer"
    model = models.TransformerLanguageModel.load(model_dir)
    model.build()
    print("model build")
    model.restore(os.path.join(model_dir, "model.hdf5"))
    res["res"] = model.predict(input_data)
    print("prediction done")


if __name__ == '__main__':
    # load images
    images = glob.glob(os.path.join(INPUT_DIR, "*"))

    manager = Manager()
    model_result = manager.dict()

    # htr
    examples = [Example(img=img) for img in images]
    p = Process(target=build_restore_predict_htr, args=(examples, model_result))
    p.start()
    pid = p.pid
    print("pid htr:", pid)
    p.join()

    char_prob = model_result['res']
    beam_width = 100
    top_paths = 100
    input_length = [char_prob.shape[1]] * char_prob.shape[0]
    decoded, log_prob_ctc = tf.keras.backend.ctc_decode(
        char_prob,
        input_length=input_length,
        greedy=False,
        beam_width=beam_width,
        top_paths=top_paths
    )

    log_prob_ctc = log_prob_ctc.numpy()
    del model_result['res']

    # ids -> chars
    char2id = json.load(open('./htr/char_encodings.json'))
    char_encoder = CharEncoder(char2id=char2id)
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

        # birnn
        # os.kill(pid, 9)
        p.close()
        # p.terminate()
        manager2 = Manager()
        model_result2 = manager2.dict()
        p2 = Process(target=build_restore_predict_bigru, args=(examples, model_result2))
        p2.start()
        print("pid birnn:", p2.pid)
        p2.join()

        log_prob_birnn = model_result['res']
        log_prob_birnn = log_prob_birnn.reshape((-1, top_paths))
        del model_result['res']

        # # transformer
        # p3 = Process(target=build_restore_predict,
        #              args=("./lm/transformer", models.TransformerLanguageModel, examples, model_result))
        # p3.start()
        # p3.join()
        # # p.close()
        # log_prob_transformer = model_result['res']
        # log_prob_transformer = log_prob_transformer.reshape((-1, top_paths))
        # del model_result['res']
        #
        # w_ctc = 50
        # w_birnn = 30
        # w_transformer = 20
        # scores = log_prob_ctc * w_ctc + log_prob_birnn * w_birnn + log_prob_transformer * w_transformer  # [num_examples, top_paths]
        # indices = scores.argmax(1)  # [num_examples]
        # texts = list(map(process_output_text, (
        #     candidates_flat[top_paths * id_example + id_path] for id_example, id_path in enumerate(indices)
        # )))

        # texts = lm_score(candidates=candidates, log_prob_ctc=log_prob_ctc, lm=lm, ctc_weight=postprocessing_config["ctc_weight"])

    # saving predictions
    # save_predictions(output_dir=OUTPUT_DIR, images=images, texts=texts)
