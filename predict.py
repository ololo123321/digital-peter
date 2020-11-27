import os
import glob
from argparse import ArgumentParser
from typing import List

import kenlm
import tensorflow as tf

import models
from utils import Example, clean_text
from postprocessing import LanguageModelKen, get_beam_states, get_ctc_prob_and_candidates


IMPROBABLE_SEQ = '000000'


def infer_moder_dir(model_folder_name: str, val_mode: bool) -> str:
    group = 'train' if val_mode else 'train-valid'
    return os.path.join('./models', group, model_folder_name)


def build_and_restore(model_cls, model_dir: str, **kwargs):
    tf.keras.backend.clear_session()
    model = model_cls.load(model_dir)
    model.build(**kwargs)
    model.restore(model_dir)
    return model


def process_input_text(text: str) -> str:
    """
    чтобы гарантировать отсутствие падения языковой модели,
    заменим пустые строки на какие-то маловероятные
    """
    text = clean_text(text)
    if text == "":
        return IMPROBABLE_SEQ
    return text


def process_output_text(text: str) -> str:
    if text == IMPROBABLE_SEQ:
        return ""
    return text


def save_predictions(output_dir: str, images: List[str], texts: List[str]):
    os.makedirs(output_dir, exist_ok=True)
    for image, text in zip(images, texts):
        name = os.path.basename(image).split(".")[0]
        with open(os.path.join(output_dir, str(name) + ".txt"), "w") as f:
            f.write(text)


def main(args):
    print('load images')
    images = glob.glob(os.path.join(args.input_dir, "*"))
    examples = [Example(img=img) for img in images]

    print('inference of the first htr model')
    htr_model = build_and_restore(model_cls=models.HTRModel, model_dir=infer_moder_dir('htr', val_mode=args.val_mode))
    char_prob = htr_model.get_ctc_prob(examples, batch_size=128)

    print("decoding...")
    lm = LanguageModelKen(lm=kenlm.Model(infer_moder_dir('lm/kneser-ney/model.arpa', val_mode=args.val_mode)))
    id2char = {v: k for k, v in htr_model.char2id.items()}
    states = get_beam_states(
        char_prob,
        classes=id2char,
        lm=lm,
        beam_width=args.beam_width,
        alpha=args.alpha,
        beta=args.beta,
        min_char_prob=0.001
    )
    candidates_str, log_prob_ctc = get_ctc_prob_and_candidates(states, beam_width=args.beam_width, id2char=id2char)

    print("flatten...")
    candidates = []
    img2id = {}
    for i, candidates_i in enumerate(candidates_str):
        img = images[i]
        img2id[img] = i
        for text in candidates_i:
            text_clean = process_input_text(text)
            x = Example(img=img, text=text_clean)
            candidates.append(x)

    print("attn scoring...")
    joint_model = build_and_restore(
        model_cls=models.JointModel,
        model_dir=infer_moder_dir('joint', val_mode=args.val_mode),
        training=False
    )
    log_prob_joint = joint_model.predict(
        examples=examples,
        candidates=candidates,
        img2id=img2id,
        batch_size_enc=128,
        batch_size_dec=512
    )
    log_prob_joint = log_prob_joint.reshape((-1, args.beam_width))

    print("birnn scoring...")
    lm_birnn = build_and_restore(
        model_cls=models.BiRNNLanguageModel,
        model_dir=infer_moder_dir('lm/birnn', val_mode=args.val_mode)
    )
    log_prob_birnn = lm_birnn.predict(candidates, batch_size=256)  # [num_examples * beam_width]
    log_prob_birnn = log_prob_birnn.reshape((-1, args.beam_width))

    print("transformer scoring...")
    lm_transformer = build_and_restore(
        model_cls=models.TransformerLanguageModel,
        model_dir=infer_moder_dir('lm/transformer', val_mode=args.val_mode)
    )
    log_prob_transformer = lm_transformer.predict(candidates, batch_size=256)  # [num_examples * beam_width]
    log_prob_transformer = log_prob_transformer.reshape((-1, args.beam_width))

    # weighting
    print("weighting...")
    scores = log_prob_ctc * args.w_ctc \
        + log_prob_birnn * args.w_birnn \
        + log_prob_joint * args.w_joint \
        + log_prob_transformer * args.w_transformer  # [num_examples, beam_width]
    indices = scores.argmax(1)  # [num_examples]
    texts = list(map(process_output_text, (
        candidates[args.beam_width * id_example + id_path].text for id_example, id_path in enumerate(indices)
    )))

    # saving predictions
    print("saving...")
    save_predictions(output_dir=args.output_dir, images=images, texts=texts)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--val_mode', action='store_true',
                        help='если true, то подгружать модели, которые видели 90% выборки (./models/train).'
                             'нужно для оценки качества.'
                             'иначе - те, которые видели всю выборку (./models/train-valid)')
    parser.add_argument('--input_dir', type=str, required=True, help='')
    parser.add_argument('--output_dir', type=str, required=True, help='')
    parser.add_argument('--w_ctc', default=50, type=int, required=False, help='')
    parser.add_argument('--w_birnn', default=15, type=int, required=False, help='')
    parser.add_argument('--w_transformer', default=10, type=int, required=False, help='')
    parser.add_argument('--w_joint', default=25, type=int, required=False, help='')
    parser.add_argument('--alpha', default=0.7, type=float, required=False, help='')
    parser.add_argument('--beta', default=5, type=float, required=False, help='')
    parser.add_argument('--beam_width', default=100, type=int, required=False, help='')
    args_ = parser.parse_args()
    print(args_)
    main(args_)
