import os
import glob
import json

import kenlm
import tensorflow as tf

import models
from utils import Example, save_predictions, clean_text
from postprocessing import LanguageModelKen, get_beam_states, get_ctc_prob_and_candidates
from layers import TransformerDecoder
from iterators import JointIterator


INPUT_DIR = '/data'
OUTPUT_DIR = '/output'

# BEAM_WIDTH = 10
# BEAM_WIDTH = 25
# BEAM_WIDTH = 30
BEAM_WIDTH = 100
# BEAM_WIDTH = 500

ALPHA = 0.7
BETA = 5
W_CTC = 50
W_BIRNN = 15
W_JOINT = 25
W_TRANSFORMER = 10

IMPROBABLE_SEQ = '000000'

# AD-HOC функции !!!1!


def build_joint_model():
    tf.keras.backend.clear_session()

    lm = models.TransformerLanguageModel.load("./joint", config_name="config_lm.json")
    lm.build()
    char2id = json.load(open("./joint/char_encodings.json"))
    vocab_size_lm = len(char2id)

    # setup layers
    config_dec = json.load(open("./joint/config_decoder.json"))
    d_model = config_dec["num_heads"] * config_dec["head_dim"]

    dec = TransformerDecoder(vocab_size=vocab_size_lm, **config_dec)

    dence_input_dec = tf.keras.layers.Dense(d_model, name='dence_input_dec')
    dense_vocab_dec = tf.keras.layers.Dense(vocab_size_lm, name='dense_vocab_dec')
    dense_vocab_lm = tf.keras.layers.Dense(vocab_size_lm, name='dense_vocab_lm')

    def build_join_body(char_ids, frames_features):
        x_lm = lm.model.get_layer("transformer_model").output  # [N, num_chars, hidden_lm]
        lm_logits = dense_vocab_lm(x_lm)  # [N, num_chars, vocab_size_lm]
        frames_features_proj = dence_input_dec(frames_features)  # [N, num_frames, d_model]
        x_dec = dec(char_ids, frames_features_proj)  # [N, num_chars, d_model]
        dec_logits = dense_vocab_dec(x_dec)  # [N, num_chars, vocab_size_lm]
        outputs = tf.keras.layers.Activation("softmax")(lm_logits + dec_logits)  # [N, num_chars, vocab_size_lm]
        return outputs

    # inference
    char_ids_ph = lm.model.inputs[0]
    frames_features_ph = tf.keras.layers.Input(shape=[None, 256], dtype=tf.float32, name="frames_features")
    inputs_inference = [char_ids_ph, frames_features_ph]
    outputs_inference = build_join_body(*inputs_inference)
    joint_decoder_inference = tf.keras.Model(inputs=inputs_inference, outputs=outputs_inference)

    return joint_decoder_inference, char2id


def predict_fast(decoder, examples, features_frames, char2id, img2id, batch_size=128):
    it = JointIterator(
        examples=examples,
        direction="fw",
        char2id=char2id,
        add_seq_borders=True,
        training=False,
        p_aug=-1,
        min_seq_len=1,
        pad=True,
        maxlen=100
    )
    ds = build_inference_ds(it, batch_size=batch_size)

    res = []
    for img_paths, char_ids, char_ids_next in iter(ds):
        indices = [img2id[img.decode()] for img in img_paths.numpy()]
        batch = char_ids, features_frames[indices]
        prob = decoder.predict_on_batch(batch)
        vocab_size = tf.shape(prob)[-1]
        y_oh = tf.one_hot(char_ids_next, vocab_size)
        prob_oh = prob * y_oh
        prob_target = tf.reduce_sum(prob_oh, axis=-1)
        log_prob_target = tf.math.log(prob_target)
        log_prob_target *= tf.cast(tf.math.not_equal(char_ids_next, 0), tf.float32)
        log_prob_target = log_prob_target[:, :-1]
        seq_scores = tf.reduce_sum(log_prob_target, axis=1)
        res.append(seq_scores)
    res = tf.concat(res, axis=0).numpy()
    return res


def build_inference_ds(it, batch_size):
    """
    Yields tuples:
    * img_path
    * char_ids
    * char_ids_next
    """
    ds = tf.data.Dataset.from_generator(
        lambda: iter(it),
        output_types=it.output_types,
        output_shapes=it.output_shapes
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


# старые функции

def build_and_restore(model_cls, model_dir, restore=True):
    tf.keras.backend.clear_session()
    model = model_cls.load(model_dir)
    model.build()
    if restore:
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
    print('load images')
    images = glob.glob(os.path.join(INPUT_DIR, "*"))
    examples = [Example(img=img) for img in images]

    print('inference of the first htr model')
    htr_model = build_and_restore(model_cls=models.HTRModel, model_dir="./htr")
    char_prob = htr_model.predict_v2(examples, batch_size=128)

    print('inference of the second htr model')
    # htr_model = build_and_restore(model_cls=models.HTRModel, model_dir="./htr", restore=False)
    encoder = tf.keras.Model(
        inputs=htr_model.dec_model.inputs,
        outputs=htr_model.dec_model.get_layer("decoder_baseline").output
    )
    encoder.load_weights("./joint/encoder.hdf5")
    ds = htr_model.data_pipeline.build_test_dataset(examples, batch_size=128)
    features_frames = encoder.predict(ds)

    # ctc decoding with prefix beam search and kneser-ney smoothing LM
    id2char = {v: k for k, v in htr_model.char2id.items()}

    # kenlm
    print("decoding...")
    kenlm_path = "./lm/kneser-ney/model.arpa"
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

    print("flatten...")
    candidates_flat = []
    examples = []
    img2id = {}
    for i, candidates_i in enumerate(candidates):
        img2id[images[i]] = i
        for text in candidates_i:
            text_clean = process_input_text(text)
            x_new = Example(img=images[i], text=text_clean)
            examples.append(x_new)
            candidates_flat.append(text_clean)

    print("birnn scoring...")
    lm_birnn = build_and_restore(model_cls=models.BiRNNLanguageModel, model_dir="./lm/birnn")
    log_prob_birnn = lm_birnn.predict(examples, batch_size=256)  # [num_examples * beam_width]
    log_prob_birnn = log_prob_birnn.reshape((-1, BEAM_WIDTH))

    print("transformer scoring...")
    lm_transformer = build_and_restore(model_cls=models.TransformerLanguageModel, model_dir="./lm/transformer")
    log_prob_transformer = lm_transformer.predict(examples, batch_size=256)  # [num_examples * beam_width]
    log_prob_transformer = log_prob_transformer.reshape((-1, BEAM_WIDTH))

    print("attn scoring...")
    joint_decoder_inference, char2id = build_joint_model()
    joint_decoder_inference.load_weights("./joint/decoder.hdf5")
    log_prob_joint = predict_fast(
        decoder=joint_decoder_inference,
        examples=examples,
        features_frames=features_frames,
        char2id=char2id,
        img2id=img2id,
        batch_size=128
    )
    log_prob_joint = log_prob_joint.reshape((-1, BEAM_WIDTH))

    # weighting
    print("weighting...")
    scores = log_prob_ctc * W_CTC + log_prob_birnn * W_BIRNN + log_prob_joint * W_JOINT + log_prob_transformer * W_TRANSFORMER  # [num_examples, beam_width]
    indices = scores.argmax(1)  # [num_examples]
    texts = list(map(process_output_text, (
        candidates_flat[BEAM_WIDTH * id_example + id_path] for id_example, id_path in enumerate(indices)
    )))

    # saving predictions
    print("saving...")
    save_predictions(output_dir=OUTPUT_DIR, images=images, texts=texts)


if __name__ == '__main__':
    main()
