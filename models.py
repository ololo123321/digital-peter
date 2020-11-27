import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict
from shutil import copyfile

import numpy as np
import tensorflow as tf

import encoders
from utils import Example, CustomSchedule, LMDirection, ce_masked_loss, get_optimizer, get_seq_scores
from layers import TransformerModel, BiRNNLayer, TransformerDecoder
from tf_data import DataPipelineBuilderHTR, DataPipelineBuilderLM, build_joint_decoder_inference_ds
from iterators import HTRIterator, LMIterator, JointIterator


class BaseModel(ABC):
    def __init__(self, config, char2id):
        """
        config = {
            "data": {
                "pipeline": <name of class from tf_data.py>,
                <common params of input data>
            },
            "model": {
                <параметры модели>
            },
            "train": {
                "model_dir": "",
                "epochs": 1000,
                "batch_size": 16,
                "buffer": 1000  # число объектов, из которых сэмплируется батч
            },
            "predict": {
                "batch_size": 16,
                <прочие атрибуты: например beam_width и top_path в случае HTRModel>
            }
        }
        """
        self.config = config
        self.char2id = char2id

        self.model = None
        self.data_pipeline = None

        self.setup_data_pipeline()

    @abstractmethod
    def build(self, *args, **kwargs): ...

    @abstractmethod
    def predict(self, examples: List[Example], **kwargs): ...

    @abstractmethod
    def setup_data_pipeline(self): ...

    def fit(
            self,
            examples_train: List[Example],
            examples_valid: List[Example] = None,
            train_batch_size: int = 16,
            valid_batch_size: int = 256,
            buffer: int = 1000
    ):
        """
        data = [images, padded_labels, input_lengths, label_lengths]
        """
        assert self.data_pipeline is not None, "setup data pipeline!"
        assert self.model is not None, "build model first!"

        ds_train = self.data_pipeline.build_train_dataset(examples_train, batch_size=train_batch_size, buffer=buffer)

        if examples_valid is not None:
            callbacks = self.callbacks
            ds_valid = self.data_pipeline.build_valid_dataset(examples_valid, batch_size=valid_batch_size)
        else:
            callbacks = None
            ds_valid = None

        steps_per_epoch = len(examples_train) // self.config["train"]["batch_size"] + 1
        history = self.model.fit(
            x=ds_train,
            epochs=self.config["train"]["epochs"],
            steps_per_epoch=steps_per_epoch,
            validation_data=ds_valid,
            callbacks=callbacks,
            verbose=self.config['train'].get("verbose", 1)
        )
        return history

    @property
    def callbacks(self):
        verbose = self.config['train'].get("verbose", 1)
        callbacks = [
            tf.keras.callbacks.CSVLogger(
                filename=os.path.join(self.config["train"]["model_dir"], "epochs.log"),
                separator=";",
                append=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config["train"]["model_dir"], "model.hdf5"),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=verbose
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config["train"].get("es_patience", 20),
                min_delta=1e-8,
                restore_best_weights=True,
                verbose=verbose
            )
        ]
        if not self.config['train'].get('noam_scheme', False):
            reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config["train"].get("lr_reduction_factor", 0.2),
                patience=self.config["train"].get("lr_reduction_patience", 15),
                min_lr=1e-5,
                min_delta=1e-8,
                verbose=verbose
            )
            callbacks.append(reduce_lr_cb)
        return callbacks

    def save(self):
        model_dir = self.config["train"]["model_dir"]

        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

        with open(os.path.join(model_dir, "char_encodings.json"), "w") as f:
            json.dump(self.char2id, f, indent=4, ensure_ascii=False)

        with open(os.path.join(model_dir, "model.summary"), "w") as f:
            self.model.summary(print_fn=lambda line: f.write(line + '\n'))

        if hasattr(self.data_pipeline, "augmentations"):
            with open(os.path.join(model_dir, "augmentations.json"), "w") as f:
                json.dump(self.data_pipeline.augmentations, f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, model_dir: str, config_name: str = "config.json"):
        """
        Подгрузка всего, кроме весов:
        * конфига модели
        * конфига аугментаций (если нужно продолжать обучение)
        * кодировок символов
        """
        with open(os.path.join(model_dir, config_name)) as f:
            config = json.load(f)

        with open(os.path.join(model_dir, "char_encodings.json")) as f:
            char2id = json.load(f)

        model = cls(config=config, char2id=char2id)

        aug_path = os.path.join(model_dir, "augmentations.json")
        if os.path.exists(aug_path):
            with open(aug_path) as f:
                model.data_pipeline.augmentations = json.load(f)

        return model

    def restore(self, model_dir=None, model_name="model.hdf5"):
        """
        Подгрузка весов модели.
        При разработке путь указан в self.config.
        При сабмите путь из конфига.
        """
        model_dir = model_dir or self.config["train"]["model_dir"]
        self.model.load_weights(os.path.join(model_dir, model_name))

# htr


class HTRModel(BaseModel):
    def __init__(self, config, char2id):
        """
        Пример конфига:
        {
            "data": {
                "image": {
                    "num_channels": 1,
                    "target_height": 128,
                    "target_width": 1024,
                    "rot90": True,
                    "erode": False
                },
                "aug": {
                    "max_delta_stretch": 0.3.
                    "max_delta_brightness": 0.1
                }
            },
            "model": {
                "image_encoder": {
                    "name": "EncoderFlor",
                    "params": {},
                },
                "frames_encoder": {
                    "name": "BiRNNEncoder",
                    "params": {
                        "cell_name": "GRU",
                        "num_layers": 2,
                        "cell_dim": 128,
                        "dropout": 0.5
                    },
                },
                "vocab_size": 70
            },
            "train": {
                "batch_size": 16,
                "epochs": 1000,
                "buffer": 1000,
                "model_dir": "",
                "es_patience": 20,
                "lr_reduction_factor": 0.2,
                "lr_reduction_patience": 15
            },
            "predict": {
                "batch_size": 32,
                "beam_width": 100,
                "top_paths": 100
            }
        }
        """
        super().__init__(config=config, char2id=char2id)

        self.images_ph = None
        self.labels_ph = None
        self.logits_length_ph = None
        self.labels_length_ph = None

        self.dec_model = None

    def setup_data_pipeline(self):
        self.data_pipeline = DataPipelineBuilderHTR(
            num_channels=self.config["data"]["image"]["num_channels"],
            target_height=self.config["data"]["image"]["target_height"],
            target_width=self.config["data"]["image"]["target_width"],
            rot90=self.config["data"]["image"]["rot90"]
        )

    def build(self, lr=1e-3):
        # inputs
        config_image = self.config["data"]["image"]
        if config_image["rot90"]:
            img_shape = config_image["target_width"], config_image["target_height"], config_image["num_channels"]
        else:
            img_shape = config_image["target_height"], config_image["target_width"], config_image["num_channels"]
        self.images_ph = tf.keras.layers.Input(name='images', shape=img_shape)
        self.labels_ph = tf.keras.layers.Input(name='the_labels', shape=[None], dtype='float32')
        self.logits_length_ph = tf.keras.layers.Input(name='input_length', shape=[1], dtype='int64')
        self.labels_length_ph = tf.keras.layers.Input(name='label_length', shape=[1], dtype='int64')

        # encoder
        encoder_cls = getattr(encoders, self.config["model"]["image_encoder"]["name"])
        encoder = encoder_cls(**self.config["model"]["image_encoder"]["params"])

        # decoder
        decoder_cls = getattr(encoders, self.config["model"]["frames_encoder"]["name"])
        decoder = decoder_cls(**self.config["model"]["frames_encoder"]["params"])

        # fc on vocab (+ 1 for blank symbol, used by ctc)
        dense_vocab = tf.keras.layers.Dense(self.config["model"]["vocab_size"] + 1, activation=tf.nn.softmax)

        # graph
        x = encoder(self.images_ph)
        x = decoder(x)
        dec_logits = dense_vocab(x)

        # loss
        def ctc_lambda_func(args):
            return tf.keras.backend.ctc_batch_cost(*args)

        loss_layer = tf.keras.layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')
        loss_inputs = [self.labels_ph, dec_logits, self.logits_length_ph, self.labels_length_ph]
        loss_out = loss_layer(loss_inputs)

        # training model
        model_inputs = [self.images_ph, self.labels_ph, self.logits_length_ph, self.labels_length_ph]
        self.model = tf.keras.Model(inputs=model_inputs, outputs=loss_out)

        opt = tf.optimizers.Adam(learning_rate=lr)
        self.model.compile(
            loss={'ctc': lambda y_true, y_pred: y_pred},
            optimizer=opt,
            metrics=['accuracy']
        )

        # decoding model
        self.dec_model = tf.keras.Model(inputs=self.images_ph, outputs=dec_logits)

    def predict(self, examples: List[Example], **kwargs):
        """
        Пусть num_texts = len(texts). Тогда данная функция возвращает следующее:
        * decoded - список длины top_paths.
        decoded[i] - tf.Tensor of type tf.int32 and shape [num_texts, maxlen].
        decoded[i][j, k] - айдишник k-ого символа j-ой последовательности i-ых "лучшей путей", полученных
        с помощью алгоритма beam search.
        * log_prob - tf.Tensor of type tf.float32 and shape [num_texts, top_paths].
        log_prob[i, j] - логарифм вероятности j-ой лучшей последовательности примера i
        """
        batch_size = kwargs.get("batch_size", self.config['predict']['batch_size'])
        beam_width = kwargs.get("beam_width", self.config['predict']['beam_width'])
        top_paths = kwargs.get("top_paths", self.config['predict']['top_paths'])

        char_prob = self.get_ctc_prob(examples, batch_size=batch_size)  # [N, T, V]
        input_length = [char_prob.shape[1]] * char_prob.shape[0]
        greedy = beam_width == 1  # для этого частного случая реализация более эффективная: просто взять argmax
        decoded, log_prob_ctc = tf.keras.backend.ctc_decode(
            char_prob,
            input_length=input_length,
            greedy=greedy,
            beam_width=beam_width,
            top_paths=top_paths
        )
        return decoded, log_prob_ctc

    def get_ctc_prob(self, examples: List[Example], **kwargs) -> np.ndarray:
        """возвращает вероятности символов"""
        batch_size = kwargs.get("batch_size", self.config['predict']['batch_size'])
        it = HTRIterator(examples=examples, test_mode=True)
        ds_test = self.data_pipeline.build_test_dataset(it, batch_size=batch_size)
        char_prob = self.dec_model.predict(ds_test)  # [N, T, V]
        return char_prob


# lm


class BaseLanguageModel(BaseModel):
    def __init__(self, config, char2id):
        super().__init__(config=config, char2id=char2id)

    @abstractmethod
    def build(self, *args, **kwargs): ...

    def setup_data_pipeline(self):
        self.data_pipeline = DataPipelineBuilderLM()

    def predict(self, examples: List[Example], *args, **kwargs):
        batch_size = kwargs.get("batch_size", self.config["predict"]["batch_size"])
        it = LMIterator(
            examples=examples,
            direction=self.config["model"]["direction"],
            char2id=self.char2id,
            add_seq_borders=self.config["data"]["add_seq_borders"],
            training=False,
            pad=True
        )
        ds = self.data_pipeline.build_test_dataset(it, batch_size=batch_size)
        res = []
        for x, y in ds:
            prob = self.model.predict_on_batch(x)
            seq_scores = get_seq_scores(
                prob=prob,
                char_ids_next=y,
                eos=False,
                pad_id=0,
                eos_id=self.char2id[LMIterator.EOS]
            )
            res.append(seq_scores)
        res = tf.concat(res, axis=0).numpy()
        return res


class BiRNNLanguageModel(BaseLanguageModel):
    def __init__(self, config, char2id):
        """
        Пример конфига:
        {
            "data": {
                "add_seq_borders": True
            },
            "model": {
                "direction": "bi",  # для DataPipelineBuilderLM
                "vocab_size": 80,
                "emb_dim": 256,
                "birnn": {
                    "cell_name": "GRU",
                    "cell_dim": 128,
                    "num_layers": 3,
                    "dropout": 0.5,
                    "skip_connections": True,
                    "stateful": False
                },
            }
            },
            "train": {},
            "predict": {}
        }
        """
        super().__init__(config=config, char2id=char2id)

    def build(self):
        config_model = self.config["model"]
        char_ids = tf.keras.layers.Input(name='char_ids', shape=[None], dtype=tf.int32)
        x = tf.keras.layers.Embedding(config_model["vocab_size"], config_model["emb_dim"], mask_zero=False)(char_ids)

        config_birnn = config_model['birnn']
        if config_birnn['skip_connections'] and (config_model["emb_dim"] != config_birnn["cell_dim"]):
            x = tf.keras.layers.Dense(config_birnn["cell_dim"])(x)

        x = BiRNNLayer(**config_birnn)(x, char_ids=char_ids)
        outputs = tf.keras.layers.Dense(config_model["vocab_size"], activation=tf.nn.softmax)(x)

        self.model = tf.keras.Model(inputs=char_ids, outputs=outputs)

        self.model.compile(loss=ce_masked_loss, optimizer='adam', metrics=['accuracy'])


class TransformerLanguageModel(BaseLanguageModel):
    def __init__(self, config, char2id):
        super().__init__(config=config, char2id=char2id)

    def build(self):
        inputs = tf.keras.layers.Input(name='char_ids', shape=[None], dtype=tf.int32)
        x = TransformerModel(vocab_size=self.config["model"]["vocab_size"], **self.config["model"]["transformer"])(inputs)
        outputs = tf.keras.layers.Dense(self.config["model"]["vocab_size"], activation=tf.nn.softmax)(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        if self.config["train"]["noam_scheme"]:
            # https://www.tensorflow.org/tutorials/text/transformer#optimizer
            d_model = self.config["model"]["transformer"]["num_heads"] * self.config["model"]["transformer"]["head_dim"]
            warmup_steps = self.config["train"]["warmup_steps"]
            learning_rate = CustomSchedule(d_model=d_model, warmup_steps=warmup_steps)
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.98,
                epsilon=1e-9
            )
        else:
            optimizer = tf.keras.optimizers.Adam()

        self.model.compile(loss=ce_masked_loss, optimizer=optimizer, metrics=['accuracy'])


class RNNLM(BaseLanguageModel):
    def __init__(self, config, char2id):
        super().__init__(config, char2id)
        self.model_inference = None

    def build(self):
        char_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        prev_hidden_states = tf.keras.layers.Input(shape=(None,), dtype=tf.float32)

        vocab_size = self.config["model"]["vocab_size"]
        emb_dim = self.config["model"]["emb_dim"]
        cell_name = self.config["model"]["rnn"]["cell_name"]
        cell_dim = self.config["model"]["rnn"]["cell_dim"]
        dropout = self.config["model"]["rnn"]["dropout"]

        #         emb = tf.keras.layers.Embedding(vocab_size, emb_dim, mask_zero=True)
        emb = tf.keras.layers.Embedding(vocab_size, emb_dim)
        cell_cls = getattr(tf.keras.layers, cell_name)
        cell = cell_cls(cell_dim, dropout=dropout)
        rnn = tf.keras.layers.RNN(cell, return_sequences=True)
        dense_vocab = tf.keras.layers.Dense(vocab_size, activation='softmax')

        embeddings = emb(char_ids)  # [N, T, emb_dim]

        # train
        x = rnn(embeddings)
        prob_train = dense_vocab(x)
        self.model = tf.keras.Model(inputs=char_ids, outputs=prob_train)

        self.model.compile(loss=ce_masked_loss, optimizer='adam', metrics=['accuracy'])

        # inference
        # предполагается, что первое измерение embeddings равно 1
        x = tf.squeeze(embeddings, 1)  # [N, emb_dim]
        h, new_hidden_states = cell(x, prev_hidden_states)  # h: [N, cell_dim], new_hidden_states: [N, cell_dim]
        prob_inference = dense_vocab(h)  # [N, vocab_size]
        self.model_inference = tf.keras.Model(inputs=[char_ids, prev_hidden_states],
                                              outputs=[prob_inference, new_hidden_states])

    def predict_step(self, chars, hidden):
        """
        char_ids: List[int]
        hidden: List[np.ndarray[cell_dim]]
        """
        char_ids = np.array([[self.char2id.get(char, '<unk>')] for char in chars])
        cell_dim = self.config["model"]["rnn"]["cell_dim"]
        hidden = np.array([x if x is not None else np.zeros(cell_dim) for x in hidden])
        prob, next_hidden = self.model_inference([char_ids, hidden])
        return prob.numpy(), next_hidden.numpy()


# joint

class JointModel(BaseModel):
    def __init__(self, config, char2id):
        super().__init__(config=config, char2id=char2id)

        self.htr = None  # для кодирования изображения
        self.model = None

        self.enc_inference = None  # htr без головы на логиты словаря
        self.dec_inference = None  # transformer + lm

        self.data_pipeline_dec = None

    def setup_data_pipeline(self):
        self.data_pipeline = DataPipelineBuilderHTR(
            num_channels=self.config["data"]["image"]["num_channels"],
            target_height=self.config["data"]["image"]["target_height"],
            target_width=self.config["data"]["image"]["target_width"],
            rot90=self.config["data"]["image"]["rot90"]
        )

    def build(self, training: bool):
        """
        TRAIN:
        config = {
            "model": {
                "enc": {"pretrained_dir": "/path/to/htr"},
                "lm": {"pretrained_dir": "/path/to/transformer"},
                "dec": {
                    "num_layers": 4,
                    "num_heads": 4,
                    "head_dim": 32,
                    "dff": 1024,
                    "dropout": 0.2
                },
                "joint_dir": ""
            },
            ...
        }

        transformer
            model.hdf5
            config.json

        htr
            model.hdf5
            nofig.json

        INFERENCE:
        config = {
            "model": {
                "joint_dir": ""
            },
            ...
        }
        joint
            encoder.hdf5
            decoder.hdf5
            config_encoder.json
            config_decoder.json
            config_lm.json
            char_encodings.json

        """
        if training:
            model_dir_enc = self.config['model']['enc']['pretrained_dir']
            model_dir_lm = self.config['model']['lm']['pretrained_dir']
            config_name_enc = config_name_lm = 'config.json'
            config_dec = self.config['model']['dec']
        else:
            model_dir_enc = model_dir_dec = model_dir_lm = self.config['model']['joint_dir']
            config_name_enc = 'config_encoder.json'
            config_name_lm = 'config_lm.json'
            config_dec = json.load(open(os.path.join(model_dir_dec, 'config_decoder.json')))

        # encoder
        self.htr = HTRModel.load(model_dir=model_dir_enc, config_name=config_name_enc)
        self.htr.build()

        # lm
        lm = TransformerLanguageModel.load(model_dir=model_dir_lm, config_name=config_name_lm)
        lm.build()
        if training:
            lm.restore(model_dir=model_dir_lm, model_name='model.hdf5')
        vocab_size_lm = len(lm.char2id)

        # decoder
        dec = TransformerDecoder(vocab_size=vocab_size_lm, **config_dec)

        d_model = config_dec['num_heads'] * config_dec['head_dim']
        dense_input_dec = tf.keras.layers.Dense(d_model, name='dence_input_dec')  # TODO: исправить очепятку dence -> dense
        dense_vocab_dec = tf.keras.layers.Dense(vocab_size_lm, name='dense_vocab_dec')
        dense_vocab_lm = tf.keras.layers.Dense(vocab_size_lm, name='dense_vocab_lm')

        def build_joint_body(char_ids, frames_features):
            x_lm = lm.model.get_layer("transformer_model").output  # [N, num_chars, hidden_lm]
            lm_logits = dense_vocab_lm(x_lm)  # [N, num_chars, vocab_size_lm]
            frames_features_proj = dense_input_dec(frames_features)  # [N, num_frames, d_model]
            x_dec = dec(char_ids, frames_features_proj)  # [N, num_chars, d_model]
            dec_logits = dense_vocab_dec(x_dec)  # [N, num_chars, vocab_size_lm]
            outputs = tf.keras.layers.Activation("softmax")(lm_logits + dec_logits)  # [N, num_chars, vocab_size_lm]
            return outputs

        # graph
        outputs = build_joint_body(
            char_ids=lm.model.inputs[0],
            frames_features=self.htr.dec_model.get_layer('decoder_baseline').output
        )

        # build model
        # x = (img, char_ids)
        # y = next_char_ids
        inputs = [self.htr.dec_model.inputs[0], lm.model.inputs[0]]
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # inference
        cell_dim_htr = self.htr.config["model"]["frames_encoder"]["params"]["cell_dim"]
        char_ids_ph = lm.model.inputs[0]
        frames_features_ph = tf.keras.layers.Input(shape=[None, cell_dim_htr * 2], dtype=tf.float32,
                                                   name="frames_features")
        inputs_inference = [char_ids_ph, frames_features_ph]
        outputs_inference = build_joint_body(*inputs_inference)
        self.dec_inference = tf.keras.Model(inputs=inputs_inference, outputs=outputs_inference)

        self.enc_inference = tf.keras.Model(
            inputs=self.htr.dec_model.inputs,
            outputs=self.htr.dec_model.get_layer("decoder_baseline").output  # TODO: не забыть поправить при смене названия соответствующего слоя
        )

    def compile(self, model, tvars: list = None, noam_scheme: bool = True, lr: float = 1e-3):
        if tvars is not None:
            for x in model.layers:
                x.trainable = x.name in tvars
        d_model = self.config['model']['dec']['num_heads'] * self.config['model']['dec']['head_dim']
        opt = get_optimizer(noam_scheme=noam_scheme, d_model=d_model, warmup_steps=4000, lr=lr)
        model.compile(loss=ce_masked_loss, optimizer=opt, metrics=['accuracy'])

    def predict(
            self,
            examples: List[Example],
            candidates: List[Example] = None,
            img2id: Dict[str, int] = None,
            batch_size_enc: int = 128,
            batch_size_dec: int = 512
    ):
        # кодирование изображений
        it_htr = HTRIterator(examples, test_mode=True)
        ds = self.data_pipeline.build_test_dataset(it_htr, batch_size=batch_size_enc)
        features_frames = self.enc_inference.predict(ds)

        # декодирование
        it_joint = JointIterator(
            examples=candidates,
            direction=LMDirection.FORWARD,
            char2id=self.char2id,
            add_seq_borders=True,
            training=False,
            p_aug=-1,
            min_seq_len=1,
            pad=True,
            maxlen=100
        )
        ds = build_joint_decoder_inference_ds(it_joint, batch_size=batch_size_dec)

        res = []
        for img_paths, char_ids, char_ids_next in iter(ds):
            indices = [img2id[img.decode()] for img in img_paths.numpy()]
            batch = char_ids, features_frames[indices]
            prob = self.dec_inference.predict_on_batch(batch)
            seq_scores = get_seq_scores(
                prob=prob,
                char_ids_next=char_ids_next,
                eos=False,
                pad_id=0,
                eos_id=self.char2id[JointIterator.EOS]
            )
            res.append(seq_scores)
        res = tf.concat(res, axis=0).numpy()
        return res

    def save(self):
        model_dir = self.config['model']['joint_dir']

        # 0. создание чистой папки
        os.system(f'rm -r {model_dir}')
        os.system(f'mkdir {model_dir}')

        # 1. сохранение конфга энкодера
        path_in = os.path.join(self.config['model']['enc']['pretrained_dir'], "config.json")
        path_out = os.path.join(model_dir, "config_encoder.json")
        copyfile(path_in, path_out)

        # 2. сохранение конфига языковой модели
        path_in = os.path.join(self.config['model']['lm']['pretrained_dir'], "config.json")
        path_out = os.path.join(model_dir, "config_lm.json")
        copyfile(path_in, path_out)

        # 3. сохранение кодировок символов
        path_in = os.path.join(self.config['model']['lm']['pretrained_dir'], "char_encodings.json")
        path_out = os.path.join(model_dir, "char_encodings.json")
        copyfile(path_in, path_out)

        # 4. сохранение конфига с нуля обученного декодера
        path_out = os.path.join(model_dir, "config_decoder.json")
        with open(path_out, "w") as f:
            json.dump(self.config['model']['dec'], f, indent=4, ensure_ascii=False)

        # 5. сохранение весов энкодера
        path_out = os.path.join(model_dir, "encoder.hdf5")
        self.htr.save_weights(path_out)

        # 6. сохранение весов декодера + языковой модели
        path_out = os.path.join(model_dir, "joint_decoder.hdf5")
        self.model_inference.save_weights(path_out)

    @classmethod
    def load(cls, model_dir: str, verbose: bool = True, config_name: str = "config.json"):
        # TODO: рассмотреть случай обучения
        enc_config = json.load(open(os.path.join(model_dir, 'config_encoder.json')))
        config = {
            "data": enc_config["data"],
            "model": {
                "joint_dir": model_dir
            }
        }
        char2id = json.load(open(os.path.join(model_dir, 'char_encodings.json')))
        model = cls(config=config, char2id=char2id)
        return model

    def restore(self, model_dir=None, model_name="model.hdf5"):
        # TODO: рассмотреть случай обучения
        model_dir = model_dir or self.config["model"]["joint_dir"]
        self.enc_inference.load_weights(os.path.join(model_dir, "encoder.hdf5"))
        self.dec_inference.load_weights(os.path.join(model_dir, "decoder.hdf5"))
