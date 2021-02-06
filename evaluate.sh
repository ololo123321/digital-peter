#!/usr/bin/env bash

predictions_dir="$(pwd)/predictions/valid_predictions"
answers_dir="$(pwd)/data/valid_texts"

predictions_dir_mnt=/predictions
answers_dir_mnt=/answers

image=ololo123321/tensorflow_kenlm:1.1.4

docker run -it \
    -v $(pwd):/tmp \
    -v ${predictions_dir}:${predictions_dir_mnt} \
    -v ${answers_dir}:${answers_dir_mnt} \
    -w /tmp \
    ${image} python evaluate.py \
        --predictions_dir=${predictions_dir_mnt} \
        --answers_dir=${answers_dir_mnt}