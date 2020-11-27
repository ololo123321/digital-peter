#!/usr/bin/env bash

input_dir=./data/sample_images
output_dir=./predictions/sample_predictions

#input_dir=./data/valid_images
#output_dir=./predictions/valid_predictions

rm -r ${output_dir}
mkdir ${output_dir}
chmod 777 ${output_dir}

input_dir_mnt=/data
output_dir_mnt=/output

time docker run -it \
    -v $(pwd):/tmp \
    -v ${input_dir}:${input_dir_mnt} \
    -v ${output_dir}:${output_dir_mnt} \
    -w /tmp \
    --gpus all \
    tensorflow_kenlm:1.1.4 python predict.py \
        --input_dir=${input_dir_mnt} \
        --output_dir=${output_dir_mnt} \
        --w_ctc=50 \
        --w_birnn=15 \
        --w_transformer=10 \
        --w_joint=25 \
        --alpha=0.7 \
        --beta=5 \
        --beam_width=100 \
        --val_mode
