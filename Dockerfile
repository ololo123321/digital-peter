FROM tensorflow/tensorflow:2.3.0-gpu
WORKDIR /app
COPY ./ctc_pyx /app
COPY ./requirements_wo_tf.txt /app
RUN python3 -m pip install -r requirements_wo_tf.txt
RUN python3 /app/ctc_pyx/setup.py build_ext --inplace
