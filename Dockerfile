FROM tensorflow/tensorflow:2.3.0-gpu
WORKDIR /app
COPY ./ctc_pyx /app
COPY ./requirements.txt /app
RUN python3 -m pip install -r requirements.txt
RUN python3 ./ctc_pyx/setup.py build_ext --inplace
