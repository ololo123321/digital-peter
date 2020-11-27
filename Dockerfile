FROM tensorflow/tensorflow:2.3.0-gpu
WORKDIR /app
COPY ./ctc_pyx ./ctc_pyx
COPY ./setup.py .
COPY ./requirements_docker.txt .
RUN python3 -m pip install -r requirements_docker.txt
RUN python3 setup.py build_ext --inplace