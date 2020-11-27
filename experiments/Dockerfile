FROM odsai/python-gpu

#RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#        build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev

#RUN pip install https://github.com/kpu/kenlm/archive/master.zip
#RUN yum -y install wget
#RUN git clone https://github.com/kpu/kenlm.git && \
#    cd ./kenlm && \
#    mkdir -p build && \
#    cd build && \
#    cmake .. && \
#    make -j 4

RUN pip install Cython
#RUN gir clone https://github.com/glample/fastBPE.git \
#    && cd fastBPE \
#    && g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast \
#    && python setup.py install
