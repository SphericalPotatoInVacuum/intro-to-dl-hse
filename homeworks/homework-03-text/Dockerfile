FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y software-properties-common
RUN apt update && add-apt-repository ppa:deadsnakes/ppa
RUN apt update && apt install -y python3.9 python3.9-dev python3-pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

COPY requirements.txt .
RUN pip3 install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html

COPY word_linear_glove_500d word_linear_glove_500d/
COPY data data/
COPY cli cli/
COPY Makefile text.py ./

ENTRYPOINT [ "make" ]
