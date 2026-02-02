# -------------------------------------------------------------
# Base Image: NVIDIA CUDA 12.4 Devel on Ubuntu 22.04
# -------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    git \
    vim \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    conda config --remove channels defaults || true && \
    conda init bash

ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda init bash

RUN conda create -n quest -c conda-forge --override-channels python=3.10 fmt=10 cmake -y

RUN echo "conda activate quest" >> ~/.bashrc

RUN pip install ninja packaging

WORKDIR /quest
