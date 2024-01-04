#!/bin/bash

env_name=voice-search-server

source /disk/scratch4/unmute/miniconda3/etc/profile.d/conda.sh
conda create -y --name $env_name python=3.7
conda activate $env_name
conda install -y -c pykaldi pykaldi
conda install -y -c conda-forge scikit-learn onnx onnxruntime grpcio
pip3 install click
