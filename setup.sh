#!/bin/bash
set -e
apt update
apt install python3-venv -y
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.4.1 torchvision==0.19.1  --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/open-mmlab/mmpretrain.git
cp custom_metric/* mmpretrain/mmpretrain/evaluation/metrics
cd mmpretrain
pip install -U openmim && mim install -e .
cd ..
pip install mmdeploy mmdeploy-runtime onnxruntime
git clone https://github.com/open-mmlab/mmdeploy.git