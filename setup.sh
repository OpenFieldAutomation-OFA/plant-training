#!/bin/bash
set -e
apt update
apt install python3-venv -y
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision
git clone https://github.com/open-mmlab/mmpretrain.git
cp custom_metric/* mmpretrain/mmpretrain/evaluation/metrics
cd mmpretrain
pip install -U openmim && mim install -e .
pip install timm pandas pyside6