#!/bin/bash
set -e
apt install python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -U openmim && mim install mmpretrain
pip install timm pandas