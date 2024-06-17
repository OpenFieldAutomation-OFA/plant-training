# weed-detection

# Installation
You need Python with the following packages installed:
- [PyTorch](https://pytorch.org/get-started/locally/)
- [MMPretrain](https://mmpretrain.readthedocs.io/en/latest/get_started.html)
- [timm](https://timm.fast.ai/)
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/)

On Linux you should be able to set everything up by running `setup.sh`.

# Data Preparation
See `preparation` folder.

# Fine-tuning
`python mmpretrain/tools/train.py plantclef-pretrained.py`

Two gpus:
CUDA_VISIBLE_DEVICES=0 PORT=29500 bash mmpretrain/tools/dist_train.sh plantclef-pretrained.py 1
CUDA_VISIBLE_DEVICES=1 PORT=29501 bash mmpretrain/tools/dist_train.sh scratch-allclasses.py 1