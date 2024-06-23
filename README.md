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
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python mmpretrain/tools/train.py caw.py
```

Two gpus different jobs:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash mmpretrain/tools/dist_train.sh dinov2.py 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 bash mmpretrain/tools/dist_train.sh dinov2-nomix.py 4
```

Two gpus same job:
```bash
bash mmpretrain/tools/dist_train.sh leafcaw.py 1
```

Test:
```bash
bash mmpretrain/tools/dist_test.sh dinov2-a100.py work_dirs/dinov2/epoch_1.pth 2
```

