# weed-detection

# Installation
You need Python with the following packages installed:
- [PyTorch](https://pytorch.org/get-started/locally/)
- [MMPretrain](https://mmpretrain.readthedocs.io/en/latest/get_started.html)
- [MMDeploy](https://mmdeploy.readthedocs.io/en/latest/get_started.html)

On Linux you should be able to set everything up by running `setup.sh`.

# Data Preparation
See `preparation` folder.

# Fine-tuning
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python mmpretrain/tools/train.py configs/caw.py
```

Two gpus same job:
```bash
bash mmpretrain/tools/dist_train.sh configs/leafcaw.py 2
```

Test:
```bash
bash mmpretrain/tools/dist_test.sh configs/dinov2-a100.py work_dirs/dinov2/epoch_1.pth 2
```

Convert to ONNX:
```bash
python mmdeploy/tools/deploy.py \
    mmdeploy/configs/mmpretrain/classification_onnxruntime_dynamic.py \
    configs/finetune.py \
    /mnt/data/saved/20240724_105901_finetune.pth \
    /mnt/data/caw/classification/1363199/00dcd0ff0c50e304d43e519f0eafc849.jpg \
    --work-dir mmdeploy_model \
    --device cpu \
    --show \
    --dump-info
```