# plant-training
This repository contains all files that were used to train our plant classification network.

Our final model can be downloaded from the **Releases** page.

# Setup
To reproduce the training, you need Python with the following packages installed:
- [PyTorch](https://pytorch.org/get-started/locally/)
- [MMPretrain](https://mmpretrain.readthedocs.io/en/latest/get_started.html)

On Linux you should be able to set everything up by running `setup.sh`.

# Datasets
We use two datasets for the training:
- [PlantCLEF 2024](https://www.imageclef.org/node/315)
- [CropAndWeed](https://github.com/cropandweed/cropandweed-dataset)

The README in the [preparation folder](./preparation) explains how to download and prepare the datasets.

# Training
We train our model with the MMPretrain toolbox. The config files are stored in the [config folder](./config). All configs are based on a ViT base patch 14 architecture pre-trained with [Dinov2](https://mmpretrain.readthedocs.io/en/stable/papers/dinov2.html).

Before training make sure the following environment variable is set.
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Then you can start the training on one or mulitple GPUs.
```bash
python mmpretrain/tools/train.py configs/leafcaw.py  # Single GPU
bash mmpretrain/tools/dist_train.sh configs/leafcaw.py 2  # Two GPUs
```

# Test
Testing a trained network can also be done on one or multiple GPUs.
```bash
bash mmpretrain/tools/test.py configs/leafcaw.py work_dirs/leafcaw/epoch_12.pth  # Single GPU
bash mmpretrain/tools/dist_test.sh configs/leafcaw.py work_dirs/leafcaw/epoch_12.pth 2  # Two GPUs
```

## Convert Model to ONNX
Our model is deployed on a Nivida Jetson (see [main repo](https://github.com/OpenFieldAutomation-OFA/ros-weed-control)). We first export the model to the ONNX format and then use TensorRT on the Jetson for inference.

The ONNX model can be created with [MMDeploy](https://mmdeploy.readthedocs.io/en/latest/get_started.html).

```bash
python mmdeploy/tools/deploy.py \
    mmdeploy/configs/mmpretrain/classification_onnxruntime_static.py \
    configs/finetune.py \
    work_dirs/finetune/epoch_5.pth \
    /mnt/data/caw/classification/1363199/00dcd0ff0c50e304d43e519f0eafc849.jpg \
    --work-dir mmdeploy_model \
    --device cpu \
    --show \
    --dump-info
```