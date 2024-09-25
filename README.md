# plant-training
This repository contains all files that were used to train a plant classification network that can effectively distinguish plants and weeds in agricultural fields. The performance of the network was tested on sugar beet and corn plants, but it should work for other crops as well.

The network only **classifies** plants and does not localize them. It can be used together with a color-based segmentation algorithm (see [our main repo](https://github.com/OpenFieldAutomation-OFA/ros-weed-control)) to both localize and classify weeds. The results are shown in the image below. The main crop (corn) is marked blue, while all the other classes are marked red.

![Weed detection](example_classified.png)

## Models
We publish two models: a base and a small vision transformer (ViT). Both are trained on the leaf PLantCLEF & CropAndWeed dataset with all classes, as explained [below](#training).

| Model | Parameters | Accuracy | Corn F2 | Sugar Beet F2 | Download |
| --- | --- | --- | --- | --- | --- |
| ViT-B + head | 92.6 M | 84.61&nbsp;% | 94.72&nbsp;% | 91.19&nbsp;% | [PyTorch](https://github.com/OpenFieldAutomation-OFA/plant-training/releases/download/v0.0.0/finetuned.pth), [ONNX](https://github.com/OpenFieldAutomation-OFA/plant-training/releases/download/v0.0.0/finetuned.onnx) |
| ViT-S + head | 25.1 M | 82.28&nbsp;% | 93.75&nbsp;% | 88.37&nbsp;% | [PyTorch](https://github.com/OpenFieldAutomation-OFA/plant-training/releases/download/v0.0.0/finetuned_small.pth), [ONNX](https://github.com/OpenFieldAutomation-OFA/plant-training/releases/download/v0.0.0/finetuned_small.onnx) |

### Usage
To load the PyTorch checkpoint, you must have [PyTorch](https://pytorch.org/get-started/locally/) and [MMPretrain](https://mmpretrain.readthedocs.io/en/latest/get_started.html) installed. An example of loading the model and inferencing an image is shown in [`pytorch_ex.py`](sample_usage/pytorch_ex.py).

To use the exported ONNX model, only the [ONNX runtime](https://onnxruntime.ai/docs/install/) needs to be installed. Inference is demonstrated in [`onnx_ex.py`](sample_usage/onnx_ex.py).

## Reproduce Training
### Setup Environment
On Linux you should be able to set everything up by running `setup.sh`. This will create a python venv, install [PyTorch](https://pytorch.org/get-started/locally/) and [MMPretrain](https://mmpretrain.readthedocs.io/en/latest/get_started.html), and copy the custom metric into the correct location.

### Prepare Datasets
We use two datasets for the training:
- [PlantCLEF 2024](https://www.imageclef.org/node/315)
- [CropAndWeed](https://github.com/cropandweed/cropandweed-dataset)

The readme in the [preparation folder](./preparation) explains how to download and prepare the datasets.

Note that the CropAndWeed (CAW) dataset has far fewer classes than the PlantCLEF dataset (74 vs. 7806) and is more imbalanced. This leads to a strong overrepresentation of certain classes such as corn, sugar beet, and green bristlegrass. However, the data is intentionally not rebalanced as this is the expected distribution in agricultural fields. We also only use CAW images in the test set because it represents our target domain much better than PlantCLEF.

### Training
We train our network with the MMPretrain toolbox. The config files are stored in the [config folder](./config). All config files are based on a ViT base architecture pre-trained with [Dinov2](https://mmpretrain.readthedocs.io/en/stable/papers/dinov2.html).

Below is a table describing all config files.

| Config File | Dataset | Classes | Accuracy | Corn F2 | Sugar Beet F2 |
| --- | --- | --- | --- | --- | --- |
| [`plantclef.py`](configs/plantclef.py) | Entire PLantCLEF | All | 9.94&nbsp;% | - | - |
| [`caw.py`](configs/caw.py) | CAW | All | 72.47&nbsp;% | - | - |
| [`plantclefcaw.py`](configs/plantclefcaw.py) | Entire PLantCLEF & CAW | All | 74.69&nbsp;% | - | - |
| [`plantclefleafcaw.py`](configs/plantclefleafcaw.py) | Leaf PLantCLEF & CAW | All | **75.41&nbsp;%** | **92.24&nbsp;%** | **82.69&nbsp;%** |
| [`plantclefleafcaw_maize.py`](configs/plantclefleafcaw_maize.py) | Leaf PLantCLEF & CAW | Binary: corn & weed | - | 84.35&nbsp;% | - |
| [`plantclefleafcaw_sugarbeet.py`](configs/plantclefleafcaw_sugarbeet.py) | Leaf PLantCLEF & CAW | Binary: sugar beet & weed | - | - | 69.57&nbsp;% |

For the ablation the backbone was frozen and only the classification head was trained. The best performing model [`plantclefleafcaw.py`](configs/plantclefleafcaw.py) was then further finetuned on both the backbone and head with [`finetune.py`](configs/finetune.py). This process was repeated for the smaller ViT architecture. The resulting models are the ones described in the [models section](#models).


The training and evaluation of the different config files can be reproduced with the MMPretrain scripts.
```bash
# Train and test on a single GPU
python mmpretrain/tools/train.py configs/plantclefleafcaw.py
python mmpretrain/tools/test.py configs/plantclefleafcaw.py work_dirs/plantclefleafcaw/epoch_12.pth

# Train and test on two GPUs
bash mmpretrain/tools/dist_train.sh configs/plantclefleafcaw.py 2
bash mmpretrain/tools/dist_test.sh configs/plantclefleafcaw.py work_dirs/plantclefleafcaw/epoch_12.pth 2
```

### Convert Trained Model to ONNX
To export our models to ONNX we use [MMDeploy](https://mmdeploy.readthedocs.io/en/latest/get_started.html).

```bash
python mmdeploy/tools/deploy.py \
    mmdeploy/configs/mmpretrain/classification_onnxruntime_static.py \
    configs/finetune.py \
    work_dirs/finetune/epoch_5.pth \
    sample_usage/sugarbeet.jpg \
    --work-dir mmdeploy_model \
    --device cpu \
    --show \
    --dump-info
```