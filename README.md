# weed-detection

# Installation
Install Python, create a virtual environment and install the packages specified in `requirements.txt`.

# Setup Dataset
Right now our process consists of the following steps:
1. Download the CAW dataset.
2. Run the `caw_classification.py` script to create the classification dataset.
3. Download the pretrained ImageNet model.
4. Run the training script to finetune the ImageNet model on the CAW classification dataset.

# Download Dataset
To download and extract the CAW dataset, run the following commands.

```bash
wget -i caw_data.txt -P /tmp/caw
for f in /tmp/caw/*.tar; do tar xvf "$f" -C /caw/download; done
rm -r /tmp/caw
```

```bash
python caw_class_data.py 
```

