# Dataset Preparation
1. Download and extract the CAW dataset. Replace `/mnt/data/caw/download` with your desired directory.
    ```bash
    wget -i caw_data.txt -P /tmp/caw
    for f in /tmp/caw/*.tar; do tar xvf "$f" -C /mnt/data/caw/download; done
    rm -r /tmp/caw
    ```
2. Create the folder with classification images.
    ```bash
    python caw_class_data.py --bboxes_dir /caw/download/bboxes --images_dir /caw/download/images --classification_dir /caw/classification
    ```
3. Create the MMPretrain annotation file.
    ```bash
    python caw_class_data_ann.py --classification_dir /caw/classification --annotation_dir /caw/annotation
    ```
