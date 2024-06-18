# Dataset Preparation
1. Set the `DATA_DIR` environment variable to your desired data folder.
    ```bash
    export DATA_DIR=/your/data
    ```
2. Create the directories where data will be saved.
    ```bash
    mkdir -p $DATA_DIR/caw/download $DATA_DIR/caw/classification $DATA_DIR/plantclef
    ```
3. Download and extract the CAW dataset (10GB).
    ```bash
    wget -i caw_data.txt -P /tmp/caw
    for f in /tmp/caw/*.tar; do tar xvf "$f" -C $DATA_DIR/caw/download; done
    rm -r /tmp/caw
    ```
4. Convert the CAW dataset to image classification folder.
    ```bash
    python caw_to_class.py --bboxes_dir $DATA_DIR/caw/download/bboxes --images_dir $DATA_DIR/caw/download/images --classification_dir $DATA_DIR/caw/classification
    ```
5. Download and extract the PlantCLEF 2024 dataset (281 GB).
    ```bash
    wget https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata.tar -P /tmp/plantclef
    tar xvf /tmp/plantclef/PlantCLEF2024singleplanttrainingdata.tar -C $DATA_DIR/plantclef --strip-components=1
    rm -r /tmp/plantclef
    wget https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata.csv -P $DATA_DIR/plantclef
    ```
6. Create the MMPretrain annotation files (will be saved in the [annotation folder](../annotation)).
    ```bash
    python create_ann.py --caw_dir $DATA_DIR/caw/classification --plantclef_dir $DATA_DIR/plantclef --annotation_dir /caw/annotation
    ```
The last step should not be necessary, as the annotation files are saved on Git already.