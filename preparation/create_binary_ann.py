import argparse
import os
import pandas as pd

"""
This script creates binary annotation files for MMPretrain.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--caw_dir',
                    default="/mnt/data/caw/classification",
                    help="Directory of CAW classification data.")
parser.add_argument('--plantclef_dir',
                    default="/mnt/data/plantclef",
                    help="Directory of PlantCLEF classification data.")
args = parser.parse_args()

dirname = os.path.dirname(__file__)

#### CAW #####
caw_dir = args.caw_dir

filenames = []
classes = []
for class_name in os.listdir(caw_dir):
    class_filenames = os.listdir(os.path.join(caw_dir, class_name))
    class_filenames = [os.path.join('caw', 'classification', class_name, s) for s in class_filenames]
    filenames.extend(class_filenames)
    classes.extend([int(class_name)] * len(class_filenames))

caw = pd.DataFrame({'filename': filenames, 'class': classes})


caw['class'] = (caw['class'] == 1363500).astype(int)

# shuffle dataframe
caw = caw.sample(frac=1, random_state=0).reset_index(drop=True)

train_size = int(0.9 * len(caw))
val_size = int(0.05 * len(caw))
test_size = len(caw) - train_size - val_size

train_caw = caw[:train_size]
val_caw = caw[train_size:train_size + val_size]
test_caw = caw[train_size + val_size:]

#### PlantCLEF ####
plantclef_dir = args.plantclef_dir

plantclef_metadata = os.path.join(args.plantclef_dir, 'PlantCLEF2024singleplanttrainingdata.csv')
# plantclef_metadata = 'PlantCLEF2024singleplanttrainingdata.csv'
plantclef = pd.read_csv(plantclef_metadata, delimiter=";",
                        dtype={'partner': 'string'})

plantclef['class'] = (plantclef['species_id'] == 1363199).astype(int)
plantclef['filename'] = plantclef['species_id'].astype(str) + '/' + plantclef['image_name']

leaf = plantclef[(plantclef['organ'] == 'leaf')][['filename', 'class', 'learn_tag']]
train_leaf = leaf[leaf['learn_tag'] == 'train'].drop('learn_tag',
                                                             axis=1)
val_leaf = leaf[leaf['learn_tag'] == 'val'].drop('learn_tag', axis=1)
test_leaf = leaf[leaf['learn_tag'] == 'test'].drop('learn_tag', axis=1)


plantclef = plantclef[['filename', 'class', 'learn_tag']]
train_plantclef = plantclef[plantclef['learn_tag'] == 'train'].drop('learn_tag',
                                                             axis=1)
val_plantclef = plantclef[plantclef['learn_tag'] == 'val'].drop('learn_tag', axis=1)
test_plantclef = plantclef[plantclef['learn_tag'] == 'test'].drop('learn_tag', axis=1)

train_plantclef['filename'] = 'plantclef' + '/' + 'train' + '/' + train_plantclef['filename']
val_plantclef['filename'] = 'plantclef' + '/' + 'val' + '/' + val_plantclef['filename']
test_plantclef['filename'] = 'plantclef' + '/' + 'test' + '/' + test_plantclef['filename']

train_leaf['filename'] = 'plantclef' + '/' + 'train' + '/' + train_leaf['filename']
val_leaf['filename'] = 'plantclef' + '/' + 'val' + '/' + val_leaf['filename']
test_leaf['filename'] = 'plantclef' + '/' + 'test' + '/' + test_leaf['filename']

train = pd.concat([train_caw, train_plantclef])
val = pd.concat([val_caw, val_plantclef])
test = pd.concat([test_caw, test_plantclef])

train_2 = pd.concat([train_caw, train_leaf])
val_2 = pd.concat([val_caw, val_leaf])
test_2 = pd.concat([test_caw, test_leaf])

# save annotation files
train_caw.to_csv(os.path.join(dirname, '../annotation_sugarbeet/caw_train.txt'), sep=' ', header=None,
                  index=False)
val_caw.to_csv(os.path.join(dirname, '../annotation_sugarbeet/caw_val.txt'), sep=' ', header=None,
                 index=False)
test_caw.to_csv(os.path.join(dirname, '../annotation_sugarbeet/caw_test.txt'), sep=' ', header=None,
                index=False)
train_plantclef.to_csv(os.path.join(dirname, '../annotation_sugarbeet/plantclef_train.txt'), sep=' ', header=None,
                  index=False)
val_plantclef.to_csv(os.path.join(dirname, '../annotation_sugarbeet/plantclef_val.txt'), sep=' ', header=None,
                 index=False)
test_plantclef.to_csv(os.path.join(dirname, '../annotation_sugarbeet/plantclef_test.txt'), sep=' ', header=None,
                index=False)
train_leaf.to_csv(os.path.join(dirname, '../annotation_sugarbeet/leaf_train.txt'), sep=' ', header=None,
                  index=False)
val_leaf.to_csv(os.path.join(dirname, '../annotation_sugarbeet/leaf_val.txt'), sep=' ', header=None,
                 index=False)
test_leaf.to_csv(os.path.join(dirname, '../annotation_sugarbeet/leaf_test.txt'), sep=' ', header=None,
                index=False)
train.to_csv(os.path.join(dirname, '../annotation_sugarbeet/train.txt'), sep=' ', header=None,
                  index=False)
val.to_csv(os.path.join(dirname, '../annotation_sugarbeet/val.txt'), sep=' ', header=None,
                 index=False)
test.to_csv(os.path.join(dirname, '../annotation_sugarbeet/test.txt'), sep=' ', header=None,
                 index=False)
train_2.to_csv(os.path.join(dirname, '../annotation_sugarbeet/train_2.txt'), sep=' ', header=None,
                  index=False)
val_2.to_csv(os.path.join(dirname, '../annotation_sugarbeet/val_2.txt'), sep=' ', header=None,
                 index=False)
test_2.to_csv(os.path.join(dirname, '../annotation_sugarbeet/test_2.txt'), sep=' ', header=None,
                index=False)