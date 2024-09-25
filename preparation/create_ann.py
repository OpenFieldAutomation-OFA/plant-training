import argparse
import os
import pandas as pd

"""
This script creates the annotation files for MMPretrain.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--caw_dir',
                    default="/mnt/data/caw/classification",
                    help="Directory of CAW classification data.")
parser.add_argument('--plantclef_dir',
                    default="/mnt/data/plantclef",
                    help="Directory of PlantCLEF classification data.")
args = parser.parse_args()

maize_id = 1363500
sugarbeet_id = 1363199

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

caw_maize = caw.copy()
caw_maize['class'] = (caw['class'] == maize_id).astype(int)
caw_sugarbeet = caw.copy()
caw_sugarbeet['class'] = (caw['class'] == sugarbeet_id).astype(int)

# map species to class number
class_mapping = os.path.join(dirname, 'class_mapping.txt')
with open(class_mapping) as f:
    class_mappings = {int(line.strip()): i for i, line in enumerate(f)}

caw['class'] = caw['class'].map(class_mappings)

# shuffle dataframe
caw = caw.sample(frac=1, random_state=0).reset_index(drop=True)
caw_maize = caw_maize.sample(frac=1, random_state=0).reset_index(drop=True)
caw_sugarbeet = caw_sugarbeet.sample(frac=1, random_state=0).reset_index(drop=True)

train_size = int(0.7 * len(caw))
val_size = int(0.15 * len(caw))
test_size = len(caw) - train_size - val_size
train_caw = caw[:train_size]
val_caw = caw[train_size:train_size + val_size]
test_caw = caw[train_size + val_size:]

train_size = int(0.7 * len(caw_maize))
val_size = int(0.15 * len(caw_maize))
test_size = len(caw_maize) - train_size - val_size
train_caw_maize = caw_maize[:train_size]
val_caw_maize = caw_maize[train_size:train_size + val_size]
test_caw_maize = caw_maize[train_size + val_size:]

train_size = int(0.7 * len(caw_sugarbeet))
val_size = int(0.15 * len(caw_sugarbeet))
test_size = len(caw_sugarbeet) - train_size - val_size
train_caw_sugarbeet = caw_sugarbeet[:train_size]
val_caw_sugarbeet = caw_sugarbeet[train_size:train_size + val_size]
test_caw_sugarbeet = caw_sugarbeet[train_size + val_size:]

#### PlantCLEF ####
plantclef_dir = args.plantclef_dir

plantclef_metadata = os.path.join(args.plantclef_dir, 'PlantCLEF2024singleplanttrainingdata.csv')
# plantclef_metadata = 'PlantCLEF2024singleplanttrainingdata.csv'
plantclef = pd.read_csv(plantclef_metadata, delimiter=";",
                        dtype={'partner': 'string'})

plantclef['class'] = plantclef['species_id'].map(class_mappings)
plantclef['filename'] = plantclef['species_id'].astype(str) + '/' + plantclef['image_name']

plantclef_maize = plantclef.copy()
plantclef_maize['class'] = (plantclef_maize['species_id'] == maize_id).astype(int)
plantclef_sugarbeet = plantclef.copy()
plantclef_sugarbeet['class'] = (plantclef_sugarbeet['species_id'] == maize_id).astype(int)

leaf = plantclef[(plantclef['organ'] == 'leaf')][['filename', 'class', 'learn_tag']]
train_leaf = leaf[leaf['learn_tag'] == 'train'].drop('learn_tag',
                                                             axis=1)
val_leaf = leaf[leaf['learn_tag'] == 'val'].drop('learn_tag', axis=1)
test_leaf = leaf[leaf['learn_tag'] == 'test'].drop('learn_tag', axis=1)

leaf_maize = plantclef_maize[(plantclef_maize['organ'] == 'leaf')][['filename', 'class', 'learn_tag']]
train_leaf_maize = leaf_maize[leaf_maize['learn_tag'] == 'train'].drop('learn_tag',
                                                             axis=1)
val_leaf_maize = leaf_maize[leaf_maize['learn_tag'] == 'val'].drop('learn_tag', axis=1)
leaf_sugarbeet = plantclef_sugarbeet[(plantclef_sugarbeet['organ'] == 'leaf')][['filename', 'class', 'learn_tag']]
train_leaf_sugarbeet = leaf_sugarbeet[leaf_sugarbeet['learn_tag'] == 'train'].drop('learn_tag',
                                                             axis=1)
val_leaf_sugarbeet = leaf_sugarbeet[leaf_sugarbeet['learn_tag'] == 'val'].drop('learn_tag', axis=1)

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

train_plantclefcaw = pd.concat([train_caw, train_plantclef])
val_plantclefcaw = pd.concat([val_caw, val_plantclef])
test_plantclefcaw = pd.concat([test_caw, test_plantclef])

train_plantclefleafcaw = pd.concat([train_caw, train_leaf])
val_plantclefleafcaw = pd.concat([val_caw, val_leaf])
test_plantclefleafcaw = pd.concat([test_caw, test_leaf])

train_plantclefleafcaw_maize = pd.concat([train_caw_maize, train_leaf_maize])
val_plantclefleafcaw_maize = pd.concat([val_caw_maize, val_leaf_maize])
train_plantclefleafcaw_sugarbeet = pd.concat([train_caw_sugarbeet, train_leaf_sugarbeet])
val_plantclefleafcaw_sugarbeet = pd.concat([val_caw_sugarbeet, val_leaf_sugarbeet])

# save annotation files
train_caw.to_csv(os.path.join(dirname, '../annotation/caw_train.txt'), sep=' ', header=None,
                  index=False)
val_caw.to_csv(os.path.join(dirname, '../annotation/caw_val.txt'), sep=' ', header=None,
                 index=False)
test_caw.to_csv(os.path.join(dirname, '../annotation/caw_test.txt'), sep=' ', header=None,
                index=False)
train_plantclef.to_csv(os.path.join(dirname, '../annotation/plantclef_train.txt'), sep=' ', header=None,
                  index=False)
val_plantclef.to_csv(os.path.join(dirname, '../annotation/plantclef_val.txt'), sep=' ', header=None,
                 index=False)
# test_plantclef.to_csv(os.path.join(dirname, '../annotation/plantclef_test.txt'), sep=' ', header=None,
#                 index=False)
train_plantclefcaw.to_csv(os.path.join(dirname, '../annotation/plantclefcaw_train.txt'), sep=' ', header=None,
                  index=False)
val_plantclefcaw.to_csv(os.path.join(dirname, '../annotation/plantclefcaw_val.txt'), sep=' ', header=None,
                 index=False)
# plantclefcaw_test.to_csv(os.path.join(dirname, '../annotation/plantclefcaw_test.txt'), sep=' ', header=None,
#                  index=False)
train_plantclefleafcaw.to_csv(os.path.join(dirname, '../annotation/plantclefleafcaw_train.txt'), sep=' ', header=None,
                  index=False)
val_plantclefleafcaw.to_csv(os.path.join(dirname, '../annotation/plantclefleafcaw_val.txt'), sep=' ', header=None,
                 index=False)
# plantclefleafcaw_test.to_csv(os.path.join(dirname, '../annotation/plantclefleafcaw_test.txt'), sep=' ', header=None,
#                 index=False)
train_plantclefleafcaw_maize.to_csv(os.path.join(dirname, '../annotation/plantclefleafcaw_maize_train.txt'), sep=' ', header=None,
                  index=False)
val_plantclefleafcaw_maize.to_csv(os.path.join(dirname, '../annotation/plantclefleafcaw_maize_val.txt'), sep=' ', header=None,
                 index=False)
train_plantclefleafcaw_sugarbeet.to_csv(os.path.join(dirname, '../annotation/plantclefleafcaw_sugarbeet_train.txt'), sep=' ', header=None,
                  index=False)
val_plantclefleafcaw_sugarbeet.to_csv(os.path.join(dirname, '../annotation/plantclefleafcaw_sugarbeet_val.txt'), sep=' ', header=None,
                 index=False)
test_caw_maize.to_csv(os.path.join(dirname, '../annotation/caw_maize_test.txt'), sep=' ', header=None,
                index=False)
test_caw_maize.to_csv(os.path.join(dirname, '../annotation/caw_sugarbeet_test.txt'), sep=' ', header=None,
                index=False)