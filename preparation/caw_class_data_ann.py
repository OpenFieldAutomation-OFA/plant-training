import argparse
import os
import pandas as pd

"""
This script creates annotation files for MMPretrain using the folder
created in the caw_class_data.py script.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--classification_dir',
                    default="/mnt/c/caw/classification",
                    help="Directory of classification data.")
parser.add_argument('--annotation_dir',
                    default="/mnt/c/caw/annotation",
                    help="Directory where annotation files are saved.")

args = parser.parse_args()

classification_dir = args.classification_dir

filenames = []
classes = []
for class_name in os.listdir(classification_dir):
    class_filenames = os.listdir(os.path.join(classification_dir, class_name))
    class_filenames = [os.path.join(class_name, s) for s in class_filenames]
    filenames.extend(class_filenames)
    classes.extend([int(class_name)] * len(class_filenames))

df = pd.DataFrame({'filename': filenames, 'class': classes})

# map species to class number
class_mapping = os.path.join(os.path.dirname(__file__), 'class_mapping.txt')
with open(class_mapping) as f:
    class_mappings = {int(line.strip()): i for i, line in enumerate(f)}
df['class'] = df['class'].map(class_mappings)

# shuffle dataframe
df = df.sample(frac=1, random_state=0).reset_index(drop=True)

train_size = int(0.9 * len(df))
val_size = int(0.05 * len(df))
test_size = len(df) - train_size - val_size

train_df = df[:train_size]
val_df = df[train_size:train_size + val_size]
test_df = df[train_size + val_size:]

# save annotation files
train_df.to_csv('annotation/caw_train.txt', sep=' ', header=None,
                  index=False)
val_df.to_csv('annotation/caw_val.txt', sep=' ', header=None,
                 index=False)
test_df.to_csv('annotation/caw_test.txt', sep=' ', header=None,
                index=False)