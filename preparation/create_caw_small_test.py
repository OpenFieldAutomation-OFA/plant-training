import os
import pandas as pd
from PIL import Image

dirname = os.path.dirname(__file__)
caw_test = os.path.join(dirname, '../annotation/caw_test.txt')
caw_dir = '/mnt/c'

small_images = []
with open (caw_test, 'r') as file:
    for line in file:
        words = line.strip().split()
        image = Image.open(os.path.join(caw_dir, words[0]))
        width, height = image.size
        if width * height < 32**2:
            small_images.append(words)

caw_small = pd.DataFrame(small_images, columns=['filename', 'class'])
caw_small.to_csv(os.path.join(dirname, '../annotation/caw_test_small.txt'), sep=' ', header=None,
                index=False)
