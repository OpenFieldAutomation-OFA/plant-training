import time
import cv2
import numpy as np
import os
import csv
import onnxruntime as ort

model_path = '/mnt/c/finetuned.onnx'
image_file = 'sugarbeet.jpg'
class_mapping_file = 'class_mapping.txt'
species_mapping_file = 'species_id_to_name.txt'
dirname = os.path.dirname(__file__)

# read and resize image
image = cv2.imread(os.path.join(dirname, image_file))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
target_size = 518
height, width = image.shape[:2]
if width < height:
    new_width = target_size
    new_height = int(target_size * height / width)
else:
    new_height = target_size
    new_width = int(target_size * width / height)
image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
start_x = (new_width - target_size) // 2
start_y = (new_height - target_size) // 2
image = image[start_y:start_y + target_size, start_x:start_x + target_size]
mean=[123.675, 116.28, 103.53]
std=[58.395, 57.12, 57.375]
image = (image - mean) / std
image = np.transpose(image, (2, 0, 1))
input_tensor = np.expand_dims(image, axis=0)
input_tensor = input_tensor.astype(np.float32)

# load network
providers = [
    # 'TensorrtExecutionProvider',
    # 'CUDAExecutionProvider',
    'CPUExecutionProvider'
]
session = ort.InferenceSession(model_path, providers=providers)
input_name = session.get_inputs()[0].name

# forward pass
start = time.time()
outputs = session.run(None, {input_name: input_tensor})
end = time.time()
output_name = session.get_outputs()[0].name
predictions = outputs[0][0]
max_index = np.argmax(predictions)

# load mappings
class_mapping = {}
species_mapping = {}
class_mapping_file = os.path.join(dirname, class_mapping_file)
species_mapping_file = os.path.join(dirname, species_mapping_file)
with open(class_mapping_file) as f:
    class_mapping = {i: int(line.strip()) for i, line in enumerate(f)}
with open(species_mapping_file, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter=';')
    next(reader)
    for row in reader:
        species_id = int(row[0].strip('"'))
        species_name = row[1].strip('"')
        species_mapping[species_id] = species_name

# map output to species
species_id = class_mapping[max_index]
species_name = species_mapping[species_id]

print(f'Inference time: {end-start}')
print(f'Predicted class: {species_id}, {species_name}')
print(f'Confidence: {predictions[max_index]*100} %')
