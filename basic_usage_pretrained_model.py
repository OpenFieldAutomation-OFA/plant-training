import pandas as pd
from PIL import Image
import timm
import torch
import os


def load_class_mapping(class_list_file):
    with open(class_list_file) as f:
        class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
    return class_index_to_class_name


def load_species_mapping(species_map_file):
    df = pd.read_csv(species_map_file, sep=';', quoting=1, dtype={'species_id': str})
    df = df.set_index('species_id')
    return  df['species'].to_dict()


pretrained_path = "model_best.pth.tar"
class_mapping = "preparation/class_mapping.txt"
species_mapping = "species_id_to_name.txt"
device = "cuda"

test = torch.load(pretrained_path)

cid_to_spid = load_class_mapping(class_mapping)
spid_to_sp = load_species_mapping(species_mapping)
    
device = torch.device(device)

model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(cid_to_spid), checkpoint_path=pretrained_path)
model = model.to(device)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

folder = "/mnt/data/seedlings/Maize"
folder = "/mnt/data/OPPD/DATA/images_plants/CENCY"
for image in os.listdir(folder):
    print(image)
    img = Image.open(os.path.join(folder, image))
    
    img = transforms(img).unsqueeze(0)
    img = img.to(device)
    output = model(img)  # unsqueeze single image into batch of 1
    top_probabilities, top_class_indices = torch.topk(output.softmax(dim=1) * 100, k=1)
    top_probabilities = top_probabilities.cpu().detach().numpy()
    top_class_indices = top_class_indices.cpu().detach().numpy()

    proba = top_probabilities[0,0]
    cid = top_class_indices[0,0]
    species_id = cid_to_spid[cid]
    species = spid_to_sp[species_id]
    print(species_id, species, proba)

    # for proba, cid in zip(top5_probabilities[0], top5_class_indices[0]):
    #     species_id = cid_to_spid[cid]
    #     species = spid_to_sp[species_id]
    #     print(species_id, species, proba)


