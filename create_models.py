import argparse
import timm
import torch
import os
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--plantclef_model',
                    default="model_best.pth.tar",
                    help="Path to pretrained PlantCLEF model.")
parser.add_argument('--model_dir',
                    default="/mnt/data/plantclef2/pretrained",
                    help="Directory where models are saved.")
args = parser.parse_args()

plantclef_model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m',
                                    pretrained=False, num_classes=7806,
                                    checkpoint_path=args.plantclef_model)

empty_head = nn.Linear(plantclef_model.embed_dim, 7806, bias=True)
head_state_dict = empty_head.state_dict()
state_dict = plantclef_model.state_dict()
state_dict['head.weight'] = head_state_dict['weight']
state_dict['head.bias'] = head_state_dict['bias']

torch.save(state_dict, os.path.join(args.model_dir, 'plantclef_empty_head.pth'))

