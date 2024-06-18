import argparse
import timm
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--plantclef_model',
                    default="/mnt/data/plantclef/pretrained/model_best.pth.tar",
                    help="Path to pretrained PlantCLEF model.")
parser.add_argument('--model_dir',
                    default="/mnt/data/plantclef/pretrained",
                    help="Directory where models are saved.")
args = parser.parse_args()

plantclef_model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m',
                                    pretrained=False, num_classes=7806,
                                    checkpoint_path=args.plantclef_model)

dinov2_model = timm.create_model('vit_base_patch14_reg4_dinov2',
                                 pretrained=True, num_classes=7806)

torch.save(plantclef_model.state_dict(), os.path.join(args.model_dir, 'plantclef.pth'))
torch.save(dinov2_model.state_dict(), os.path.join(args.model_dir, 'dinov2.pth'))

