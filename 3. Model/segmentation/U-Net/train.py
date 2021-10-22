import os
import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from torchvision.utils import draw_segmentation_masks

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2

from utils.dataset import PennFudanDataset
from utils.utils import collate_fn
from utils.engine import train_one_epoch, evaluate
from model import UNet

# Dataloader
train_transform = A.Compose([
    A.Rotate(limit=40, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', min_area=1, label_fields=['category_id']))

test_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', min_area=1, label_fields=['category_id']))

base_url = r'C:\Users\gjust\Documents\Github\data\PennFudanPed'
trainset = PennFudanDataset(base_url=base_url, transform=train_transform)
testset = PennFudanDataset(base_url=base_url, transform=test_transform)

train_loader = DataLoader(trainset, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(testset, batch_size=2, shuffle=False, collate_fn=collate_fn)


# Cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Model
model = UNet()
model.to(device)

print(model)