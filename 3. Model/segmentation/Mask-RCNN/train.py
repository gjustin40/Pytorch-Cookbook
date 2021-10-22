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

from dataset import PennFudanDataset
from utils import collate_fn
from engine import train_one_epoch, evaluate

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

# Model
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

num_classes = 2

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                    hidden_layer,
                                                    num_classes)



# Training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

EPOCHS = 10

for epoch in range(EPOCHS):
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    evaluate(model, test_loader, device=device)