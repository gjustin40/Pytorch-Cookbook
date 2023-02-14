import os
import glob
import random

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision.utils import draw_segmentation_masks
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import numpy as np

from dataset import MyCocoLimit
from utils import collate_fn
from engine import train_one_epoch, evaluate

random_seed = 40
torch.set_printoptions(precision=5, sci_mode=False)
torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

# DataLoader
train_transform = A.Compose([
    # A.Rotate(limit=40, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', min_area=1, label_fields=['category_id']))

test_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_id']))

print(f'Preparing Dataset....COCO Dataset')
train_path = r'C:\Users\gjust\Documents\Github\data\COCO\train2017'
train_ann = r'C:\Users\gjust\Documents\Github\data\COCO\annotations\instances_train2017.json'
test_path = r'C:\Users\gjust\Documents\Github\data\COCO\val2017'
test_ann = r'C:\Users\gjust\Documents\Github\data\COCO\annotations\instances_val2017.json'
class_list = ['person']

trainset = MyCocoLimit(root=train_path, annFile=train_ann, class_list=class_list, transform=train_transform)
testset = MyCocoLimit(root=test_path, annFile=test_ann, class_list=class_list, transform=test_transform)
train_loader = DataLoader(trainset, batch_size=4, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(testset, batch_size=4, shuffle=False, collate_fn=collate_fn)


# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
print(f'Using {device}')


# Model
print(f'Preparing Model....Faster RCNN')
model = fasterrcnn_resnet50_fpn(pretrained=True)
for p in model.parameters():
    p.requires_grad = False
    
num_classes = len(class_list) + 1
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)


# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)


# Training
EPOCHS = 30

for epoch in range(EPOCHS):
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1000)
    evaluate(model, test_loader, device=device)