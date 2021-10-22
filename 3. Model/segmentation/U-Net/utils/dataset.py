import glob
import cv2

import os
from random import sample

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms

from pycocotools.coco import COCO

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2


class PennFudanDataset(Dataset):
    def __init__(self, base_url, transform):
        self.base_url = base_url
        self.transform = transform
        
        self.images = glob.glob(os.path.join(self.base_url, 'PNGImages', '*'))
        self.masks = glob.glob(os.path.join(self.base_url, 'PedMasks', '*'))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.open(image_path)
        mask = Image.open(mask_path)
        mask = np.array(mask)
        
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        
        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = np.array(boxes, dtype=np.float32)
        masks = np.array(masks, dtype=np.uint8)
        labels = np.ones((num_objs), dtype=np.int64)
        
        if self.transform:
            augmentations = self.transform(image=img, bboxes=boxes, masks=list(masks), category_id=labels)
            img = augmentations['image']
            masks = augmentations['masks']
            boxes = augmentations['bboxes']
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        
        return img, target
    