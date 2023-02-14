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

class MyCocoDetection(CocoDetection):
    def __init__(
        self, root, annFile, transform, remove_invalid_data=True, show=False
    ):
        super(myCocoDetection, self).__init__(root, annFile)
        
        # self.coco = COCO(annFile)
        # self.ids = list(sorted(self.coco.imgs.keys()))
        if remove_invalid_data:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anno = self.coco.loadAnns(ann_ids)
                # print(len(anno))
                if self.is_valid_data(anno):
                    ids.append(img_id)
            self.ids = ids
            
        self.transform = transform
        self.show = show
        
    # 물체가 없거나 bbox가 1보다 작은 경우 제외
    def is_valid_data(self, anno):
        if len(anno) == 0:
            return False
        for obj in anno:
            for o in obj['bbox'][2:]:
                if o <= 1:
                    return False
        return True
            
    
    def __getitem__(self, index: int):
        idx = self.ids[index]
        image = self._load_image(idx)
        target = self._load_target(idx)
        
        bboxes, labels = [], []
        for obj in target:
            xmin = obj['bbox'][0]
            ymin = obj['bbox'][1]
            xmax = obj['bbox'][0] + obj['bbox'][2]
            ymax = obj['bbox'][1] + obj['bbox'][3]
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['category_id'])
            
        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
                
        if self.transform:
            
            
            
            image = self.transform(image)
            bboxes = self.transform(np.array(bboxes)).reshape(-1, 4)
            
            targets ={}
            labels = torch.tensor(labels).type(torch.int64)
            bboxes = bboxes.type(torch.FloatTensor)
            targets['boxes'] = bboxes.type(torch.FloatTensor)
            targets['labels'] = labels.type(torch.LongTensor)
            
            return idx, image, targets        
            
        return image, bboxes, labels
    
    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))


# Specific Classes
class MyCocoLimit(CocoDetection):
    def __init__(
        self, root, annFile, transform, class_list, remove_invalid_data=True, show=False
    ):
        super(MyCocoLimit, self).__init__(root, annFile)
        
        self.class_list = class_list
        self.catIds = self.coco.getCatIds(catNms=self.class_list)
        self.new_cls_ids, self.ids = {}, []
        for i, cls_id in enumerate(sorted(self.catIds)):
            self.new_cls_ids[cls_id] = i + 1
            self.ids.extend(self.coco.getImgIds(catIds=cls_id))

        if remove_invalid_data:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anno = self.coco.loadAnns(ann_ids)
                # print(len(anno))
                if self.is_valid_data(anno):
                    ids.append(img_id)
            self.ids = ids
            
        self.transform = transform
        self.show = show
        
    # 물체가 없거나 bbox가 1보다 작은 경우 제외
    def is_valid_data(self, anno):
        if len(anno) == 0:
            return False
        for obj in anno:
            for o in obj['bbox'][2:]:
                if o <= 1:
                    return False
        return True
            
    
    def __getitem__(self, index: int):
        idx = self.ids[index]
        image = self._load_image(idx)
        target = self._load_target(idx)
        
        bboxes, labels = [], []
        for obj in target:
            xmin = obj['bbox'][0]
            ymin = obj['bbox'][1]
            xmax = obj['bbox'][0] + obj['bbox'][2]
            ymax = obj['bbox'][1] + obj['bbox'][3]
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['category_id'])
            
        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        if self.transform:
            augmentations = self.transform(image=np.array(image), bboxes=bboxes, category_id=labels)
            image = augmentations['image']
            bboxes = augmentations['bboxes']
            
        boxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        print(boxes)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
                
        return image, target
    
    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id):
        original = self.coco.loadAnns(self.coco.getAnnIds(id))
        new = [ann for ann in original if ann['category_id'] in self.catIds]
        return new
    
import glob
import cv2

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
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        
        return img, target
        



def coco_show(dataset, shape=(2,2), figsize=(10,10)):
    CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                        std=[1/0.229, 1/0.224, 1/0.225])
    
    size = shape[0] * shape[1]
    nums = sample(range(0, len(dataset)), size)
    
    fig, axes = plt.subplots(*shape, figsize=figsize)
    axes = axes.ravel() if isinstance(axes, np.ndarray) else [axes]
    
    for i, num in enumerate(nums):
        image_id, image, target = dataset.__getitem__(num)
        image = inv_normalize(image)
        image = image.permute(1,2,0)
        boxes = target['boxes']
        labels = target['labels']
        
        for box, label in zip(boxes, labels):
            rect = patches.Rectangle((box[0], box[1]), 
                                     box[2], box[3], 
                                     linewidth=1, edgecolor='r', facecolor='none')
            axes[i].add_patch(rect)
            axes[i].text(box[0], box[1], CLASSES[label], fontsize=15)
            
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(f'Image ID :{image_id}')
        
    plt.tight_layout()
    plt.show()