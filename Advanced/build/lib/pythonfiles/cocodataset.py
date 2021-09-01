import os

import torch
from torchvision.datasets import CocoDetection

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
   
class myCocoDetection(CocoDetection):
    def __init__(
        self, root, annFile, transform, remove_invalid_data=True):
        super(myCocoDetection, self).__init__(root, annFile)
        
        if remove_invalid_data:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anno = self.coco.loadAnns(ann_ids)
                if self.is_valid_data(anno):
                    ids.append(img_id)
            self.ids = ids
            
        self.transform = transform
        
    def is_valid_data(self, anno):
        if len(anno) == 0:
            return False
        for obj in anno:
            for o in obj['bbox'][2:]:
                if o <= 1:
                    return False
        return True
    
    def _load_image(self, idx: int):
        path = self.coco.loadImgs(idx)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")
    
    def _load_target(self, idx):
        return self.coco.loadAnns(self.coco.getAnnIds(idx))
    
    def __getitem__(self, index: int):
        idx = self.ids[index]
        image = self._load_image(idx)
        target = self._load_target(idx)
        
        bboxes, labels = [], []
        for obj in target:
            bbox = [obj['bbox'][0],
                    obj['bbox'][1],
                    obj['bbox'][0] + obj['bbox'][2],
                    obj['bbox'][1] + obj['bbox'][3]]
            bboxes.append(bbox)
            labels.append(obj['category_id'])
        
        if self.transform:
            image = self.transform(image)
            bboxes = self.transform(np.array(bboxes)).reshape(-1, 4)
            
            targets ={}
            labels = torch.tensor(labels).type(torch.int64)
            bboxes = bboxes.type(torch.FloatTensor)
            targets['boxes'] = bboxes
            targets['labels'] = labels
            
            return image, targets        
            
        return image, bboxes, labels
    
    
def coco_show(dataloader, figsize):
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
    
    images, targets = iter(dataloader).next()
    
    fig, axes = plt.subplots(*figsize, figsize=(15,10))
    axes = axes.ravel()
    for i in range(len(images)):
        img = images[i].permute(1,2,0)
        
        for box, label in targets[i]:
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none') 
            axes[i].add_patch(rect)
            axes[i].text(box[0], box[1], CLASSES[label], fontsize=15)
            
        axes[i].imshow(img)
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show() 
    
    
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

image_path = '../../data/COCO/val2017'
ann_path = '../../data/COCO/annotations/instances_val2017.json'
def collate_fn(batch):
    return tuple(zip(*batch))

dataset = myCocoDetection(root=image_path, annFile=ann_path, transform=transforms.ToTensor())
dataset_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
print(iter(dataset_loader).next())



# import time
# device = 'cuda'

# for e in range(10):
    
#     target, targets = {}, []
#     for i, (images, targets) in enumerate(dataset_loader):
#         start = time.time()
#         images = [img.to(device) for img in images]
#         targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
#         print(targets)