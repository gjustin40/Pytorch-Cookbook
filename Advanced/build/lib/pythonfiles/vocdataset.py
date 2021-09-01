import torch
from torchvision.datasets import VOCDetection

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
    
    
classes = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

class myVOCDetection(VOCDetection):
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
        
        targets, labels = [], []
        for t in target['annotation']['object']:
            bbox = [int(t['bndbox']['xmin']), 
                    int(t['bndbox']['ymin']),
                    int(t['bndbox']['xmax']),
                    int(t['bndbox']['ymax'])]
            label = classes.index(t['name'])
            
            targets.append(bbox)
            labels.append(label)
            
        labels = torch.tensor(labels)
        if self.transform:
            img = self.transform(img)
            targets = self.transform(np.array(targets))
        
        return img, targets, labels
    

def pascal_show(dataloader, figsize):
    CLASSES = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    
    images, bboxes, labels = iter(dataloader).next()
    
    fig, axes = plt.subplots(*figsize, figsize=(15,10))
    axes = axes.ravel()
    for i in range(len(images)):
        img = images[i].permute(1,2,0)
        
        for box, label in zip(bboxes[i][0], labels[i]):
            rect = patches.Rectangle((box[0], box[1]), 
                                     box[2]-box[0], 
                                     box[3]-box[1], 
                                     linewidth=3, edgecolor='r', facecolor='none')
            axes[i].add_patch(rect)
            axes[i].text(box[0], box[1], CLASSES[label], fontsize=15)
            
        axes[i].imshow(img)
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show() 