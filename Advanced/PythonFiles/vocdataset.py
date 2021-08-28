import torch
from torchvision.datasets import VOCDetection
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