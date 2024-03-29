import os
import argparse
import time
from time import strftime, gmtime
import datetime
import random

import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import MyCocoLimit
from metric import get_bbox, mean_average_precision

random_seed = 40
torch.set_printoptions(precision=5, sci_mode=False)
torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='A or B dataset', default='cifar10', type=str)
    parser.add_argument('--model', help='Model Name', default='resnet18', type=str)
    parser.add_argument('--in_channels', help='Channels of input Image', default=3, type=int)
    parser.add_argument('--num_classes', help='Number of Classes', default=10, type=int)
    parser.add_argument('--batch_norm', help='Using Batch Normalization', default=False, type=bool)
    parser.add_argument('--train_batch_size', help='Batch size of Training dataset', default=256, type=int)
    parser.add_argument('--test_batch_size', help='Batch size of Testing dataset', default=128, type=int)
    parser.add_argument('--epoch', help='Size of Epoch', default=20, type=int)
    parser.add_argument('--lr', help='Learning Rate', default=0.01, type=float)
    parser.add_argument('--momentum', help='Momentum', default=0.9, type=float)
    parser.add_argument('--cuda', help='Using GPU', default=True, type=bool)
    parser.add_argument('--resume', help='Start from checkpoint', default='', type=str)
    parser.add_argument('--save_result', help='Save Result of Train&Test', default=True, type=bool)
    parser.add_argument('--save_folder', help='Directory of Saving weight', default='train0', type=str)
    opt = parser.parse_args()
    
    return opt

def coco2pascal(bboxes):
    new_bboxes = torch.zeros_like(bboxes)
    new_bboxes[:, :2] = bboxes[:, :2]
    new_bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]
    
    return new_bboxes
    
def train(model, dataloader, optimizer, device, EPOCH, e):
    model.train()
    iter_loss = []
    data_size = 0
    times = 0
        
    for i, (idx, images, targets) in enumerate(dataloader):
        start = time.time()
        
        images = [image.to(device) for image in images]
        new_targets = []
        for target in targets:
            target['boxes'] = coco2pascal(target['boxes'])
            new_targets.append({k:v.to(device) for k,v in target.items()})
        
        data_size += len(images)
        
        optimizer.zero_grad()    
        loss_dict = model(images, new_targets)
        loss = sum(v for v in loss_dict.values())
        loss.backward()
        optimizer.step()
        
        iter_loss.append(loss.item())
        
        end = time.time()
        if ((i+1) % 200 == 0) or ((i+1) == len(dataloader)):
            times = (end-start)*200 if not (i+1) == len(dataloader) else (end-start)*(i+1)
            
            print(f'EPOCH: [{e}/{EPOCH}]' \
                  f' --- Iter: [{i+1}/{len(dataloader)}]'\
                  f' --- Loss: {sum(iter_loss)/data_size:0.4f}'\
                  f' --- Time: {strftime("%H:%M:%S", gmtime(times))}'\
                  f' --- LR: ')

@torch.no_grad()
def test(model, dataloader, device, EPOCH, e):
    start = time.time()
    model.eval()
    
    total_gt_bboxes = torch.tensor([])
    total_pred_bboxes = torch.tensor([])
    for i, (img_ids, images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for img_id, target, output in zip(img_ids, targets, outputs):
            gt_bboxes = get_bbox(img_id, target)
            pred_bboxes = get_bbox(img_id, output, pred=True)

            total_gt_bboxes = torch.cat([total_gt_bboxes, gt_bboxes])
            total_pred_bboxes = torch.cat([total_pred_bboxes, pred_bboxes])
            
    end = time.time()
    print('Shape of total ground Truths bboxes :', total_gt_bboxes.shape)
    print('Shape of total predicted bboxes :', total_pred_bboxes.shape)
    print(strftime("%H:%M:%S", gmtime(end-start)))
    
    start = time.time()
    mAP, AP_per_classes = mean_average_precision(total_pred_bboxes,
                                             total_gt_bboxes,
                                             iou_threshold=0.5,
                                             box_format='coco',
                                             num_classes=4)
    
    end = time.time()
    print('mAP :', mAP)
    print(AP_per_classes)
    print(strftime("%H:%M:%S", gmtime(end-start)))
    
    
def collate_fn(batch):
    return tuple(zip(*batch))
    
def main():
    # make folder
        
    # Dataset
    print(f'Preparing Dataset....COCO Dataset')
    train_path = r'C:\Users\gjust\Documents\Github\data\COCO\train2017'
    train_ann = r'C:\Users\gjust\Documents\Github\data\COCO\annotations\instances_train2017.json'
    test_path = r'C:\Users\gjust\Documents\Github\data\COCO\val2017'
    test_ann = r'C:\Users\gjust\Documents\Github\data\COCO\annotations\instances_val2017.json'
    class_list = ['person']
    
    transform = A.Compose([
        A.Resize(width=512, height=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    
    trainset = MyCocoLimit(root=train_path, annFile=train_ann, class_list=class_list, transform=transform)
    testset = MyCocoLimit(root=test_path, annFile=test_ann, class_list=class_list, transform=transform)
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
    
    # resuming
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Training
    print('Training....')
    EPOCH = 50
    for e in range(EPOCH):
        train(model, train_loader, optimizer, device, EPOCH, e)
        # test(model, test_loader, device, EPOCH, e)
    
if __name__ == '__main__':
    opt = parse_opt()
    main()