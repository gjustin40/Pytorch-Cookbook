import os
import time
from time import strftime, gmtime
import datetime
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms.transforms import RandomHorizontalFlip
import torchmetrics

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# from model import ResNet, Bottleneck, BasicBlock
from models import get_model
from dataset import get_dataset, get_dataset
from utils import save_result, make_folder
from train_utils import train, val, test

import warnings
warnings.filterwarnings('ignore')

random_seed = 10
torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)


################################## Config ##################################################
batch_sizes = [8, 16, 32]
image_sizes = [64, 128, 256, 512]
learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001]
train_set_ratios = [0.9, 0.8, 0.7, 0.6]
# optimizer_names = ['SGD', 'Adam']
optimizer_name = 'SGD'
# model_names = ['vgg16', 'resnet50']
model_name = 'vgg16'

config = {
    'cuda': 'cuda',
    'testing': False,
    'pretrained': True,
    'val_batch_size': 16,
    'test_batch_size': 16,
    'num_classes': 2,
    'step_size': 15,
    'gamma': 0.5,
    'epoch': 50
}


################################## GPU ##################################################
if torch.cuda.is_available() and config['cuda']:
    device = config['cuda']
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'
print(f'Using {device}')

################################## Training ##################################################
train_acc_list, train_loss_list = [], []
val_acc_list, val_loss_list = [], []
test_acc_list, test_loss_list = [], []
best_acc = 0



for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        
        writer = SummaryWriter(f'runs/catdog/Batchsize {batch_size} Lr {learning_rate}')
        
        ################################## Dataset ##################################################
        train_transform = transforms.Compose([
        transforms.Resize((32,32)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        

        ################################## model ##################################################
        print(f'Preparing Model....{model_name}') # model_name
        model = get_model(model_name, config['num_classes'], config['pretrained']) # model_name


        ################################## Optimizer ##################################################
        if optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005) # Learning rate, Optimizer
        elif optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005) # Learning rate, Optimizer
            
        loss_func = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])

        # train_set, val_set, test_set = get_dataset2(train_set_ratio, train_transform, test_transform)  # train_set_ratio
        train_set, test_set = get_dataset('catdog', train_transform, test_transform)
        print(train_set)
        if config['testing']:
            # train_set = Subset(train_set, range(batch_size)) # batch_size
            # # val_set = Subset(test_set, range(config['val_batch_size']))
            # test_set = Subset(test_set, range(config['test_batch_size']))
            train_set = Subset(train_set, range(len(train_set)/5)) # batch_size
            test_set = Subset(test_set, range(len(test_set)/5))
            
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) # batch_size
        # val_loader = DataLoader(val_set, batch_size=config['val_batch_size'], shuffle=False, num_workers=8)
        test_loader = DataLoader(test_set, batch_size=config['val_batch_size'], shuffle=False)
        print(len(train_loader))
        
        for e in range(config['epoch']):        
            train_acc, train_loss = train(model, train_loader, optimizer, loss_func, device, scheduler, writer, config['epoch'], e)
            # val_acc, val_loss = val(model, val_loader, loss_func, device, writer, e)
            test_acc, test_loss = test(model, test_loader, loss_func, device, writer, e)    
            scheduler.step()
    
    
       
batch_sizes = [8, 16, 32]
image_sizes = [64, 128, 256, 512]
learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001]
train_set_ratios = [0.9, 0.8, 0.7, 0.6]
optimizer_names = ['SGD', 'Adam']
model_names = ['vgg16', 'resnet50']   
    
    
# batch_sizes = [8, 16, 32]
# image_sizes = [64, 128, 256, 512]
# learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001]
# train_set_ratios = [0.9, 0.8, 0.7, 0.6]
# optimizer_names = ['SGD', 'Adam']
# model_names = ['vgg16', 'resnet50']