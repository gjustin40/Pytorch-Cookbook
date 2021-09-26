import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST

import matplotlib.pyplot as plt

from model import VGG
from utils import save_result, make_folder

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='A or B dataset', default='cifar10', type=str)
    parser.add_argument('--model', help='Model if VGG', default='vgg16', type=str)
    parser.add_argument('--in_channels', help='Channels of input Image', default=3, type=int)
    parser.add_argument('--num_classes', help='Number of Classes', default=10, type=int)
    parser.add_argument('--batch_norm', help='Using Batch Normalization', default=False, type=bool)
    parser.add_argument('--train_batch_size', help='Batch size of Training dataset', default=256, type=int)
    parser.add_argument('--test_batch_size', help='Batch size of Testing dataset', default=128, type=int)
    parser.add_argument('--epoch', help='Size of Epoch', default=20, type=int)
    parser.add_argument('--lr', help='Learning Rate', default=0.01, type=float)
    parser.add_argument('--momentum', help='Momentum', default=0.9, type=float)
    parser.add_argument('--cuda', help='Using GPU', default=True, type=bool)
    parser.add_argument('--save_result', help='Save Result of Train&Test', default=True, type=bool)
    parser.add_argument('--save_folder', help='Directory of Saving weight', default='train0', type=str)
    opt = parser.parse_args()
    
    return opt


def train(model, dataloader, optimizer, loss_func, device, e):
    print(f'EPOCH[{e+1}/{opt.epoch}] Training....')
    model.train()
    iter_loss = []
    corrects = 0
    data_size = 0
      
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.cuda(), labels.cuda()
        data_size += images.shape[0]
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        iter_loss.append(loss.item())
        corrects += sum(outputs.argmax(axis=1) == labels).item()

        if ((i+1) % 40 == 0) or ((i+1) == len(dataloader)) :
            print(f'Iter[{i+1}/{len(dataloader)}] --- Loss: {sum(iter_loss)/data_size:0.4} --- Accuracy: {corrects/data_size:0.2}')
    return [sum(iter_loss)/data_size, corrects/data_size]


def test(model, dataloader, loss_func, device, e):
    print(f'EPOCH[{e+1}/{opt.epoch}] Teseting....')
    model.eval()
    iter_loss = []
    corrects = 0
    
    with torch.no_grad():
        data_size = 0
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.cuda(), labels.cuda()
            data_size += images.shape[0]
            
            outputs = model(images)
            loss = loss_func(outputs, labels)
            
            iter_loss.append(loss.item())
            corrects += sum(outputs.argmax(axis=1) == labels).item()
    
    print(f'Iter[{i+1}/{len(dataloader)}] --- Loss: {sum(iter_loss)/data_size:0.4} --- Accuracy: {corrects/data_size:0.2}')
    return [sum(iter_loss)/data_size, corrects/data_size]
            
        
def main(opt):
    # make folder
    base_path = 'result'
    os.makedirs(base_path, exist_ok=True)
    result_path = make_folder(base_path, opt.save_folder)
    print(result_path)
    
    datasets = {
        'mnist': r'C:\Users\gjust\Documents\Github\data',
        'cifar10': r'C:\Users\gjust\Documents\Github\data',
        'cifar100': r'C:\Users\gjust\Documents\Github\data'
    }
    data_path = datasets[opt.dataset]
    
    # Dataset
    print('Preparing Dataset....')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
        
    if opt.dataset == 'mnist':
        train_set = MNIST(root=data_path, transform=transform, train=True, download=True)
        test_set = MNIST(root=data_path, transform=transform, train=False, download=True)
    elif opt.dataset == 'cifar10':
        train_set = CIFAR10(root=data_path, transform=transform, train=True, download=True)
        test_set = CIFAR10(root=data_path, transform=transform, train=False, download=True)
    elif opt.dataset == 'cifar100':
        train_set = CIFAR100(root=data_path, transform=transform, train=True, download=True)
        test_set = CIFAR100(root=data_path, transform=transform, train=False, download=True)
               
    train_loader = DataLoader(train_set, batch_size=opt.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=opt.test_batch_size, shuffle=False)
     
    # GPU
    device = 'cuda' if (torch.cuda.is_available() and opt.cuda) else 'cpu'
        
    # model
    print('Preparing Model....')
    model = VGG(opt.model, opt.in_channels, opt.num_classes, opt.batch_norm)
    model.to(device)
    
    # optmizer
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
    loss_func = nn.CrossEntropyLoss()
    
    # Training
    train_result, test_result = [], []   
    best = 0
    for e in range(opt.epoch):
        train_result += train(model, train_loader, optimizer, loss_func, device, e)
        test_result += test(model, test_loader, loss_func, device, e)

        if test_result[1::2][-1] > best:
            print('Saving Model....')
            torch.save(model, f'{result_path}/{opt.model}.pth')
            best = test_result[1::2][-1]
            
        if opt.save_result:
            save_result(train_result, test_result, result_path)
        
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)