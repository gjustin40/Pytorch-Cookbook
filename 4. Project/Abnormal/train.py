import os
import glob
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from dataset import myDataset
from model import AutoEncoder

# Train
def train(opt, model, dataloader, optimizer, loss_func, device, e):
    model.train()
    train_iter_loss = []

    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_iter_loss.append(loss.item())
        
    print(f'Epoch[{e+1}/{opt.epoch}] / Loss : {sum(train_iter_loss)/len(dataloader)}')
    
    return train_iter_loss

# val
def val(opt, model, dataloader, loss_func, device, e):
    model.eval()
    val_iter_loss = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_func(outputs, labels)
            val_iter_loss.append(loss.item())
    
        print(f'Validation Epoch[{e+1}/{opt.epoch}] / Loss : {sum(val_iter_loss)/len(dataloader)}')
    
    return val_iter_loss
    
# HyperParameters
## LR, EPOCH, Batchsize, 
def parse_opt():
    parser = argparse.ArgumentParser(description='Train AutoEncoder Network')
    parser.add_argument('--train_path', help='training dataset', type=str)
    parser.add_argument('--val_path', help='validation dataset', type=str)

    parser.add_argument('--epoch', help='epoch', default=20, type=int)
    parser.add_argument('--train_batch_size', help='train batch_size', default=32, type=int)
    parser.add_argument('--val_batch_size', help='validation batch_size', default=32, type=int)
    parser.add_argument('--lr', help='learning rate', default=0.001, type=float)
    
    opt = parser.parse_args()
    
    return opt

def main(opt):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dataset
    print('Dataset....')
    transform = transforms.Compose([
        transforms.Resize((600,600)),
        transforms.Grayscale(3),
        transforms.ToTensor()
    ])
    
    train_set = myDataset(image_path=opt.train_path, transform=transform)
    val_set = myDataset(image_path=opt.val_path, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=opt.train_batch_size)
    val_loader = DataLoader(val_set, batch_size=opt.val_batch_size)
    
    # Model
    print('Model....')
    model = AutoEncoder()
    model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    loss_func = nn.MSELoss()
        
    # Train
    print('Training....')
    
    train_epoch_loss = []
    val_epoch_loss = []
    train_iter_losses = []
    val_iter_losses = []
    for e in range(opt.epoch):
        train_iter_loss = train(opt, model, train_loader, optimizer, loss_func, device, e)
        train_iter_losses += train_iter_loss
        train_epoch_loss.append(sum(train_iter_loss))
        
        val_iter_loss = val(opt, model, val_loader, loss_func, device, e)
        val_iter_losses += val_iter_loss
        val_epoch_loss.append(sum(val_iter_loss))
            
        # save model
        best = 10000
        if val_epoch_loss[-1] < best:
            print('Saving Model....')
            torch.save(model, 'weights/AutoEncoder_try1.pth')
            best = val_epoch_loss[-1]
    
    print('Saving Result')
    plt.figure(figsize=(10,10))
    plt.plot(train_iter_losses)
    plt.plot(val_iter_losses)
    plt.legend(['Train_loss', 'Val_loss'])
    
    plt.savefig('Result.jpg')
    
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)