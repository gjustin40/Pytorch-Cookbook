import os

import torch
import torch.nn as nn
import torchvision

import numpy as np

class UNet(nn.Module):
    
    def __init__(self, input_size=1):
        self.input_size = input_size
        self.down1 = Downsampling_Block(self.input_size, 64)
        self.donw2 = Downsampling_Block(64, 128)
        self.donw3 = Downsampling_Block(128, 256)
        self.donw4 = Downsampling_Block(256, 512)



class Downsampling_Block(nn.Module):
    def __init__(self, input_size, output_size):
        super(Downsampling_Block, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.conv1 = nn.Conv2d(self.input_size, self.output_size, kernel_size=3)
        self.conv2 = nn.Conv2d(self.output_size, self.output_size, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(self.output_size)
    
    def forward(self, x):
        
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(self.bn(self.conv2(x)))
        x = self.pool(x)
        
        return x
    


example = torch.randn(1, 3, 572, 572)
print(example.shape)

model = Downsampling_Block(input_size=3, output_size=64)
model.cuda()

output = model(example.cuda())
print(output.shape)        