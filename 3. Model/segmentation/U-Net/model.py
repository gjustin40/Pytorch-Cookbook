import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np

class UNet(nn.Module):
    
    def __init__(self, input_size=1, num_classes=2):
        super(UNet, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        self.down1 = ConvBlock(self.input_size, 64, nn.MaxPool2d(2))
        self.down2 = ConvBlock(64, 128, nn.MaxPool2d(2))
        self.down3 = ConvBlock(128, 256, nn.MaxPool2d(2))
        self.down4 = ConvBlock(256, 512, nn.MaxPool2d(2))
        
        self.middle = ConvBlock(512, 1024, nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2))
        
        self.up4 = ConvBlock(1024, 512, nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))
        self.up3 = ConvBlock(512, 256, nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2))
        self.up2 = ConvBlock(256, 128, nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2))
        self.up1 = ConvBlock(128, 64)
        
        self.classifier = nn.Conv2d(64, self.num_classes, kernel_size=1)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        
        middle = self.middle(down4)
        up4 = torch.cat((self._crop(down4, middle.shape[2]), middle), dim=1)
        
        up3 = self.up4(up4)
        up3 = torch.cat((self._crop(down3, up3.shape[2]), up3), dim=1)
        
        up2 = self.up3(up3)
        up2 = torch.cat((self._crop(down2, up2.shape[2]), up2), dim=1)
        
        up1 = self.up2(up2)
        up1 = torch.cat((self._crop(down1, up1.shape[2]), up1), dim=1)
        
        up0 = self.up1(up1)
        output = self.classifier(up0)
        
        return output
    
    def _crop(self, x, size):
        return transforms.CenterCrop(size)(x)
        
class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, pool_layer=False):
        super(ConvBlock, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.conv1 = nn.Conv2d(self.input_size, self.output_size, kernel_size=3)
        self.conv2 = nn.Conv2d(self.output_size, self.output_size, kernel_size=3)
        
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(self.output_size)
        self.pool = pool_layer
        
    def forward(self, x):
        
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(self.bn(self.conv2(x)))
        if self.pool:
            x = self.pool(x)
        
        return x
    
    

if __name__ == '__main__':
    model = UNet()
    print(model)
    example = torch.randn(1, 1, 572, 572)
    print(example.shape)

    model = UNet()
    model.cuda()
    output = model(example.cuda())
    print(output.shape)        