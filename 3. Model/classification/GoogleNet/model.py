import torch
import torch.nn as nn


class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogLeNet, self).__init__()
        
        self.in_channels = in_channels
        
        self.conv1 = conv_block(self.in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception3a = inception_block(64, 96, 128, 16, 32, 32)
        
        

class inception_block(nn.Module):
    def __init__(self, in_channels, out1_1x1, out3_1x1, out_3x3, out5_1x1, out_5x5, outpool_1x1):
        super(inception_block, self).__init__()
        
        self.in_channels = in_channels
        
        self.branch1 = conv_block(in_channels, out1_1x1, kernel_size=1)
        
        self.branch2 = nn.Sequential(
            conv_block(self.in_channels, out3_1x1, kernel_size=1),
            conv_block(out3_1x1, out_3x3, kernel_size=3, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            conv_block(self.in_channels, out5_1x1, kernel_size=1),
            conv_block(out5_1x1, out_5x5, kernel_size=5, padding=2)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.conv_block(self.in_channels, outpool_1x1, kernel_size=1)
        )
        
    def forward(self, x):
        out = torch.cat([self.branch(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
        
        return out
    
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(self.out_channels)
        
    def forward(self, x):
        out = self.relu(self.batchnorm(self.conv(x)))
        
        return out
    

x = torch.randn(1, 3, 224, 224)
conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
# maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
print(conv1(x).shape)
# print(maxpool(x).shape)