import torch
import torch.nn as nn


class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, conv_block=None):
        super(GoogLeNet, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        if conv_block is None:
            self.conv_block = BasicConv2d
            
        self.conv1 = self.conv_block(self.in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self.conv_block(64, 192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout2d(0.4)
        self.fc = nn.Linear(1024, self.num_classes)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)

        x = self.pool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)

        x = self.pool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

    
class inception_block(nn.Module):
    def __init__(self, in_channels, out1_1x1, out3_1x1, out_3x3, out5_1x1, out_5x5, outpool_1x1, conv_block=None):
        super(inception_block, self).__init__()
        
        self.in_channels = in_channels
        if conv_block is None:
            self.conv_block = BasicConv2d
            
        self.branch1 = self.conv_block(self.in_channels, out1_1x1, kernel_size=1)
        
        self.branch2 = nn.Sequential(
            self.conv_block(self.in_channels, out3_1x1, kernel_size=1),
            self.conv_block(out3_1x1, out_3x3, kernel_size=3, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            self.conv_block(self.in_channels, out5_1x1, kernel_size=1),
            self.conv_block(out5_1x1, out_5x5, kernel_size=5, padding=2)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            self.conv_block(self.in_channels, outpool_1x1, kernel_size=1)
        )
        
    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
        
        return out
    
    
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(self.out_channels)
        
    def forward(self, x):
        out = self.relu(self.batchnorm(self.conv(x)))
        
        return out
    

if __name__ == '__main__':
    from torchvision.models import googlenet
    
    x = torch.randn(8, 3, 843, 134)
    google = googlenet()
    model = GoogLeNet()
    print(google(x).shape)
    print(model(x).shape)
      