import torch
import torch.nn as nn

configs = {
    'vgg11' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    
    def __init__(self, config, in_channels=3, num_classes=10, batch_norm=True):
        super(VGG, self).__init__()
        self.config = config
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.batch_norm = batch_norm
        
        self.features = self.make_features()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = self.make_classifier()
        # self._weight_init()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x
        
    def make_features(self):
        in_channels = self.in_channels
        config = self.config
        batch_norm = self.batch_norm
        layers = []
        for out_channels in configs[config]:
            if type(out_channels) == int:
                conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)] 
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                    
                in_channels = out_channels
            else:
                layers += [nn.MaxPool2d(2)]
                
        return nn.Sequential(*layers)
    
    def make_classifier(self):
        num_classes = self.num_classes
        
        layers = []
        layers += [nn.Linear(512*1*1, 4096), nn.ReLU(inplace=True), nn.Dropout(p=0.5)]
        layers += [nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(p=0.5)]
        layers += [nn.Linear(4096, num_classes)]
        
        return nn.Sequential(*layers)

    def _weight_init(self):
        for m in self.modules():
            if (type(m) == nn.Conv2d) or (type(m) == nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                

if __name__ == '__main__':
    from torchvision.models import vgg16_bn
    x = torch.randn(10, 3, 32, 32)
    model16 = vgg16_bn()
    model = VGG('vgg16')
    
    # print(model(x).shape)
    for name, p in model.named_parameters():
        print(name)
    # print(model)