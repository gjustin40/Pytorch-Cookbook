from VGG.model import VGG
from ResNet.model import ResNet, BasicBlock, Bottleneck
from GoogLeNet.model import GoogLeNet


def get_model(name='resnet18', num_classes=10):
    models = {
        
        'vgg11': VGG('vgg11', num_classes=num_classes),
        'vgg13': VGG('vgg13', num_classes=num_classes),
        'vgg16': VGG('vgg16', num_classes=num_classes),
        'vgg19': VGG('vgg19', num_classes=num_classes),
        
        'resnet18': ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes),
        'resnet34': ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes),
        'resnet50': ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes),
        'resnet101': ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes),
        'resnet152': ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes),
        
        'googlenet': GoogLeNet(num_classes=num_classes)
    }
    
    return models[name]