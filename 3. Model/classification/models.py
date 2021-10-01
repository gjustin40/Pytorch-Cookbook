from ResNet.model import ResNet, BasicBlock, Bottleneck


def get_model(name='resnet18', num_classes=10):
    models = {
        'resnet18': ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes),
        'resnet34': ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes),
        'resnet50': ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes),
        'resnet101': ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes),
        'resnet152': ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
    }
    
    return models[name]
