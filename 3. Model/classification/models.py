from ResNet.model import ResNet, BasicBlock, Bottleneck

models = {
    'resnet18': ResNet(BasicBlock, [2, 2, 2, 2]),
    'resnet34': ResNet(BasicBlock, [3, 4, 6, 3]),
    'resnet50': ResNet(Bottleneck, [3, 4, 6, 3]),
    'resnet101': ResNet(Bottleneck, [3, 4, 23, 3]),
    'resnet152': ResNet(Bottleneck, [3, 8, 36, 3])
}

def get_model(name='resnet18'):
    return models[name]

