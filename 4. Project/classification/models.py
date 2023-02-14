import torch.nn as nn
import torchvision.models as m
from efficientnet_pytorch import EfficientNet

def get_model(name, num_classes, pretrained=False):
    
    if name == 'vgg11':  
        model = m.vgg11_bn(pretrained=pretrained)
    
    elif name == 'vgg13': 
        model = m.vgg13_bn(pretrained=pretrained)
    
    elif name == 'vgg16':        
        model = m.vgg16_bn(pretrained=pretrained)
    
        for p in model.parameters():
            p.requires_grad = False
        model.classifier[-1] = nn.Linear(4096, num_classes)

    elif name == 'vgg19': 
        model = m.vgg19_bn(pretrained=pretrained)
        

    
    elif name == 'resnet50':
        model = m.resnet50(pretrained=pretrained)
        
#         for p in model.parameters():
#             p.requires_grad = False
        model.fc = nn.Linear(2048, 2)
        
    elif name == 'resnet101':
        model = m.resnet101(pretrained=pretrained)
        


    elif name == 'squeezenet':
        model = m.squeezenet1_0(pretrained=pretrained)
        
        for p in model.parameters():
            p.requires_grad = False
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1, stride=1)

    elif name == 'efficientnet-b0':
        model = EfficientNet.from_pretrained('efficientnet-b0')
        for p in model.parameters():
            p.requires_grad = False
        
        model._fc = nn.Linear(in_features=1280, out_features=num_classes)
    
    elif name == 'efficientnet-b7':
        model = EfficientNet.from_pretrained('efficientnet-b7')
        for p in model.parameters():
            p.requires_grad = False    
    
        model._fc = nn.Linear(in_features=2560, out_features=num_classes)
    
    return model



if __name__ == '__main__':
    model = get_model('resnet50', 2, pretrained=True)
    for p in model.parameters():
        print(p.requires_grad)
    print(model)