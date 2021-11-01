import torch.nn as nn
import torchvision.models as m

def get_model(name, num_classes, pretrained=False):
    
    if name == 'vgg11':  
        model = m.vgg11_bn(pretrained=pretrained)
    
    elif name == 'vgg13': 
        model = m.vgg13_bn(pretrained=pretrained)
    
    elif name == 'vgg16':        
        model = m.vgg16_bn(pretrained=pretrained)

    elif name == 'vgg19': 
        model = m.vgg19_bn(pretrained=pretrained)
        
    for p in model.parameters():
        p.requires_grad = False
    model.classifier[-1] = nn.Linear(4096, num_classes)
    
    return model

if __name__ == '__main__':
    model = get_model('vgg11', 2, pretrained=True)
    print(model)
        