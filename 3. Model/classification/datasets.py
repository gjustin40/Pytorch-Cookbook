import os
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

def get_dataset_path(name='cifar10', transform):
    data_paths = {
        'mnist': r'C:\Users\gjust\Documents\Github\data',
        'cifar10': r'C:\Users\gjust\Documents\Github\data',
        'cifar100': r'C:\Users\gjust\Documents\Github\data',
        'coco': r'C:\Users\gjust\Documents\Github\data\COCO'
    }
    
    if name == 'mnist':
        train_set = MNIST(root=data_path, transform=transform, train=True, download=True)
        test_set = MNIST(root=data_path, transform=transform, train=False, download=True)
        
    elif name == 'cifar10':
        train_set = CIFAR10(root=data_path, transform=transform, train=True, download=True)
        test_set = CIFAR10(root=data_path, transform=transform, train=False, download=True)
        
    elif name == 'cifar100':
        train_set = CIFAR100(root=data_path, transform=transform, train=True, download=True)
        test_set = CIFAR100(root=data_path, transform=transform, train=False, download=True)
    
    elif name == 'coco':
        base_path = data_paths[name]
        train_path = os.path.join(base_path, 'train2017')
        train_ann = os.path.join(base_path, 'instances_train2017.json')
        train_set = CIFAR100(root=data_path, transform=transform, train=True, download=True)
        test_set = CIFAR100(root=data_path, transform=transform, train=False, download=True)
