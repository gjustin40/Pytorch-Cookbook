import os
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder

def get_dataset(name, train_transform=None, test_transform=None):
    data_paths = {
        'mnist': r'C:\Users\gjust\Documents\Github\data',
        'cifar10': r'C:\Users\gjust\Documents\Github\data',
        'cifar100': r'C:\Users\gjust\Documents\Github\data',
        'coco': r'C:\Users\gjust\Documents\Github\data\COCO',
        'catdog': r'C:\Users\gjust\Documents\Github\data\dogs-vs-cats'
    }
    
    data_path = data_paths[name]
    
    if name == 'mnist':
        train_set = MNIST(root=data_path, transform=train_transform, train=True, download=True)
        test_set = MNIST(root=data_path, transform=test_transform, train=False, download=True)
        
    elif name == 'cifar10':
        train_set = CIFAR10(root=data_path, transform=train_transform, train=True, download=True)
        test_set = CIFAR10(root=data_path, transform=test_transform, train=False, download=True)
        
    elif name == 'cifar100':
        train_set = CIFAR100(root=data_path, transform=train_transform, train=True, download=True)
        test_set = CIFAR100(root=data_path, transform=test_transform, train=False, download=True)
    
    elif name == 'coco':
        base_path = data_paths[name]
        train_path = os.path.join(base_path, 'train2017')
        train_ann = os.path.join(base_path, 'instances_train2017.json')
        train_set = CIFAR100(root=data_path, transform=train_transform, train=True, download=True)
        test_set = CIFAR100(root=data_path, transform=test_transform, train=False, download=True)
        
    elif name == 'catdog':
        train_path = os.path.join(data_path, 'train')
        # test_path = os.path.join(data_path, 'test')
        train_set = ImageFolder(root=train_path, transform=train_transform)
        test_set = ImageFolder(root=train_path, transform=test_transform)
        
    return train_set, test_set

if __name__ == '__main__':
    data_path = get_dataset('catdog')
    print(data_path)