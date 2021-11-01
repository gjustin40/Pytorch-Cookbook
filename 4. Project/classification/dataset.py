import os
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder

def get_dataset(name, train_transform=None, test_transform=None):
    data_paths = {
        'catdog': r'C:\Users\gjust\Documents\Github\data\dogs-vs-cats'
    }
    
    data_path = data_paths[name]
        
    if name == 'catdog':
        train_path = os.path.join(data_path, 'train')
        # test_path = os.path.join(data_path, 'test')
        train_set = ImageFolder(root=train_path, transform=train_transform)
        test_set = ImageFolder(root=train_path, transform=test_transform)
        
    return train_set, test_set

if __name__ == '__main__':
    data_path = get_dataset('catdog')
    print(data_path)