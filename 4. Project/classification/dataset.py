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


import os
import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset

import torchvision
import torchvision.models as m
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split

class FloodDataset(Dataset):
    def __init__(self, dataframe, class_to_idx, transform=None):
        super(FloodDataset).__init__()

        self.dataframe = dataframe
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        data = self.dataframe.iloc[idx]

        image_path = data['fname']
        class_name = data['class']
        label = self.class_to_idx[class_name]

        img = Image.open(image_path)
        label = torch.tensor(label).type(torch.int64)

        if self.transform:
            img = self.transform(img)

        return img, label

def get_dataset2(train_size, train_transform=None, test_transform=None):
    data_path = os.path.join(os.getenv('HOME'), 'sakong/flooding/data')
    df = pd.read_csv(data_path + '/labels2.csv')

    flood_df = df.loc[df['class'] == 'flood']
    notflood_df = df.loc[df['class'] == 'not_flood']

    flood_train = flood_df.sample(frac=train_size, random_state=10)
    flood_test = flood_df.drop(flood_train.index).sample(frac = 0.5, random_state=10)
    flood_val = flood_df.drop(flood_train.index).drop(flood_test.index)

    notflood_train = notflood_df.sample(frac=train_size, random_state=10)
    notflood_test = notflood_df.drop(notflood_train.index).sample(frac = 0.5, random_state=10)
    notflood_val = notflood_df.drop(notflood_train.index).drop(notflood_test.index)

    train_df = pd.concat([flood_train, notflood_train])
    test_df = pd.concat([flood_test, notflood_test])
    val_df = pd.concat([flood_val, notflood_val])

    class_to_idx = {'flood': 1, 'not_flood': 0}
    train_set = FloodDataset(train_df, class_to_idx, train_transform)
    val_set = FloodDataset(val_df, class_to_idx, test_transform)
    test_set = FloodDataset(test_df, class_to_idx, test_transform)
    
    return train_set, val_set, test_set
    
if __name__ == '__main__':
    train_set, val_set, test_set = get_dataset2(0.8)
    test_loader = DataLoader(test_set)
    print(test_loader.dataset.samples)


if __name__ == '__main__':
    data_path = get_dataset('catdog')
    print(data_path)
    