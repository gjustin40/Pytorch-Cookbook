# py파일로 저장 후 실행

import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import warnings
warnings.filterwarnings('ignore')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 10, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(10 * 7 * 7, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x)) # 28 28
        x = self.pool(x)          # 14 14
        x = F.relu(self.conv2(x)) # 14 14
        x = self.pool(x)          # 7 7
        x = x.view(-1, 10 * 7 * 7)
        x = F.relu(self.fc1(x))
        
        return x

def main_worker(rank, world_size):
    # Init DDP
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    # DataLoader
    transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0], [1])
    ])

    data_path = os.path.join(os.getenv('HOME'), 'data')
    train_batch = 2048
    test_batch = 128
    trainset = torchvision.datasets.MNIST(root = data_path, train = True,
                                          download = True, transform=transform)
    train_sampler = DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = train_batch,
                                              num_workers=4, pin_memory=True, sampler=train_sampler)

    testset = torchvision.datasets.MNIST(root = data_path, train = False,
                                         download = True, transform=transform)
    test_sampler = DistributedSampler(testset)
    testloader = torch.utils.data.DataLoader(testset, batch_size = test_batch,
                                             num_workers=4, pin_memory=True, sampler=test_sampler)    
    # Model
    model = Net()
    model_ddp = DDP(model.to(rank), device_ids=[rank])
    
    # Optimizer
    loss_func = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(model_ddp.parameters(), lr = 0.1)
    
    # train
    EPOCH = 2
    for e in range(1, EPOCH+1):
        train_sampler.set_epoch(e)
        model_ddp.train()
        
        start_time = time.time()
        running_loss = 0
        for i, data in enumerate(trainloader):
            images, labels = data
            images, labels = images.to(rank), labels.to(rank)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss
            now = time.time()
            print('\r[%d/%d]-----[%d/%d] LOSS : %.3f------ Time : %d' 
                  %(e, EPOCH, i, 60000/512, running_loss, now - start_time), end = '')
        print('\n')

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker,
            args=(world_size,),
            nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
