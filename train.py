
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from vgg import myVGG

parser = argparse.ArgumentParser(description='PyTorch myVGG Training')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--mm', '--momentum', default=0.9, type=float, help='momentum rate')
parser.add_argument('--batch', '--batch-size', default=256, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
parser.add_argument('--name', default='prography.pth', type=str, help='The name of saved model')

args = parser.parse_args()

def train(trainloader, model, loss_func, optimizer, epoch, device):
    
    model.train()  
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_func(output, labels)
        loss.backward()
        optimizer.step()
        
        if (i % 20 == 0) or (i == 50000 // args.batch):
            print('Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss : {:.6f}'.format(
                epoch, i * len(images), len(trainloader.dataset),
                100 * i / len(trainloader), loss.item()))

def val(valloader, model, loss_func, device):  
    model.eval()
    val_loss = 0
    val_correct = 0
    
    with torch.no_grad():
        for (images, labels) in valloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            val_loss += loss_func(output, labels)
            pred = output.argmax(dim=1, keepdim=True)
            val_correct += pred.eq(labels.view_as(pred)).sum().item()
        
        
    val_loss = val_loss / len(valloader.dataset)
    
    val_accuracy = (100 * val_correct) / len(valloader.dataset)
    print('\nVal set : Average loss: {:.4f}, Val_Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, val_correct, len(valloader.dataset), val_accuracy))
          
    return val_correct
                 
def main():
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU로 학습합니다.')
    else:
        device = torch.device('cpu')
        print('CPU로 학습합니다.')
        
        
    transform = transforms.Compose(
        [
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ])

    train_val_set = torchvision.datasets.MNIST(root = './data', train = True,
                                          download = True, transform=transform)
    
    trainset, valset = torch.utils.data.random_split(train_val_set, [50000,10000])
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = args.batch,
                                              shuffle = True, num_workers=1)
    valloader = torch.utils.data.DataLoader(valset, batch_size = args.batch,
                                              shuffle = False, num_workers=1)
    
    model = myVGG().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mm)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    loss_func = nn.CrossEntropyLoss()
    
    epoch = args.epochs
    best_accuracy = 0
    for epoch in range(1, epoch+1):
        train(trainloader, model, loss_func, optimizer, epoch, device)
        accuracy = val(valloader, model, loss_func, device)
        scheduler.step()            
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), '{}'.format(args.name)) # overwrite 방지를 위해 이름을 바꾸겠습니다.
                          
if __name__ == '__main__':
    main()
