
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from vgg import myVGG

def test(testloader, model, loss_func, device):  
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for (images, labels) in testloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            test_loss += loss_func(output, labels)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
        
        
    test_loss = test_loss / len(testloader.dataset)
    
    accuracy = (100 * correct) / len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset), accuracy))
    print('이 모델의 정확도는 약 98%입니다. 감사합니다.')
       
          
def main():
    print('안녕하세요! 사공용협입니다. 사전과제 제출하겠습니다.')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU 사용 중.......')
    else:
        device = torch.device('cpu')
        print('CPU 사용 중')
        
        
    transform = transforms.Compose(
        [
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ])    
    testset = torchvision.datasets.MNIST(root = './data', train = False,
                                         download = True, transform=transform)    
    testloader = torch.utils.data.DataLoader(testset, batch_size = 32,
                                             shuffle = True, num_workers=1)    

    
    model = myVGG().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    
    loss_func = nn.CrossEntropyLoss()
    test(testloader, model, loss_func, device)
                         
if __name__ == '__main__':
    main()
