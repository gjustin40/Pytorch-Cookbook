import argparse
import matplotlib.pyplot as plt
from PIL import Image
import torch
from model import AutoEncoder
import numpy as np
from torchvision.transforms import ToTensor

import warnings
warnings.filterwarnings("ignore")

def parse_opt():
    parser = argparse.ArgumentParser(description='Decode Image')
    parser.add_argument('--image_path', help='Image path', default='./example.jpg', type=str)
    
    opt = parser.parse_args()
    
    return opt

def main(opt):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Model
    model = torch.load('weights/AutoEncoder_try1.pth')
    model.to(device)
    model.eval()
    
    # Image
    image = Image.open(opt.image_path)
    image = ToTensor()(image).unsqueeze(0)

    with torch.no_grad():
        model.eval()
        output = model(image.to(device))
       
    image_np = np.array(output.cpu().squeeze(0).permute(1,2,0))

    image_np = Image.open('example.jpg')
    plt.imshow(image_np)
    
    plt.show()
    
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
