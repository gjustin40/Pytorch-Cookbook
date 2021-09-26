import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder1 = nn.Conv2d(3, 16, 3, padding=1)
        self.encoder2 = nn.Conv2d(16, 8, 3, padding=1)
        self.encoder3 = nn.Conv2d(8, 4, 3, padding=1)
        self.encoder4 = nn.Conv2d(4, 4, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.up1 = nn.Upsample(75, mode='nearest')
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.decoder1 = nn.Conv2d(4, 4, 3, padding=1)
        self.decoder2 = nn.Conv2d(4, 4, 3, padding=1)
        self.decoder3 = nn.Conv2d(4, 8, 3, padding=1)
        self.decoder4 = nn.Conv2d(8, 16, 3, padding=1)
        self.decoder5 = nn.Conv2d(16, 3, 3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.encoder1(x))
        x = self.pool(x)
        x = F.relu(self.encoder2(x))
        x = self.pool(x)
        x = F.relu(self.encoder3(x))
        x = self.pool(x)
        x = F.relu(self.encoder4(x))
        encoder = self.pool(x)
        
        x = F.relu(self.decoder1(encoder))
        x = self.up1(x)
        x = F.relu(self.decoder2(x))
        x = self.up(x)
        x = F.relu(self.decoder3(x))
        x = self.up(x)
        x = F.relu(self.decoder4(x))
        x = self.up(x)
        decoder = self.decoder5(x)
        
        return decoder