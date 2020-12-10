
import torch
import torch.nn as nn
import torch.nn.functional as F

class myVGG(nn.Module):
    
    def __init__(self, class_num=10, init_weights=True):
        super(myVGG, self).__init__()
        self.class_num = class_num
        
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode = True) # 결과를 반올림하여 size가 0이 되는 것을 방지
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.dropout = nn.Dropout2d(p=0.5)
        
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.class_num)
        
        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        skip = self.maxpool(x) # skip connection 구조를 만들어주기 위해 결과를 따로 저장
        
        x = F.relu(self.conv2_1(skip))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool(x)
        
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.maxpool(x)
        
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.maxpool(x)        
        
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.maxpool(x)
        
        x = self.avgpool(x)        
        x = torch.flatten(x, 1)
        
        skip_flatten = torch.flatten(skip, 1) # 첫 번째 Dense의 입력과 사이즈가 달라 강제로 맞춰주기 위해 
        skip_input = x + skip_flatten.repeat(1,2) # Conv2_1 layer의 input값의 size을 2배로 확장한 후 결합
        
        x = F.relu(self.fc1(skip_input))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        out = self.fc3(x)
        
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
