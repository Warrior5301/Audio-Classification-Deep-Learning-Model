import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        #Convolutional Layer1
        self.conv1= nn.Conv2d(in_channels,out_channels,3,stride,padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        #Convolutional Layer2
        self.conv2= nn.Conv2d(out_channels,out_channels,3,padding=1, bias=False) 
        self.bn2=nn.BatchNorm2d(out_channels)
        self.shortcut=nn.Sequential()
        self.use_shortcut=stride!=1 or in_channels != out_channels
        if self.use_shortcut:
            self.shortcut=nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride,bias=False),nn.BatchNorm2d(out_channels))
    
    def forward(self,x):
        out=self.conv1(x)
        out=self.bn1(out)
        out = nn.functional.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        shortcut=self.shortcut(x) if self.use_shortcut else x
        out_add=out+shortcut
        out=nn.functional.relu(out_add)
        
        return out

class AudioCNN(nn.Module):
    def __init__(self,num_classes=50):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1=nn.ModuleList([ResidualBlock(64,64) for i in range(2)])
        self.layer2=nn.ModuleList([
            ResidualBlock(64, 128, stride=2),
            *[ResidualBlock(128, 128) for _ in range(3)]])
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.dropout=nn.Dropout(0.3)
        self.fc= nn.Linear(128, num_classes)
        
    def forward(self ,x):
        x=self.conv1(x)
        for block in self.layer1:
            x=block(x)
        for block in self.layer2:
            x=block(x)
        x=self.avgpool(x)
        x=x.view(x.size(0), -1)
        x=self.dropout(x)
        x=self.fc(x)
        return x