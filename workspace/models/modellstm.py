import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.sigmoid = nn.Sigmoid()
        
        self.conv_1 = nn.Conv1d(24, 32, 3, stride=1, padding='same')
        self.conv_2 = nn.Conv1d(32, 64, 3, stride=1, padding='same')
       
        self.lstm = nn.LSTM(input_size=64, hidden_size=256,
                          num_layers=2, batch_first=True) #lstm
        
        self.pool = nn.MaxPool1d(3, padding=1)

        self.linear = nn.Linear(11*256, 768)
        self.out = nn.Linear (768,1)
    
    def forward(self,x):

        x = x.permute(0,2,1)

        x = self.conv_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.permute(0,2,1)

        # x = self.pool(x)
        x , _=  self.lstm(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        out = self.sigmoid(x)

        return out

class CONV_2D(nn.Module):
    def __init__(self):
        super(CONV_2D, self).__init__()
        

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()
        
        self.conv_1 = nn.Conv2d(1, 32, (2,2), stride=1, padding='same')
        self.conv_2 = nn.Conv2d(32, 64, (4,4), stride=1, padding='same')
        self.conv_3 = nn.Conv2d(64, 128, (4,4), stride=1, padding='same')
        self.conv_4 = nn.Conv2d(128,128, (4,4), stride=2)
       
        self.lstm = nn.LSTM(input_size=52, hidden_size=128,
                          num_layers=2, batch_first=True) #lstm
        
        self.pool = nn.MaxPool2d((3,3), padding=1)

        self.linear = nn.Linear(384, 1024)
        self.linear_2 = nn.Linear (1024,512)
        self.out = nn.Linear(512,1)
        self.dropout_linear = nn.Dropout(0.5)
    
    def forward(self,x):
        x = torch.unsqueeze(x, dim=1)
        x = self.relu(self.conv_1(x))
        x = self.dropout(x)
        x = self.relu(self.conv_2(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = self.relu(self.conv_3(x))
        x = self.dropout(x)
        x = self.relu(self.conv_4(x))
        x = self.dropout(x)
        x = self.pool(x)
        print(x.size())
        x = self.flatten(x)
        print(x.size())
        x = self.relu(self.linear(x))
        print(x.size())
        x= self.dropout_linear(x)
        print(x.size())
        x = self.relu(self.linear_2(x))
        print(x.size())
        x = self.dropout_linear(x)
        print(x.size())
        out = self.sigmoid(self.out(x))

        return out