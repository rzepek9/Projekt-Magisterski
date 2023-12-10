import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

"""
LSTM MODEL for input (batch, features, frame)
conv1d przechodzi po klatkach zachowujac ich ilosc
przed wejsciem na lstm dane sa transponowane na (batch, frame, features)
"""


class LSTM(nn.Module):
    def __init__(self, last_layers=2, droput_value=0.1, hide_state=8, num_layers=2, dropout_dense=0.1):
        super(LSTM, self).__init__()

        self.last_layers = last_layers

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout_dense)
        self.sigmoid = nn.Sigmoid()

    
        self.input = 18 * hide_state
        self.output = int(18 * hide_state / 2)

        self.lstm = nn.LSTM(input_size=3*12, hidden_size=hide_state, dropout=droput_value,
                            num_layers=num_layers, batch_first=True)  # lstm
        
        self.conv1_d = nn.Conv1d(36, 36, 5, padding='same')

        self.linear_1 = nn.Linear(self.input, self.output)
        self.linear_2 = nn.Linear(self.output, 1)
        
        self.linear_out = nn.Linear(self.input, 1)

        

    def forward(self, x):
        
        # x = x.permute(0, 2, 1)
        # x = self.leaky_relu(self.conv1_d(x))
        # x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)
        x = self.flatten(x)
        
        if self.last_layers == 2:
            x = self.relu(self.linear_1(x))
            x = self.dropout(x)
            out = self.sigmoid(self.linear_2(x))
        else:
            out = self.sigmoid(self.linear_out(x))
            

        return out
    
    

class LSTM_skill(nn.Module):
    def __init__(self, last_layers=2, droput_value=0.1, hide_state=8, num_layers=2, dropout_dense=0.1):
        super(LSTM_skill, self).__init__()

        self.last_layers = last_layers

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_dense)
        self.softmax = nn.Softmax(dim=1)

        self.input = 18 * hide_state
        self.output = int(18 * hide_state / 2)

        self.lstm = nn.LSTM(input_size=3*12, hidden_size=hide_state, dropout=droput_value,
                            num_layers=num_layers, batch_first=True)  # lstm

        self.linear_1 = nn.Linear(self.input, self.output)
        self.linear_2 = nn.Linear(self.output, 3)
        
        self.linear_out = nn.Linear(self.input, 3)

        

    def forward(self, x):

        x, _ = self.lstm(x)
        x = self.flatten(x)
        
        if self.last_layers == 2:
            x=self.dropout(x)
            x = self.relu(self.linear_1(x))
            x = self.dropout(x)
            out = self.softmax(self.linear_2(x))
        else:
            x=self.dropout(x)

            out = self.softmax(self.linear_out(x))
            

        return out
    

class MLP(nn.Module):
    def __init__(self, dropout_dense, last_layer):
        super(MLP, self).__init__()


        self.last_layer = last_layer
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_dense)
        self.sigmoid = nn.Sigmoid()

        self.input = 18*12*3
        self.linear_1 = nn.Linear(self.input, 512)
        self.linear_2 = nn.Linear(512, 256)
        self.linear_3 = nn.Linear(256, 128)
        self.linear_out = nn.Linear(256, 1)
        self.linear_out_2 = nn.Linear(128, 1)

        

    def forward(self, x):

        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.relu(self.linear_2(x))
        x = self.dropout(x)
        if self.last_layer > 1:
            x = self.relu(self.linear_3(x))
            x = self.relu(self.linear_out_2(x))
        else:
        # x = self.dropout(x)
            x = self.relu(self.linear_out(x))
        
        out = self.sigmoid(x)

        return out
