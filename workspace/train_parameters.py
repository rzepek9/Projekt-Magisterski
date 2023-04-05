import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from models.modellstm import LSTM, CONV_2D
from dataset import FeatureDataset
from torch.utils.data import DataLoader





"""
Run this script as default, you can manage the values in sweep config
In last line you define how many iteration you want to run
In line 44 and 50 set the name of the project on your own
Results are on the https://wandb.ai/home
Login: s175668@student.pg.edu.pl
password: projektbadawczy!
"""


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

device = 'cuda:0'

sweep_config = {
                "name": "lstm_with_conv_by_frames",
                'method': 'random',
                'metric': {'goal': 'maximize', 'name': 'accu_val'},
                'parameters': {
                    'batch_size': {'values': [16,32,64]},
                    'learning_rate': {'distribution': 'uniform',
                                      'max': 0.001,
                                      'min': 0.0001},
                    'optimizer': {'values': ['adam', 'sgd']},
                }
}



sweep_id = wandb.sweep(sweep_config, project='lstm_with_conv_by_frames')

print(sweep_id)

def train(config = None):
    # min_valid_loss = np.inf
    with wandb.init(project='lstm_with_conv_by_frames', entity='Rzepek', config=config, id = 'esssaa12'):
        config = wandb.config

        training_set = FeatureDataset('/home/s175668/raid/Praca-Magisterska/Repozytorium/Projekt-Magisterski/workspace/train/fold0')
        test_set = FeatureDataset('/home/s175668/raid/Praca-Magisterska/Repozytorium/Projekt-Magisterski/workspace/test/fold0')

        train_data = DataLoader(training_set, batch_size=config.batch_size, shuffle=True)
        test_data = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

        
        model = LSTM()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        loss_fn = torch.nn.BCELoss()

        if config.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum = 0.9)

        elif config.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        wandb.watch(model, loss_fn, log="all")
        for epoch in range(100):
            model.train()
            running_loss = 0.
            correct = 0
            total = 0

            for i, data in enumerate(train_data):
                # Every data instance is an input + label pair
                inputs, labels = data
                inputs = inputs.to(device)
                labels = torch.unsqueeze(labels, 1).to(device)
                print(inputs.size())


                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = model(inputs)

                # Compute the loss and its gradients
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                for i, _ in enumerate(outputs):
                    if outputs[i][0] < 0.5:
                        outputs[i][0] = 0.

                    else:
                        outputs[i][0] = 1.

                correct += outputs.eq(labels).sum()
                total += labels.size(0)


            train_loss = running_loss/len(train_data)
            train_acu =100.*correct/total
            
            model.eval()
            running_loss = 0
            correct = 0
            total = 0
            

            for i, data in enumerate(test_data):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = torch.unsqueeze(labels, 1).to(device)

                outputs = model(inputs.to(device))
                loss = loss_fn(outputs, labels)
                running_loss += loss

                for i, _ in enumerate(outputs):
                    if outputs[i][0] < 0.5:
                        outputs[i][0] = 0.

                    else:
                        outputs[i][0] = 1.
                        
                correct += outputs.eq(labels).sum()
                total += labels.size(0)

            val_loss = running_loss/len(test_data)
            val_acu =100.*correct/total

            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss} Train Accuracy: {train_acu}, Validation Accuracy: {val_acu}")
            wandb.log({'training loss': train_loss, 'validation loss': val_loss, 'Train Accuracy': train_acu,
                             'Validation Accuracy': val_acu})
            
            




wandb.agent(sweep_id, function=train, count=15, project='lstm_with_conv_by_frames')
               
