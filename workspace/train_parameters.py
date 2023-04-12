import os
import wandb

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.modellstm import LSTM, CONV_2D
from dataset import FeatureDataset


"""
Run this script as default, you can manage the values in sweep config
In last line you define how many iteration you want to run
In line 44 and 50 set the name of the project on your own
Results are on the https://wandb.ai/home
Login: s175668@student.pg.edu.pl
password: projektbadawczy!
"""

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

device = 'cuda:0'

sweep_config = {
    "name": "conv1d_1lstm_300_epoch",
    'method': 'random',
    'metric': {'goal': 'maximize', 'name': 'accu_val'},
    'parameters': {
        'batch_size': {'values': [4, 8, 16]},
        'droput_value': {'values': [0.1, 0.2, 0.3]},
        'learning_rate': {'distribution': 'uniform',
                          'max': 0.0004,
                          'min': 0.00005},
        'optimizer': {'values': ['adam', 'sgd']},
    }
}


sweep_id = wandb.sweep(sweep_config, project='conv1d_1lstm_300_epoch')

print(sweep_id)


def train(config=None):

    with wandb.init(project='conv1d_1lstm_300_epoch', entity='Rzepek', config=config, id='esssaa12'):
        config = wandb.config

        training_set = FeatureDataset('/home/s175668/raid/Praca-Magisterska/Repozytorium/Projekt-Magisterski/workspace/train/fold0')
        test_set = FeatureDataset('/home/s175668/raid/Praca-Magisterska/Repozytorium/Projekt-Magisterski/workspace/test/fold0')

        train_data = DataLoader(training_set, batch_size=config.batch_size, shuffle=True)
        test_data = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

        model = LSTM(config.droput_value)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        loss_fn = torch.nn.BCELoss()


        if config.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(),
                                  lr=config.learning_rate, momentum=0.9)

        elif config.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        wandb.watch(model, loss_fn, log="all")

        for epoch in range(300):
            model.train()
            running_loss = 0.
            correct = 0
            total = 0

            for data in train_data:
                # Every data instance is an input + label pair
                inputs, labels = data
                inputs = inputs.to(device)
                labels = torch.unsqueeze(labels, 1).to(device)

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
            train_acu = 100.*correct/total

            model.eval()
            running_loss = 0
            correct = 0
            total = 0

            for data in test_data:
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
            val_acu = 100.*correct/total

            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss} Train Accuracy: {train_acu}, Validation Accuracy: {val_acu}")

            wandb.log({'training loss': train_loss, 'validation loss': val_loss, 'Train Accuracy': train_acu,
                        'Validation Accuracy': val_acu})


wandb.agent(sweep_id, function=train, count=20, project='conv1d_1lstm_300_epoch')
