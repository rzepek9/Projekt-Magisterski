import os
import wandb

import torch.nn.functional as F

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from workspace.pose_3d.dataset import KeypointsMatrix
from workspace.pose_3d.models.model import LSTM_skill, MLP


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
    "name": "004_LSTM",
    "description": "Trening modelu LSTM, pierwszą warstwę modelu stanowi conv1d (24, 64, 3 stride=1, padding='same'), \
         Nastepnie warstwa LSTM input_size=64, hidden_size=128, num_layers=2, batch_first=True ",
    'method': 'grid',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters': {
        'batch_size': {'values': [32, 128]},
        'droput_value': {'values': [0.2]},
        'weight_decay': {'values': [0, 0.2, 0.4, 0.6, 0.8]},
        'aug': {'values': [False, True]},
        'camera': {'values': ['side', None]},
        'min_max': {'values': [False, True]},
        'height': {'values': [False, True]},
        'cords': {'values': [False, True]},
        # 'momentum': {'values':[0, 0.25, 0.5, 0.75]},
        'hide_state': {'values': [8, 12]},
        'num_layers': {'values': [1, 3, 4]},
        'dropout_dense': {'values': [0.1]},
        'last_layer': {'values': [1, 2]},
        'learning_rate': {'values': [0.001]},
    }

}

torch.manual_seed(0)

sweep_id = wandb.sweep(sweep_config, project='004_LSTM')

print(sweep_id)


def train(config=None):

    with wandb.init(project='004_LSTM', entity='Rzepek', config=config, id='esssaa12'):
        config = wandb.config

        height, cords, min_max = False, False, False
        training_set = KeypointsMatrix('/home/s175668/raid/Praca-Magisterska/dataset/folds/skill/004_train.csv', camera=config.camera, signal_normalize_0_1=config.min_max,
                                       cordinates_move=config.cords, height_normalize=config.height, aug=config.aug)
        
        test_set = KeypointsMatrix('/home/s175668/raid/Praca-Magisterska/dataset/folds/skill/004_train.csv',camera=config.camera, signal_normalize_0_1=config.min_max,
                                       cordinates_move=config.cords, height_normalize=config.height)


        train_data = DataLoader(training_set, batch_size=config.batch_size, shuffle=True)
        test_data = DataLoader(test_set, batch_size=config.batch_size, shuffle=True)

        # last_layers=2, droput_value=0.1, hide_state=8, num_layers=2, dropout_dense=0.1
        model = LSTM_skill(last_layers=config.last_layer, droput_value=config.droput_value, hide_state=config.hide_state, num_layers=config.num_layers, dropout_dense=config.dropout_dense)
        # model = MLP(0.3)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        if config.weight_decay > 0:
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        loss_fn = torch.nn.CrossEntropyLoss()
        torch.manual_seed(0)
        first_epoch = 0

        wandb.watch(model, loss_fn, log="all")

        for epoch in range(40):
            model.train()
            running_loss = 0.
            correct = 0
            total = 0
            
            for data in train_data:
                # Every data instance is an input + label pair
                inputs, labels = data
                
                labels = F.one_hot(labels.long(), 3)
                labels = labels.to(device)
                
                inputs = inputs.to(device)

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = model(inputs)

                # Compute the loss and its gradients
                loss = loss_fn(outputs, labels.float())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                correct += torch.argmax(outputs, dim=1).eq(labels.argmax(dim=1)).sum()
                total += labels.size(0)

            train_loss = running_loss/len(train_data)
            train_acu =100.*correct/total
            
            model.eval()
            running_loss = 0
            correct = 0
            total = 0
            
            for data in test_data:
                inputs, labels = data
                
                labels = F.one_hot(labels.long(), 3)
                labels = labels.to(device)
                
                inputs = inputs.to(device)

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = model(inputs)

                # Compute the loss and its gradients
                loss = loss_fn(outputs, labels.float())
                running_loss += loss.item()

                correct += torch.argmax(outputs, dim=1).eq(labels.argmax(dim=1)).sum()
                total += labels.size(0)
            
            val_loss = running_loss/len(test_data)
            val_acu =100.*correct/total
            
            if val_acu > 0.8 and first_epoch == 0:
                first_epoch = epoch


            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss} Train Accuracy: {train_acu}, Validation Accuracy: {val_acu}")
            
            wandb.log({'training loss': train_loss, 'val_loss': val_loss, 'Train Accuracy': train_acu,
                        'Validation Accuracy': val_acu, "ovf": first_epoch})

wandb.agent(sweep_id, function=train, count=400, project='004_LSTM')
