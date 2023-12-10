import os
import wandb


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from workspace.pose_3d.dataset import KeypointsMatrix
from workspace.pose_3d.models.model import LSTM, MLP


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
    "name": "001_model_cg_save_relu",
    "description": "Trening modelu LSTM, pierwszą warstwę modelu stanowi conv1d (24, 64, 3 stride=1, padding='same'), \
         Nastepnie warstwa LSTM input_size=64, hidden_size=128, num_layers=2, batch_first=True ",
    'method': 'grid',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters': {
        'batch_size': {'values': [64]},
        'droput_value': {'values': [0.2]},
        'hide_state': {'values': [12]},
        'num_layers': {'values': [4]},
        'dropout_dense': {'values': [0.2]},
        'camera': {'values': [None]},
        # 'height': {'values': [True]},
        # 'min_max': {'values':  [False]},
        # 'cords': {'values': [True, False]},
        'wight_decay':{'values': [0]},
        
        'aug': {'values': [0, 0.1]},
        'last_layer': {'values': [2]},
        'learning_rate': {'values': [0.001]},
    }
}


sweep_id = wandb.sweep(sweep_config, project='001_model_cg_save_relu')

print(sweep_id)


def train(config=None):

    with wandb.init(project='001_model_cg_save_relu', entity='Rzepek', config=config, id='esssaa12'):
        config = wandb.config
        
        # height, cords, min_max = config.height, config.cords, config.min_max
        
        training_set = KeypointsMatrix('/home/s175668/raid/Praca-Magisterska/dataset/folds/made/train_made_balance.csv', camera=config.camera, 
                                       height_normalize=False, cordinates_move=False, 
                                       signal_normalize_0_1=False, aug=config.aug, cg=True)
        
        test_set = KeypointsMatrix('/home/s175668/raid/Praca-Magisterska/dataset/folds/made/val_made_balance.csv', camera=config.camera, 
                                       height_normalize=False, cordinates_move=False, 
                                       signal_normalize_0_1=False, aug=False, cg=True)

        train_data = DataLoader(training_set, batch_size=config.batch_size, shuffle=True)
        test_data = DataLoader(test_set, batch_size=config.batch_size, shuffle=True)

        model = LSTM(config.last_layer, config.droput_value, config.hide_state, config.num_layers, config.dropout_dense)
        # model = MLP(config.droput_value, config.last_layer)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        loss_fn = torch.nn.BCELoss()

        if config.wight_decay == 0:
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        else:
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.wight_decay)

        wandb.watch(model, loss_fn, log="all")
        best_val = 1
        for epoch in range(150):
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
            if val_loss < best_val:
                if config.camera is None:
                    camera = 'both'
                else:
                    camera = 'side'
                model_name = f'001_model_{str(config.aug)}_{camera}_relu'
                path = f'/home/s175668/raid/Praca-Magisterska/models_new/{model_name}.pt'
                torch.save(model.state_dict(), path)
                best_val = val_loss
                

            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss} Train Accuracy: {train_acu}, Validation Accuracy: {val_acu}")

            wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'Train Accuracy': train_acu,
                        'Validation Accuracy': val_acu})


wandb.agent(sweep_id, function=train, count=200, project='001_model_cg_save_relu')
