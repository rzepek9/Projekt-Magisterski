import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from workspace.pose_3d.models.model import LSTM
from workspace.pose_3d.dataset import KeypointsMatrix

training_set = KeypointsMatrix('/home/s175668/raid/Praca-Magisterska/dataset/folds/made/train_made_balance.csv', aug=True)
test_set = KeypointsMatrix('/home/s175668/raid/Praca-Magisterska/dataset/folds/made/val_made_balance.csv')


train_data = DataLoader(training_set, batch_size=32, shuffle=True)
test_data = DataLoader(test_set, batch_size=32, shuffle=True)

# last_layers=2, droput_value=0.1, hide_state=8, num_layers=2, dropout_dense=0.1
model = LSTM(2, 0.1, 24, 2, 0.2)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
torch.manual_seed(0)

loss_fn = torch.nn.BCELoss()

# Training loop
for epoch in range(150):
    model.train()
    running_loss = 0.
    correct = 0
    total = 0
    
    for data in train_data:
        # Every data instance is an input + label pair
        inputs, labels = data
        
        # labels = F.one_hot(labels.long(), -1)
        
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
    train_acu =100.*correct/total
    
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
    val_acu =100.*correct/total

    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss} Train Accuracy: {train_acu}, Validation Accuracy: {val_acu}")



