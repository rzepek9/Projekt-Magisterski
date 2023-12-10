import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from workspace.pose_3d.models.model import LSTM_skill, MLP
from workspace.pose_3d.dataset import KeypointsMatrix

training_set = KeypointsMatrix('/home/s175668/raid/Praca-Magisterska/dataset/folds/skill/004_train.csv', camera ='side', cg=True, signal_normalize_0_1=True)
test_set = KeypointsMatrix('/home/s175668/raid/Praca-Magisterska/dataset/folds/skill/004_train.csv',camera='side', cg=True, signal_normalize_0_1=True)


train_data = DataLoader(training_set, batch_size=16, shuffle=True)
test_data = DataLoader(test_set, batch_size=16, shuffle=True)

# last_layers=2, droput_value=0.1, hide_state=8, num_layers=2, dropout_dense=0.1
model = LSTM_skill(last_layers=2, droput_value=0.1, hide_state=8, num_layers=2, dropout_dense=0.1)
# model = MLP(0.3)

device = "cuda:3" if torch.cuda.is_available() else "cpu"
model = model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.RMSprop(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.3)

loss_fn = torch.nn.CrossEntropyLoss()
torch.manual_seed(0)

# Training loop
for epoch in range(50):
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

    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss} Train Accuracy: {train_acu}, Validation Accuracy: {val_acu}")



