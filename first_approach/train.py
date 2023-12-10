import torch

from torch.utils.data import DataLoader

from models.modellstm import LSTM, CONV_2D
from dataset import FeatureDataset, VectorLength

training_set = VectorLength('/home/s175668/raid/Praca-Magisterska/csv_extracted/th_60_kp_with_conf/fold_1_train.csv', return_kp=True)
test_set = VectorLength('/home/s175668/raid/Praca-Magisterska/csv_extracted/th_60_kp_with_conf/fold_1_val.csv', return_kp=True)

train_data = DataLoader(training_set, batch_size=128, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

model = LSTM(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
loss_fn = torch.nn.BCELoss()

# Training loop
for epoch in range(50):
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
    train_acu =100.*correct/total
    
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    
    for data in test_data:
        inputs, labels, samples = data
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



