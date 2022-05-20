import numpy as np
import torch
import torch.nn as nn
import os
import time
from datasetloader import SingleImgDataset
from model import ExMeshCNN

def buildModel(num_classes):
    return ExMeshCNN(num_classes = num_classes)

root_dir = './dataset'
class_names = sorted(os.listdir(root_dir)) 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 0.003
training_epochs = 250
batch_size = 64

model = buildModel(len(class_names)).to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = SingleImgDataset(root_dir, 'train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_dataset = SingleImgDataset(root_dir, 'test')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

total_batch = len(train_loader)
acc_list = []
acc_max = 0
for epoch in range(training_epochs):
    start = time.time()
    model.train()
    avg_cost = 0
    acc_train = []
    for data in train_loader:
        Y = data[0].to(device)
        X1 = data[1].to(device)
        X2 = data[2].to(device)
        X3 = data[3].to(device)

        optimizer.zero_grad()
        hypothesis = model(X1, X2, X3)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        correct_prediction = torch.argmax(hypothesis, 1) == Y
        acc_train.append(correct_prediction.float().mean().item())
        avg_cost += cost.item() / total_batch

    acc_val = []
    with torch.no_grad():
        model.eval()
        for data_ in val_loader:
            Y_ = data_[0].to(device)
            X1_ = data_[1].to(device)
            X2_ = data_[2].to(device)
            X3_ = data_[3].to(device)

            hypothesis_ = model(X1_, X2_, X3_)
            correct_prediction = torch.argmax(hypothesis_, 1) == Y_
            acc_val.append(correct_prediction.float().mean().item())
        acc_list.append(np.mean(acc_val))
    acc_train = np.mean(acc_train)
    acc_val = np.mean(acc_val)
    if acc_max < np.max(acc_list):
        acc_max = np.max(acc_list)
        torch.save(model.state_dict(), "./model_save.pt")
    print('[Epoch:{:>3}] acc= {:>.3} val_acc= {:>.3} max= {:>.3} time= {:>.3}'.format(epoch,acc_train,acc_val,acc_max,time.time()-start))




