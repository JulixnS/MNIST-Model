import random

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from model import MNISTmodel

train_data = pd.read_csv("data/mnist_train.csv")
test_data = pd.read_csv("data/mnist_test.csv")

features = train_data.columns[1:]

X = torch.tensor(train_data[features].values, dtype=torch.float)               # Inputs (pixels of each image)
y = torch.tensor(train_data["label"].values, dtype=torch.long)                 # What we're trying to predict (the number in each picture)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size= 0.80,
    test_size= 0.20,
    random_state= 42
)

# Training Loop
model = MNISTmodel()
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

train_loss_values = []
train_acc_values = []

test_loss_values = []
test_acc_values = []

num_epochs = 50
for epoch in range(num_epochs):
    predictions = model(X_train)
    BCELoss = loss(predictions, y_train)
    BCELoss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    train_loss_values.append(BCELoss.item())
    pred_label = torch.argmax(predictions, dim= 1)
    acc = accuracy_score(pred_label.numpy(), y_train.numpy())
    train_acc_values.append(acc)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch[{epoch + 1}/{num_epochs}] Loss: {BCELoss.item()}")


    
    #Testing during epochs to check for overfitting
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        BCELoss = loss(predictions, y_test)
        test_loss_values.append(BCELoss.item())
        predicted_labels = torch.argmax(predictions, dim= 1)
        acc = accuracy_score(predicted_labels.numpy(), y_test.numpy())
        test_acc_values.append(acc)



# Plot loss
plt.plot(range(1, num_epochs + 1), train_loss_values, label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_loss_values, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.show()

# Plot accuracy       Both training and testing accuracy increases, 
plt.plot(range(1, num_epochs + 1), train_acc_values, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), test_acc_values, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.show()

random_num = random.randint(0, len(test_data) - 1 )

random_image = test_data.iloc[random_num].drop("label")
x = torch.tensor(random_image, dtype=torch.float).unsqueeze(0)
y = test_data.iloc[random_num]["label"]

model.eval()
with torch.no_grad():
    output = model(x)
    predicted_label = torch.argmax(output, dim=1).item()

print(f"Prediction: {predicted_label}")
print(f"Actual: {y}")




