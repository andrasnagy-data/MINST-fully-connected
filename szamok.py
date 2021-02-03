import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# hyper parameters
input_size = 784
hidden_size = 16
n_classes = 10
n_epochs = 15
batch_size = 100
learning_rate = 0.001


# data sets
train_dataset = datasets.MNIST(root= './data', train= True, 
                               transform= transforms.ToTensor(), 
                               download= True)

test_dataset = datasets.MNIST(root= './data', train= False, 
                              transform= transforms.ToTensor(), 
                              download= True)


# data loaders
train_loader = torch.utils.data.DataLoader(dataset= train_dataset, 
                                           batch_size= batch_size, 
                                           shuffle= True)

test_loader = torch.utils.data.DataLoader(dataset= test_dataset, 
                                           batch_size= batch_size, 
                                           shuffle= False)


# Fully connected with 2 hidden layers
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, n_classes)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        return self.l3(x)

# neural network instance
model = NeuralNet(input_size, hidden_size, n_classes).to(device)


# criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)


# training loop
train_acc = []
test_acc = []
train_loss = []
test_loss = []

for epoch in range(n_epochs):
    # training
    model.train()
    
    total_loss, n_correct, n_samples = 0.0, 0, 0
    for batch_i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        y_ = model(images)
        loss = criterion(y_, labels)
        
        loss.backward()
        optimizer.step()

        _, y_label_ = torch.max(y_, 1)
        n_correct += (y_label_ == labels).sum().item()
        total_loss += loss.item() * images.shape[0]
        n_samples += images.shape[0]
    
    print(
        f"Epoch {epoch+1}/{n_epochs} |"
        f"  train loss: {total_loss / n_samples:9.3f} |"
        f"  train acc:  {n_correct / n_samples * 100:9.3f}%"
    )

    # append accuracy & loss to lists
    train_acc.append(n_correct / n_samples * 100)
    train_loss.append(total_loss / n_samples)

    # evaluation
    model.eval()
    
    total_loss, n_correct, n_samples = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
                    
            y_ = model(images)
        
            _, y_label_ = torch.max(y_, 1)
            n_correct += (y_label_ == labels).sum().item()
            loss = criterion(y_, labels)
            total_loss += loss.item() * images.shape[0]
            n_samples += images.shape[0]

    print(
        f"Epoch {epoch+1}/{n_epochs} |"
        f"  valid loss: {total_loss / n_samples:9.3f} |"
        f"  valid acc:  {n_correct / n_samples * 100:9.3f}%"
    )

    # append accuracy & loss to lists
    test_acc.append(n_correct / n_samples * 100)
    test_loss.append(total_loss / n_samples)


# visualize training and testing statistics
X_values = np.arange(n_epochs)

fig = plt.figure(constrained_layout= True)
gs = gridspec.GridSpec(1, 2, figure= fig)

ax = fig.add_subplot(gs[0, 0])
ax.plot(X_values, train_acc, label= 'training')
ax.plot(X_values, test_acc, label= 'testing')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
plt.title('Training & Testing accuracy')
ax.legend()

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(X_values, train_loss, label= 'training')
ax2.plot(X_values, test_loss, label= 'testing')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
plt.title('Training & Testing loss')
ax2.legend()

plt.show()

name = 'NN_eval.png'
fig.savefig(name)