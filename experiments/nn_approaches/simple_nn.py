import torch
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F

class RelationalNN(nn.Module):
    def __init__(self, x_dim, h_dim, y_dim):
        super(RelationalNN, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, y_dim)
        self.optimizer = optim.Adam(self.parameters(), lr = 1e-2, weight_decay = 1e-4)


    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        y = self.fc2(x1)
        return y

    def loss(self, y_pred, y):
        mse_loss = nn.MSELoss(reduction = "mean")
        return mse_loss(y_pred, y)

    def train(self, epoch, xtr, ytr):
        print("========================== Train =====================")

        self.optimizer.zero_grad()
        ytr_pred = self.forward(xtr)
        train_loss = self.loss(ytr_pred, ytr)

        train_loss.backward()

        print('====> Epoch {}, Train set loss: {:.4f}'.format(epoch, train_loss.item()))
        self.optimizer.step()

    def test(self, epoch, xte, yte):
        print("========================== Test =====================")

        with torch.no_grad():
            yte_pred = self.forward(xte)
            print(xte[:5], yte_pred[:5], yte[:5])
            test_loss = self.loss(yte_pred, yte)
            print('====> Epoch {}, Test set loss: {:.4f}'.format(epoch, test_loss.item()))
