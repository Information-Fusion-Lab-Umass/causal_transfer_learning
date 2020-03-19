import torch
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import logging
from codes.utils import analyze

class RelationalNN(nn.Module):
    def __init__(self, x_dim, h_dim, y_dim, c_dict):
        super(RelationalNN, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, y_dim)
        self.optimizer = optim.Adam(self.parameters(), lr = 1e-2, weight_decay = 1e-4)
        format = "%(asctime)s.%(msecs)03d: - %(levelname)s: %(message)s"
        logging.basicConfig(format=format, level=logging.INFO,
        datefmt="%H:%M:%S")
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter()
        self.c_dict = c_dict

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        y = self.fc2(x1)
        return y

    def loss(self, y_pred, y):
        mse_loss = nn.MSELoss(reduction = "mean")
        return mse_loss(y_pred, y)

    def train(self, epoch, xtr, ytr):
        self.logger.info("========================== Train =====================")

        self.optimizer.zero_grad()
        ytr_pred = self.forward(xtr)
        train_loss = self.loss(ytr_pred, ytr)

        train_loss.backward()
        self.writer.add_scalar('Loss/train', train_loss.item(), epoch)
        self.logger.info('====> Epoch {}, Train set loss: {:.4f}'.format(epoch, train_loss.item()))
        self.optimizer.step()

    def test(self, epoch, xte, yte):
        self.logger.info("========================== Test =====================")

        with torch.no_grad():
            yte_pred = self.forward(xte)
            test_loss = self.loss(yte_pred, yte)
            self.writer.add_scalar('Loss/test', test_loss.item(), epoch)
            print(analyze(xte[:5], self.c_dict), yte[:5], yte_pred[:5])
            self.logger.info('====> Epoch {}, Test set loss: {:.4f}'.format(epoch, test_loss.item()))
