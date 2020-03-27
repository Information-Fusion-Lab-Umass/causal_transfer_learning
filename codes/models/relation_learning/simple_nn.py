import torch
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import logging
from codes.utils import analyze

torch.manual_seed(0)

class RelationalNN(nn.Module):
    def __init__(self, x_dim, h_dim, y_dim, c_dict, linear_flag = False, sparse = 1):
        super(RelationalNN, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.linear_flag = linear_flag
        self.sparse = sparse

        if self.linear_flag:
            self.fcl = nn.Linear(x_dim, y_dim, bias = False)
        else:
            self.fc1 = nn.Linear(x_dim, h_dim, bias = False)
            self.fc2 = nn.Linear(h_dim, y_dim, bias = False)
        self.optimizer = optim.Adam(self.parameters(), lr = 1e-2, weight_decay = 1e-4)

        # logging attributes
        format = "%(asctime)s.%(msecs)03d: - %(levelname)s: %(message)s"
        logging.basicConfig(format=format, level=logging.INFO,
        datefmt="%H:%M:%S")
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter()

        # to analyze input
        self.c_dict = c_dict

    def forward(self, x):
        if self.linear_flag:
            return self.fcl(x)
        else:
            x1 = F.relu(self.fc1(x))
            y = self.fc2(x1)
            return y

    def loss(self, y_pred, y):
        mse_loss = nn.MSELoss(reduction = "mean")
        reg_loss = 0
        for param in self.parameters():
            reg_loss += torch.sum(torch.abs(param))
        return mse_loss(y_pred, y) + self.sparse * reg_loss

    def train(self, epoch, xtr, ytr):
        # self.logger.info("========================== Train =====================")
        self.optimizer.zero_grad()
        ytr_pred = self.forward(xtr)
        train_loss = self.loss(ytr_pred, ytr)

        train_loss.backward()
        self.writer.add_scalar('Loss/train', train_loss.item(), epoch)
        # self.logger.info('====> Epoch {}, Train set loss: {:.4f}'.format(epoch, train_loss.item()))
        self.optimizer.step()

    def test(self, epoch, xte, yte):
        # self.logger.info("========================== Test =====================")
        with torch.no_grad():
            yte_pred = self.forward(xte)
            test_loss = self.loss(yte_pred, yte)
            self.writer.add_scalar('Loss/test', test_loss.item(), epoch)
            # print(analyze(xte[:5], self.c_dict), yte[:5], yte_pred[:5])
            # self.logger.info('====> Epoch {}, Test set loss: {:.4f}'.format(epoch, test_loss.item()))
