import torch
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import logging
from codes.utils import analyze
import math

torch.manual_seed(0)

class RelationalNN(nn.Module):
    def __init__(self, x_dim, h_dim, y_dim, c_dict, linear_flag = False, sparse = 0, group_lasso = 0, n_colors = 2, penalty = 1):
        super(RelationalNN, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.linear_flag = linear_flag
        self.sparse = sparse
        self.group_lasso = group_lasso
        self.n_colors = n_colors
        self.penalty = penalty

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
        mse = nn.MSELoss(reduction = "mean")
        # l2_crit = nn.L2Loss(size_average=False)
        reg_loss = 0
        group_lasso_loss = 0
        mse_loss = mse(y_pred, y)
        if self.group_lasso == 1:
            for p in self.parameters():
                # print("hello")
                # print(p.size(), p)
                for k in range(2):
                    idx = 0
                    for i in range(5):
                        # print(k, idx)
                        # group_lasso_loss +=  abs(p[k, idx + 0])
                        # group_lasso_loss +=  abs(p[k, idx + 1])
                        # # print(p[k, idx + 2: idx + 2+ self.n_colors])
                        group_lasso_loss +=  math.sqrt(1) * torch.norm(p[k,idx: idx + 2 + self.n_colors], p = 2, dim = 0)
                        idx = idx + 2 + self.n_colors
                    group_lasso_loss += math.sqrt(1)* torch.norm(p[k,-4:], p = 2, dim = 0)

        if self.sparse == 1:
            for param in self.parameters():
                reg_loss += torch.sum(torch.abs(param))

        # print("Group lasso flag {} loss {}".format(self.group_lasso, group_lasso_loss))
        # print("Lasso flag {} loss {}".format(self.sparse, reg_loss))
        # print("MSE loss {}".format(mse_loss))

        return  mse_loss + self.penalty * reg_loss + self.penalty * group_lasso_loss

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
