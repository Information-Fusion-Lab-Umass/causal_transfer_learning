import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, args):
        super(DQN, self).__init__()
        self.action_space = args.action_space
        self.hidden_size = args.hidden_size
        self.input_size = args.input_size
        # self.w = args.w
        # self.h = args.h
        # self.input_channels = args.input_channels
        # self.p = 0

        # self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)
        #
        # self.convw = self.conv_size_out(self.conv_size_out(self.conv_size_out(self.w)))
        # self.convh = self.conv_size_out(self.conv_size_out(self.conv_size_out(self.h)))
        # self.conv_output_size = int(self.convw) * int(self.convh) * 32

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.action_space)

    def conv_size_out(self, w, k = 5, s = 2):
        return ((w - k+(2*self.p))/s) + 1

    def forward(self, x, trial = 1, log=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
